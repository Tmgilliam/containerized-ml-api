"""Batch prediction pipeline orchestration."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from batch.io.readers import BaseReader
from batch.io.writers import BaseWriter

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch prediction pipeline."""
    batch_size: int = 1000
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 300.0
    continue_on_error: bool = True
    progress_interval: int = 10000


@dataclass
class BatchResult:
    """Results from a batch prediction run."""
    start_time: str
    end_time: str
    duration_seconds: float
    total_records: int
    successful_predictions: int
    failed_predictions: int
    error_records: list[dict[str, Any]] = field(default_factory=list)
    model_version: str = "unknown"
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.successful_predictions / self.total_records
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_records": self.total_records,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "success_rate": round(self.success_rate, 4),
            "model_version": self.model_version,
            "error_count": len(self.error_records),
        }


class BatchPipeline:
    """
    Orchestrates batch prediction jobs.
    
    Features:
    - Parallel processing with configurable workers
    - Retry logic for transient failures
    - Progress tracking and logging
    - Error collection and reporting
    """
    
    def __init__(
        self,
        predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
        config: BatchConfig | None = None,
    ) -> None:
        """
        Initialize batch pipeline.
        
        Args:
            predict_fn: Function that takes features dict and returns prediction dict
            config: Pipeline configuration
        """
        self.predict_fn = predict_fn
        self.config = config or BatchConfig()
        self._model_version = "unknown"
    
    def set_model_version(self, version: str) -> None:
        """Set the model version for result tracking."""
        self._model_version = version
    
    def _predict_with_retry(
        self,
        record: dict[str, Any],
        record_id: Any,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Execute prediction with retry logic.
        
        Returns tuple of (result, error_message).
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                result = self.predict_fn(record)
                return result, None
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                    logger.warning(
                        "Retry %d/%d for record %s: %s",
                        attempt + 1,
                        self.config.retry_attempts,
                        record_id,
                        e,
                    )
        
        return None, last_error
    
    def _process_batch(
        self,
        batch: list[tuple[Any, dict[str, Any]]],
    ) -> list[tuple[Any, dict[str, Any] | None, str | None]]:
        """Process a batch of records."""
        results = []
        
        for record_id, record in batch:
            prediction, error = self._predict_with_retry(record, record_id)
            results.append((record_id, prediction, error))
        
        return results
    
    def run(
        self,
        reader: BaseReader,
        writer: BaseWriter,
        id_field: str | None = None,
        feature_fields: list[str] | None = None,
        include_input: bool = True,
    ) -> BatchResult:
        """
        Execute batch prediction pipeline.
        
        Args:
            reader: Data reader instance
            writer: Data writer instance
            id_field: Field name for record ID (for error tracking)
            feature_fields: Subset of fields to use as features (None = all)
            include_input: Whether to include input features in output
        
        Returns:
            BatchResult with execution statistics
        """
        start_time = datetime.now(timezone.utc)
        start_timestamp = start_time.isoformat()
        
        logger.info("Starting batch prediction pipeline")
        
        total_records = 0
        successful = 0
        failed = 0
        error_records: list[dict[str, Any]] = []
        
        batch: list[tuple[Any, dict[str, Any]]] = []
        processed = 0
        
        try:
            for idx, record in enumerate(reader.read()):
                record_id = record.get(id_field, idx) if id_field else idx
                
                if feature_fields:
                    features = {k: record[k] for k in feature_fields if k in record}
                else:
                    features = {k: v for k, v in record.items() if k != id_field}
                
                batch.append((record_id, features))
                total_records += 1
                
                if len(batch) >= self.config.batch_size:
                    results = self._process_batch(batch)
                    
                    output_records = []
                    for rec_id, prediction, error in results:
                        if prediction:
                            output = {"_id": rec_id}
                            if include_input:
                                orig_idx = next(
                                    i for i, (rid, _) in enumerate(batch)
                                    if rid == rec_id
                                )
                                output.update(batch[orig_idx][1])
                            output.update(prediction)
                            output["_predicted_at"] = datetime.now(timezone.utc).isoformat()
                            output_records.append(output)
                            successful += 1
                        else:
                            failed += 1
                            if error:
                                error_records.append({
                                    "record_id": rec_id,
                                    "error": error,
                                })
                    
                    if output_records:
                        writer.write(output_records)
                    
                    batch.clear()
                    processed += len(results)
                    
                    if processed % self.config.progress_interval == 0:
                        logger.info(
                            "Progress: %d records processed (%d successful, %d failed)",
                            processed,
                            successful,
                            failed,
                        )
            
            if batch:
                results = self._process_batch(batch)
                
                output_records = []
                for rec_id, prediction, error in results:
                    if prediction:
                        output = {"_id": rec_id}
                        if include_input:
                            orig_idx = next(
                                i for i, (rid, _) in enumerate(batch)
                                if rid == rec_id
                            )
                            output.update(batch[orig_idx][1])
                        output.update(prediction)
                        output["_predicted_at"] = datetime.now(timezone.utc).isoformat()
                        output_records.append(output)
                        successful += 1
                    else:
                        failed += 1
                        if error:
                            error_records.append({
                                "record_id": rec_id,
                                "error": error,
                            })
                
                if output_records:
                    writer.write(output_records)
        
        finally:
            writer.close()
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        result = BatchResult(
            start_time=start_timestamp,
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_records=total_records,
            successful_predictions=successful,
            failed_predictions=failed,
            error_records=error_records[:100],
            model_version=self._model_version,
        )
        
        logger.info(
            "Batch pipeline complete: %d records in %.2fs (%.1f rec/s), %d successful, %d failed",
            total_records,
            duration,
            total_records / duration if duration > 0 else 0,
            successful,
            failed,
        )
        
        return result
    
    def run_parallel(
        self,
        reader: BaseReader,
        writer: BaseWriter,
        id_field: str | None = None,
        feature_fields: list[str] | None = None,
        include_input: bool = True,
    ) -> BatchResult:
        """
        Execute batch prediction with parallel processing.
        
        Uses ThreadPoolExecutor for concurrent prediction execution.
        Best for I/O-bound prediction functions.
        """
        start_time = datetime.now(timezone.utc)
        start_timestamp = start_time.isoformat()
        
        logger.info(
            "Starting parallel batch pipeline with %d workers",
            self.config.max_workers,
        )
        
        total_records = 0
        successful = 0
        failed = 0
        error_records: list[dict[str, Any]] = []
        
        all_records: list[tuple[Any, dict[str, Any]]] = []
        
        for idx, record in enumerate(reader.read()):
            record_id = record.get(id_field, idx) if id_field else idx
            
            if feature_fields:
                features = {k: record[k] for k in feature_fields if k in record}
            else:
                features = {k: v for k, v in record.items() if k != id_field}
            
            all_records.append((record_id, features))
            total_records += 1
        
        output_records: list[dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._predict_with_retry, features, record_id): (record_id, features)
                for record_id, features in all_records
            }
            
            for future in as_completed(futures):
                record_id, features = futures[future]
                
                try:
                    prediction, error = future.result(timeout=self.config.timeout_seconds)
                    
                    if prediction:
                        output = {"_id": record_id}
                        if include_input:
                            output.update(features)
                        output.update(prediction)
                        output["_predicted_at"] = datetime.now(timezone.utc).isoformat()
                        output_records.append(output)
                        successful += 1
                    else:
                        failed += 1
                        if error:
                            error_records.append({
                                "record_id": record_id,
                                "error": error,
                            })
                
                except Exception as e:
                    failed += 1
                    error_records.append({
                        "record_id": record_id,
                        "error": str(e),
                    })
                
                processed = successful + failed
                if processed % self.config.progress_interval == 0:
                    logger.info(
                        "Progress: %d/%d records processed",
                        processed,
                        total_records,
                    )
        
        if output_records:
            writer.write(output_records)
        
        writer.close()
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        result = BatchResult(
            start_time=start_timestamp,
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_records=total_records,
            successful_predictions=successful,
            failed_predictions=failed,
            error_records=error_records[:100],
            model_version=self._model_version,
        )
        
        logger.info(
            "Parallel batch complete: %d records in %.2fs (%.1f rec/s)",
            total_records,
            duration,
            total_records / duration if duration > 0 else 0,
        )
        
        return result

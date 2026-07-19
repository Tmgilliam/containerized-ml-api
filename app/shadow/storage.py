"""Storage for shadow prediction results."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ShadowStorage:
    """
    Stores shadow prediction results for analysis.
    
    Features:
    - Local file storage (JSONL)
    - BigQuery support
    - Buffered writes
    - Query capabilities
    """
    
    def __init__(
        self,
        storage_backend: str = "local",
        storage_path: Path | None = None,
        bigquery_table: str | None = None,
        project_id: str | None = None,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize shadow storage.
        
        Args:
            storage_backend: 'local' or 'bigquery'
            storage_path: Local storage directory
            bigquery_table: BigQuery table for results
            project_id: GCP project ID
            buffer_size: Records to buffer before flush
        """
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else Path("./shadow_results")
        self.bigquery_table = bigquery_table
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.buffer_size = buffer_size
        
        self._buffer: list[dict[str, Any]] = []
        
        if storage_backend == "local":
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store(
        self,
        shadow_prediction: dict[str, Any],
    ) -> None:
        """
        Store a shadow prediction result.
        
        Args:
            shadow_prediction: ShadowPrediction.to_dict() output
        """
        self._buffer.append(shadow_prediction)
        
        if len(self._buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self) -> int:
        """
        Flush buffered predictions to storage.
        
        Returns:
            Number of records flushed
        """
        if not self._buffer:
            return 0
        
        count = len(self._buffer)
        
        if self.storage_backend == "local":
            self._flush_local()
        elif self.storage_backend == "bigquery":
            self._flush_bigquery()
        
        self._buffer.clear()
        logger.debug("Flushed %d shadow predictions", count)
        return count
    
    def _flush_local(self) -> None:
        """Flush to local storage."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"shadow_{timestamp}.jsonl"
        filepath = self.storage_path / filename
        
        with open(filepath, "w") as f:
            for record in self._buffer:
                f.write(json.dumps(record) + "\n")
    
    def _flush_bigquery(self) -> None:
        """Flush to BigQuery."""
        if not self.bigquery_table:
            logger.error("BigQuery table not configured")
            return
        
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client(project=self.project_id)
            errors = client.insert_rows_json(self.bigquery_table, self._buffer)
            
            if errors:
                logger.error("BigQuery insert errors: %s", errors)
                
        except ImportError:
            logger.error("google-cloud-bigquery not installed")
        except Exception as e:
            logger.error("Failed to flush to BigQuery: %s", e)
    
    def query_local(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        Query stored shadow predictions (local backend).
        
        Args:
            start_date: Filter predictions after this date
            end_date: Filter predictions before this date
            limit: Maximum records to return
        
        Returns:
            List of shadow prediction records
        """
        if self.storage_backend != "local":
            raise ValueError("query_local only works with local backend")
        
        records = []
        
        for filepath in sorted(self.storage_path.glob("shadow_*.jsonl")):
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        
                        ts = record.get("timestamp")
                        if ts:
                            record_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            
                            if start_date and record_dt < start_date:
                                continue
                            if end_date and record_dt > end_date:
                                continue
                        
                        records.append(record)
                        
                        if len(records) >= limit:
                            return records
        
        return records
    
    def get_comparison_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics from stored comparisons.
        
        Returns:
            Summary statistics dict
        """
        if self.storage_backend == "local":
            records = self.query_local(start_date, end_date)
        else:
            return {"error": "Query not supported for this backend"}
        
        if not records:
            return {"total": 0, "error": "No records found"}
        
        total = len(records)
        
        matches = sum(
            1 for r in records
            if r.get("comparison", {}).get("predictions_match", False)
        )
        
        prob_diffs = [
            r.get("comparison", {}).get("probability_difference", 0)
            for r in records
        ]
        
        prod_latencies = [r.get("production_latency_ms", 0) for r in records]
        shadow_latencies = [
            r.get("shadow_latency_ms", 0) for r in records
            if r.get("shadow_latency_ms") is not None
        ]
        
        errors = sum(1 for r in records if r.get("shadow_error"))
        
        import numpy as np
        
        return {
            "total_predictions": total,
            "agreement_rate": round(matches / total, 4) if total > 0 else 0,
            "error_rate": round(errors / total, 4) if total > 0 else 0,
            "probability_difference": {
                "mean": round(np.mean(prob_diffs), 4),
                "std": round(np.std(prob_diffs), 4),
                "max": round(max(prob_diffs), 4) if prob_diffs else 0,
                "p95": round(np.percentile(prob_diffs, 95), 4) if prob_diffs else 0,
            },
            "latency": {
                "production_mean_ms": round(np.mean(prod_latencies), 2),
                "shadow_mean_ms": round(np.mean(shadow_latencies), 2) if shadow_latencies else None,
            },
            "date_range": {
                "start": records[0].get("timestamp") if records else None,
                "end": records[-1].get("timestamp") if records else None,
            },
        }
    
    def delete_old_records(
        self,
        days_to_keep: int = 30,
    ) -> int:
        """
        Delete records older than specified days.
        
        Returns:
            Number of files deleted
        """
        if self.storage_backend != "local":
            return 0
        
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        deleted = 0
        
        for filepath in self.storage_path.glob("shadow_*.jsonl"):
            timestamp_str = filepath.stem.replace("shadow_", "")
            try:
                file_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                file_dt = file_dt.replace(tzinfo=timezone.utc)
                
                if file_dt < cutoff:
                    filepath.unlink()
                    deleted += 1
                    
            except ValueError:
                continue
        
        if deleted:
            logger.info("Deleted %d old shadow result files", deleted)
        
        return deleted
    
    def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        if self.storage_backend == "local":
            files = list(self.storage_path.glob("shadow_*.jsonl"))
            total_size = sum(f.stat().st_size for f in files)
            
            return {
                "backend": "local",
                "files": len(files),
                "total_size_bytes": total_size,
                "buffer_size": len(self._buffer),
                "storage_path": str(self.storage_path),
            }
        
        return {
            "backend": self.storage_backend,
            "buffer_size": len(self._buffer),
        }

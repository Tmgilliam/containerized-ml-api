"""Outcome ingestion for feedback loop."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OutcomeRecord:
    """A recorded outcome event."""
    entity_id: str
    outcome: bool
    outcome_timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "outcome": self.outcome,
            "outcome_timestamp": self.outcome_timestamp,
            "metadata": self.metadata,
        }


class OutcomeIngester:
    """
    Ingests actual outcome events from production.
    
    Captures whether predictions were correct (e.g., did the order actually delay)
    to enable feedback loop and model retraining.
    """
    
    def __init__(
        self,
        storage_backend: str = "local",
        storage_path: Path | None = None,
        bigquery_table: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """
        Initialize outcome ingester.
        
        Args:
            storage_backend: 'local', 'bigquery', or 'pubsub'
            storage_path: Local storage directory
            bigquery_table: BigQuery table for outcomes
            project_id: GCP project ID
        """
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else Path("./outcomes")
        self.bigquery_table = bigquery_table
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        self._buffer: list[OutcomeRecord] = []
        self._buffer_size = 100
        
        if storage_backend == "local":
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def ingest(
        self,
        entity_id: str,
        outcome: bool,
        timestamp: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OutcomeRecord:
        """
        Ingest a single outcome event.
        
        Args:
            entity_id: Entity identifier (e.g., order_id)
            outcome: Whether the predicted event occurred (e.g., delay=True)
            timestamp: When the outcome was observed
            metadata: Additional context
        
        Returns:
            OutcomeRecord that was created
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        record = OutcomeRecord(
            entity_id=entity_id,
            outcome=outcome,
            outcome_timestamp=timestamp,
            metadata=metadata or {},
        )
        
        self._buffer.append(record)
        
        if len(self._buffer) >= self._buffer_size:
            self.flush()
        
        logger.debug("Ingested outcome for entity %s: %s", entity_id, outcome)
        return record
    
    def ingest_batch(
        self,
        records: list[dict[str, Any]],
        entity_field: str = "entity_id",
        outcome_field: str = "outcome",
        timestamp_field: str = "timestamp",
    ) -> int:
        """
        Ingest multiple outcome events.
        
        Args:
            records: List of outcome records
            entity_field: Field name for entity ID
            outcome_field: Field name for outcome
            timestamp_field: Field name for timestamp
        
        Returns:
            Number of records ingested
        """
        count = 0
        
        for record in records:
            self.ingest(
                entity_id=record[entity_field],
                outcome=record[outcome_field],
                timestamp=record.get(timestamp_field),
                metadata={k: v for k, v in record.items() 
                         if k not in (entity_field, outcome_field, timestamp_field)},
            )
            count += 1
        
        self.flush()
        return count
    
    def flush(self) -> int:
        """
        Flush buffered outcomes to storage.
        
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
        logger.info("Flushed %d outcome records", count)
        return count
    
    def _flush_local(self) -> None:
        """Flush to local storage."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"outcomes_{timestamp}.jsonl"
        filepath = self.storage_path / filename
        
        with open(filepath, "w") as f:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict()) + "\n")
    
    def _flush_bigquery(self) -> None:
        """Flush to BigQuery."""
        if not self.bigquery_table:
            logger.error("BigQuery table not configured")
            return
        
        try:
            from google.cloud import bigquery
            
            client = bigquery.Client(project=self.project_id)
            rows = [r.to_dict() for r in self._buffer]
            
            errors = client.insert_rows_json(self.bigquery_table, rows)
            
            if errors:
                logger.error("BigQuery insert errors: %s", errors)
                
        except ImportError:
            logger.error("google-cloud-bigquery not installed")
        except Exception as e:
            logger.error("Failed to flush to BigQuery: %s", e)
    
    def get_outcomes(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 10000,
    ) -> list[OutcomeRecord]:
        """
        Retrieve stored outcomes.
        
        Args:
            start_date: Filter outcomes after this date
            end_date: Filter outcomes before this date
            limit: Maximum records to return
        
        Returns:
            List of OutcomeRecord
        """
        records = []
        
        if self.storage_backend == "local":
            for filepath in sorted(self.storage_path.glob("outcomes_*.jsonl")):
                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            record = OutcomeRecord(
                                entity_id=data["entity_id"],
                                outcome=data["outcome"],
                                outcome_timestamp=data["outcome_timestamp"],
                                metadata=data.get("metadata", {}),
                            )
                            
                            if start_date:
                                ts = datetime.fromisoformat(record.outcome_timestamp.replace("Z", "+00:00"))
                                if ts < start_date:
                                    continue
                            
                            if end_date:
                                ts = datetime.fromisoformat(record.outcome_timestamp.replace("Z", "+00:00"))
                                if ts > end_date:
                                    continue
                            
                            records.append(record)
                            
                            if len(records) >= limit:
                                return records
        
        return records
    
    def stats(self) -> dict[str, Any]:
        """Get ingestion statistics."""
        if self.storage_backend == "local":
            files = list(self.storage_path.glob("outcomes_*.jsonl"))
            total_records = 0
            
            for filepath in files:
                with open(filepath) as f:
                    total_records += sum(1 for _ in f)
            
            return {
                "backend": "local",
                "files": len(files),
                "total_records": total_records,
                "buffer_size": len(self._buffer),
            }
        
        return {
            "backend": self.storage_backend,
            "buffer_size": len(self._buffer),
        }

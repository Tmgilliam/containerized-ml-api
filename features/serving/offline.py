"""Offline feature store for batch feature retrieval."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from features.registry import FeatureRegistry

logger = logging.getLogger(__name__)


class OfflineFeatureStore:
    """
    Offline feature store for training data generation.
    
    Provides:
    - Point-in-time correct feature retrieval
    - Historical feature snapshots
    - BigQuery/local file backends
    - Training dataset generation
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        backend: str = "local",
        storage_path: Path | None = None,
        bigquery_dataset: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """
        Initialize offline feature store.
        
        Args:
            registry: Feature registry for schema validation
            backend: 'local' or 'bigquery'
            storage_path: Local storage directory (for local backend)
            bigquery_dataset: BigQuery dataset (for bigquery backend)
            project_id: GCP project ID
        """
        self.registry = registry
        self.backend = backend
        self.storage_path = Path(storage_path) if storage_path else Path("./feature_store")
        self.bigquery_dataset = bigquery_dataset
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if backend == "local":
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._bq_client = None
    
    def _get_bq_client(self):
        """Get BigQuery client."""
        if self._bq_client is None:
            try:
                from google.cloud import bigquery
                self._bq_client = bigquery.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-bigquery required for BigQuery backend")
        return self._bq_client
    
    def _local_path(self, group_name: str) -> Path:
        """Get local storage path for a feature group."""
        return self.storage_path / group_name
    
    def ingest_features(
        self,
        group_name: str,
        records: list[dict[str, Any]],
        timestamp_field: str = "event_timestamp",
        entity_field: str = "entity_id",
    ) -> int:
        """
        Ingest feature records into the offline store.
        
        Args:
            group_name: Feature group name
            records: List of feature records
            timestamp_field: Field containing event timestamp
            entity_field: Field containing entity identifier
        
        Returns:
            Number of records ingested
        """
        if self.backend == "local":
            return self._ingest_local(group_name, records, timestamp_field, entity_field)
        elif self.backend == "bigquery":
            return self._ingest_bigquery(group_name, records, timestamp_field, entity_field)
        
        return 0
    
    def _ingest_local(
        self,
        group_name: str,
        records: list[dict[str, Any]],
        timestamp_field: str,
        entity_field: str,
    ) -> int:
        """Ingest to local storage."""
        group_path = self._local_path(group_name)
        group_path.mkdir(parents=True, exist_ok=True)
        
        for record in records:
            ts = record.get(timestamp_field, datetime.now(timezone.utc).isoformat())
            entity_id = record.get(entity_field, "unknown")
            
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            
            ts_safe = ts.replace(":", "-").replace(".", "-")
            filename = f"{entity_id}_{ts_safe}.json"
            
            filepath = group_path / filename
            filepath.write_text(json.dumps(record, indent=2, default=str))
        
        logger.info("Ingested %d records to local store for %s", len(records), group_name)
        return len(records)
    
    def _ingest_bigquery(
        self,
        group_name: str,
        records: list[dict[str, Any]],
        timestamp_field: str,
        entity_field: str,
    ) -> int:
        """Ingest to BigQuery."""
        client = self._get_bq_client()
        
        table_id = f"{self.project_id}.{self.bigquery_dataset}.{group_name}"
        
        errors = client.insert_rows_json(table_id, records)
        
        if errors:
            logger.error("BigQuery insert errors: %s", errors)
            raise RuntimeError(f"BigQuery insert failed: {errors}")
        
        logger.info("Ingested %d records to BigQuery table %s", len(records), table_id)
        return len(records)
    
    def get_historical_features(
        self,
        group_name: str,
        entity_ids: list[str],
        timestamps: list[datetime],
        feature_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get point-in-time correct features.
        
        For each (entity_id, timestamp) pair, retrieves the most recent
        feature values that were available at that point in time.
        
        Args:
            group_name: Feature group name
            entity_ids: List of entity identifiers
            timestamps: List of timestamps (one per entity)
            feature_names: Optional subset of features
        
        Returns:
            List of feature dictionaries
        """
        if len(entity_ids) != len(timestamps):
            raise ValueError("entity_ids and timestamps must have same length")
        
        if self.backend == "local":
            return self._get_historical_local(
                group_name, entity_ids, timestamps, feature_names
            )
        elif self.backend == "bigquery":
            return self._get_historical_bigquery(
                group_name, entity_ids, timestamps, feature_names
            )
        
        return []
    
    def _get_historical_local(
        self,
        group_name: str,
        entity_ids: list[str],
        timestamps: list[datetime],
        feature_names: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Get historical features from local storage."""
        group_path = self._local_path(group_name)
        results = []
        
        for entity_id, ts in zip(entity_ids, timestamps):
            pattern = f"{entity_id}_*.json"
            files = list(group_path.glob(pattern))
            
            best_record = None
            best_ts = None
            
            for filepath in files:
                record = json.loads(filepath.read_text())
                record_ts = record.get("event_timestamp")
                
                if isinstance(record_ts, str):
                    record_ts = datetime.fromisoformat(record_ts.replace("Z", "+00:00"))
                
                if record_ts and record_ts <= ts:
                    if best_ts is None or record_ts > best_ts:
                        best_record = record
                        best_ts = record_ts
            
            if best_record:
                if feature_names:
                    best_record = {
                        k: v for k, v in best_record.items()
                        if k in feature_names or k in ("entity_id", "event_timestamp")
                    }
                results.append(best_record)
            else:
                results.append({"entity_id": entity_id, "_missing": True})
        
        return results
    
    def _get_historical_bigquery(
        self,
        group_name: str,
        entity_ids: list[str],
        timestamps: list[datetime],
        feature_names: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Get historical features from BigQuery using point-in-time join."""
        client = self._get_bq_client()
        
        table_id = f"{self.project_id}.{self.bigquery_dataset}.{group_name}"
        
        columns = "*"
        if feature_names:
            columns = ", ".join(["entity_id", "event_timestamp"] + feature_names)
        
        entity_ts_pairs = [
            f"('{eid}', TIMESTAMP('{ts.isoformat()}'))"
            for eid, ts in zip(entity_ids, timestamps)
        ]
        
        query = f"""
        WITH request AS (
            SELECT entity_id, request_ts
            FROM UNNEST([
                {', '.join(entity_ts_pairs)}
            ]) AS t(entity_id, request_ts)
        ),
        ranked AS (
            SELECT 
                f.*,
                r.request_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY f.entity_id 
                    ORDER BY f.event_timestamp DESC
                ) as rn
            FROM `{table_id}` f
            JOIN request r ON f.entity_id = r.entity_id
            WHERE f.event_timestamp <= r.request_ts
        )
        SELECT {columns}
        FROM ranked
        WHERE rn = 1
        """
        
        results = []
        for row in client.query(query).result():
            results.append(dict(row.items()))
        
        return results
    
    def generate_training_dataset(
        self,
        group_name: str,
        labels: list[dict[str, Any]],
        entity_field: str = "entity_id",
        timestamp_field: str = "event_timestamp",
        label_field: str = "label",
        feature_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a training dataset with point-in-time correct features.
        
        Args:
            group_name: Feature group name
            labels: List of dicts with entity_id, timestamp, and label
            entity_field: Field name for entity ID in labels
            timestamp_field: Field name for timestamp in labels
            label_field: Field name for label value
            feature_names: Optional subset of features
        
        Returns:
            List of training examples with features and labels
        """
        entity_ids = [l[entity_field] for l in labels]
        timestamps = []
        
        for l in labels:
            ts = l[timestamp_field]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            timestamps.append(ts)
        
        features = self.get_historical_features(
            group_name, entity_ids, timestamps, feature_names
        )
        
        dataset = []
        for label_record, feature_record in zip(labels, features):
            example = {**feature_record}
            example[label_field] = label_record[label_field]
            example["_label_timestamp"] = label_record[timestamp_field]
            dataset.append(example)
        
        logger.info(
            "Generated training dataset with %d examples for %s",
            len(dataset),
            group_name,
        )
        
        return dataset
    
    def list_entities(self, group_name: str, limit: int = 1000) -> list[str]:
        """List entity IDs with stored features."""
        if self.backend == "local":
            group_path = self._local_path(group_name)
            if not group_path.exists():
                return []
            
            entities = set()
            for filepath in group_path.glob("*.json"):
                entity_id = filepath.stem.rsplit("_", 1)[0]
                entities.add(entity_id)
                if len(entities) >= limit:
                    break
            
            return list(entities)[:limit]
        
        elif self.backend == "bigquery":
            client = self._get_bq_client()
            table_id = f"{self.project_id}.{self.bigquery_dataset}.{group_name}"
            
            query = f"""
            SELECT DISTINCT entity_id
            FROM `{table_id}`
            LIMIT {limit}
            """
            
            return [row.entity_id for row in client.query(query).result()]
        
        return []
    
    def stats(self, group_name: str) -> dict[str, Any]:
        """Get statistics for a feature group."""
        if self.backend == "local":
            group_path = self._local_path(group_name)
            if not group_path.exists():
                return {"exists": False}
            
            files = list(group_path.glob("*.json"))
            return {
                "backend": "local",
                "record_count": len(files),
                "storage_path": str(group_path),
            }
        
        elif self.backend == "bigquery":
            client = self._get_bq_client()
            table_id = f"{self.project_id}.{self.bigquery_dataset}.{group_name}"
            
            try:
                table = client.get_table(table_id)
                return {
                    "backend": "bigquery",
                    "row_count": table.num_rows,
                    "size_bytes": table.num_bytes,
                    "created": table.created.isoformat() if table.created else None,
                    "modified": table.modified.isoformat() if table.modified else None,
                }
            except Exception as e:
                return {"backend": "bigquery", "error": str(e)}
        
        return {}

"""Data writers for batch prediction pipeline."""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseWriter(ABC):
    """Base class for data writers."""
    
    @abstractmethod
    def write(self, records: list[dict[str, Any]]) -> int:
        """Write records to the destination. Returns number of records written."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Finalize and close the writer."""
        pass


class LocalWriter(BaseWriter):
    """Write data to local CSV or JSON files."""
    
    def __init__(
        self,
        file_path: str | Path,
        file_format: str = "auto",
        append: bool = False,
    ) -> None:
        """
        Initialize local file writer.
        
        Args:
            file_path: Path to output file
            file_format: 'csv', 'json', 'jsonl', or 'auto'
            append: Whether to append to existing file
        """
        self.file_path = Path(file_path)
        self.append = append
        self._records_written = 0
        self._file = None
        self._csv_writer = None
        self._header_written = False
        
        if file_format == "auto":
            ext = self.file_path.suffix.lower()
            if ext == ".csv":
                self.file_format = "csv"
            elif ext == ".json":
                self.file_format = "json"
            elif ext == ".jsonl":
                self.file_format = "jsonl"
            else:
                self.file_format = "jsonl"
        else:
            self.file_format = file_format
        
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_format in ("csv", "jsonl"):
            mode = "a" if append else "w"
            self._file = open(self.file_path, mode, newline="")
            if self.file_format == "csv" and append and self.file_path.exists():
                self._header_written = True
    
    def write(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        
        if self.file_format == "csv":
            return self._write_csv(records)
        elif self.file_format == "jsonl":
            return self._write_jsonl(records)
        elif self.file_format == "json":
            return self._write_json(records)
        
        return 0
    
    def _write_csv(self, records: list[dict[str, Any]]) -> int:
        if self._csv_writer is None:
            fieldnames = list(records[0].keys())
            self._csv_writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            if not self._header_written:
                self._csv_writer.writeheader()
                self._header_written = True
        
        for record in records:
            self._csv_writer.writerow(record)
        
        self._records_written += len(records)
        return len(records)
    
    def _write_jsonl(self, records: list[dict[str, Any]]) -> int:
        for record in records:
            self._file.write(json.dumps(record) + "\n")
        
        self._records_written += len(records)
        return len(records)
    
    def _write_json(self, records: list[dict[str, Any]]) -> int:
        existing = []
        if self.append and self.file_path.exists():
            with open(self.file_path, "r") as f:
                existing = json.load(f)
        
        all_records = existing + records
        
        with open(self.file_path, "w") as f:
            json.dump(all_records, f, indent=2)
        
        self._records_written += len(records)
        return len(records)
    
    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        
        logger.info(
            "Wrote %d records to %s",
            self._records_written,
            self.file_path,
        )


class GCSWriter(BaseWriter):
    """Write data to Google Cloud Storage."""
    
    def __init__(
        self,
        bucket: str,
        blob_path: str,
        file_format: str = "jsonl",
        project_id: str | None = None,
    ) -> None:
        """
        Initialize GCS writer.
        
        Args:
            bucket: GCS bucket name
            blob_path: Path for the output blob
            file_format: 'csv', 'json', or 'jsonl'
            project_id: GCP project ID
        """
        self.bucket_name = bucket
        self.blob_path = blob_path
        self.file_format = file_format
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._buffer: list[dict[str, Any]] = []
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-storage required for GCS support")
        return self._client
    
    def write(self, records: list[dict[str, Any]]) -> int:
        self._buffer.extend(records)
        return len(records)
    
    def close(self) -> None:
        if not self._buffer:
            return
        
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_path)
        
        if self.file_format == "jsonl":
            content = "\n".join(json.dumps(r) for r in self._buffer)
        elif self.file_format == "json":
            content = json.dumps(self._buffer, indent=2)
        elif self.file_format == "csv":
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=self._buffer[0].keys())
            writer.writeheader()
            writer.writerows(self._buffer)
            content = output.getvalue()
        else:
            content = ""
        
        blob.upload_from_string(content)
        
        logger.info(
            "Wrote %d records to gs://%s/%s",
            len(self._buffer),
            self.bucket_name,
            self.blob_path,
        )
        
        self._buffer.clear()


class BigQueryWriter(BaseWriter):
    """Write data to BigQuery."""
    
    def __init__(
        self,
        table: str,
        project_id: str | None = None,
        write_disposition: str = "WRITE_APPEND",
        batch_size: int = 10000,
    ) -> None:
        """
        Initialize BigQuery writer.
        
        Args:
            table: Full table reference (project.dataset.table)
            project_id: GCP project ID
            write_disposition: 'WRITE_APPEND', 'WRITE_TRUNCATE', or 'WRITE_EMPTY'
            batch_size: Records per insert batch
        """
        self.table = table
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.write_disposition = write_disposition
        self.batch_size = batch_size
        self._buffer: list[dict[str, Any]] = []
        self._total_written = 0
        self._client = None
        self._first_write = True
    
    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import bigquery
                self._client = bigquery.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-bigquery required for BigQuery support")
        return self._client
    
    def write(self, records: list[dict[str, Any]]) -> int:
        self._buffer.extend(records)
        
        written = 0
        while len(self._buffer) >= self.batch_size:
            batch = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
            self._write_batch(batch)
            written += len(batch)
        
        return written
    
    def _write_batch(self, records: list[dict[str, Any]]) -> None:
        client = self._get_client()
        
        from google.cloud import bigquery
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=True,
        )
        
        if self._first_write:
            if self.write_disposition == "WRITE_TRUNCATE":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            elif self.write_disposition == "WRITE_EMPTY":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            else:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            self._first_write = False
        else:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        
        errors = client.insert_rows_json(self.table, records)
        
        if errors:
            logger.error("BigQuery insert errors: %s", errors)
            raise RuntimeError(f"BigQuery insert failed: {errors}")
        
        self._total_written += len(records)
        logger.debug("Wrote batch of %d records to BigQuery", len(records))
    
    def close(self) -> None:
        if self._buffer:
            self._write_batch(self._buffer)
            self._buffer.clear()
        
        logger.info(
            "Wrote %d total records to BigQuery table %s",
            self._total_written,
            self.table,
        )

"""Data readers for batch prediction pipeline."""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class BaseReader(ABC):
    """Base class for data readers."""
    
    @abstractmethod
    def read(self) -> Iterator[dict[str, Any]]:
        """Yield records from the data source."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of records (if known)."""
        pass


class LocalReader(BaseReader):
    """Read data from local CSV or JSON files."""
    
    def __init__(
        self,
        file_path: str | Path,
        file_format: str = "auto",
        batch_size: int = 1000,
    ) -> None:
        """
        Initialize local file reader.
        
        Args:
            file_path: Path to the data file
            file_format: 'csv', 'json', 'jsonl', or 'auto' (detect from extension)
            batch_size: Number of records to yield at a time
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        
        if file_format == "auto":
            ext = self.file_path.suffix.lower()
            if ext == ".csv":
                self.file_format = "csv"
            elif ext == ".json":
                self.file_format = "json"
            elif ext == ".jsonl":
                self.file_format = "jsonl"
            else:
                raise ValueError(f"Cannot auto-detect format for extension: {ext}")
        else:
            self.file_format = file_format
    
    def read(self) -> Iterator[dict[str, Any]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        logger.info("Reading from local file: %s", self.file_path)
        
        if self.file_format == "csv":
            yield from self._read_csv()
        elif self.file_format == "json":
            yield from self._read_json()
        elif self.file_format == "jsonl":
            yield from self._read_jsonl()
    
    def _read_csv(self) -> Iterator[dict[str, Any]]:
        with open(self.file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted = {}
                for key, value in row.items():
                    try:
                        if "." in value:
                            converted[key] = float(value)
                        else:
                            converted[key] = int(value)
                    except (ValueError, TypeError):
                        converted[key] = value
                yield converted
    
    def _read_json(self) -> Iterator[dict[str, Any]]:
        with open(self.file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
    
    def _read_jsonl(self) -> Iterator[dict[str, Any]]:
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    def count(self) -> int:
        if self.file_format == "csv":
            with open(self.file_path, "r") as f:
                return sum(1 for _ in f) - 1
        elif self.file_format == "json":
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 1
        elif self.file_format == "jsonl":
            with open(self.file_path, "r") as f:
                return sum(1 for line in f if line.strip())
        return -1


class GCSReader(BaseReader):
    """Read data from Google Cloud Storage."""
    
    def __init__(
        self,
        bucket: str,
        blob_path: str,
        file_format: str = "jsonl",
        project_id: str | None = None,
    ) -> None:
        """
        Initialize GCS reader.
        
        Args:
            bucket: GCS bucket name
            blob_path: Path to blob within bucket
            file_format: 'csv', 'json', or 'jsonl'
            project_id: GCP project ID (uses default if not specified)
        """
        self.bucket = bucket
        self.blob_path = blob_path
        self.file_format = file_format
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-storage required for GCS support")
        return self._client
    
    def read(self) -> Iterator[dict[str, Any]]:
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(self.blob_path)
        
        logger.info("Reading from GCS: gs://%s/%s", self.bucket, self.blob_path)
        
        content = blob.download_as_text()
        
        if self.file_format == "jsonl":
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    yield json.loads(line)
        elif self.file_format == "json":
            data = json.loads(content)
            if isinstance(data, list):
                yield from data
            else:
                yield data
        elif self.file_format == "csv":
            lines = content.split("\n")
            reader = csv.DictReader(lines)
            for row in reader:
                yield row
    
    def count(self) -> int:
        return -1


class BigQueryReader(BaseReader):
    """Read data from BigQuery."""
    
    def __init__(
        self,
        query: str | None = None,
        table: str | None = None,
        project_id: str | None = None,
        page_size: int = 10000,
    ) -> None:
        """
        Initialize BigQuery reader.
        
        Args:
            query: SQL query to execute
            table: Full table reference (project.dataset.table)
            project_id: GCP project ID
            page_size: Number of rows per API page
        """
        if not query and not table:
            raise ValueError("Either query or table must be specified")
        
        self.query = query
        self.table = table
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.page_size = page_size
        self._client = None
        self._total_rows: int | None = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import bigquery
                self._client = bigquery.Client(project=self.project_id)
            except ImportError:
                raise ImportError("google-cloud-bigquery required for BigQuery support")
        return self._client
    
    def read(self) -> Iterator[dict[str, Any]]:
        client = self._get_client()
        
        if self.query:
            logger.info("Executing BigQuery query")
            query_job = client.query(self.query)
            results = query_job.result(page_size=self.page_size)
        else:
            logger.info("Reading from BigQuery table: %s", self.table)
            results = client.list_rows(self.table, page_size=self.page_size)
        
        self._total_rows = results.total_rows
        
        for row in results:
            yield dict(row.items())
    
    def count(self) -> int:
        if self._total_rows is not None:
            return self._total_rows
        
        client = self._get_client()
        
        if self.table:
            table_ref = client.get_table(self.table)
            return table_ref.num_rows
        
        return -1

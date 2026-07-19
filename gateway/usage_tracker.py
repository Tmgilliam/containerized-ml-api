"""API usage tracking and analytics."""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """A single API usage record."""
    timestamp: str
    client_id: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 2),
            "request_size_bytes": self.request_size_bytes,
            "response_size_bytes": self.response_size_bytes,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


class UsageTracker:
    """
    Tracks API usage for analytics and billing.
    
    Features:
    - Request counting per client
    - Latency tracking
    - Endpoint breakdown
    - Usage export
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        max_records_in_memory: int = 100000,
        flush_interval_seconds: float = 300.0,
    ) -> None:
        """
        Initialize usage tracker.
        
        Args:
            storage_path: Path to persist usage data
            max_records_in_memory: Maximum records before auto-flush
            flush_interval_seconds: Auto-flush interval
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_records = max_records_in_memory
        self.flush_interval = flush_interval_seconds
        
        self._records: list[UsageRecord] = []
        self._stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "total_latency_ms": 0.0,
            "error_count": 0,
            "endpoints": defaultdict(int),
            "first_request": None,
            "last_request": None,
        })
        self._lock = threading.Lock()
        self._last_flush = datetime.now(timezone.utc)
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def record(
        self,
        client_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """
        Record an API request.
        
        Args:
            client_id: Client identifier
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            latency_ms: Request latency
            request_size_bytes: Request body size
            response_size_bytes: Response body size
            model_version: Model version used
            metadata: Additional metadata
        
        Returns:
            UsageRecord that was created
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        record = UsageRecord(
            timestamp=timestamp,
            client_id=client_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            model_version=model_version,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._records.append(record)
            
            stats = self._stats[client_id]
            stats["request_count"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["endpoints"][endpoint] += 1
            
            if status_code >= 400:
                stats["error_count"] += 1
            
            if stats["first_request"] is None:
                stats["first_request"] = timestamp
            stats["last_request"] = timestamp
            
            if len(self._records) >= self.max_records:
                self._flush_unsafe()
        
        return record
    
    def _flush_unsafe(self) -> int:
        """Flush records to storage (must hold lock)."""
        if not self._records or not self.storage_path:
            return 0
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"usage_{timestamp}.jsonl"
        filepath = self.storage_path / filename
        
        with open(filepath, "w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict()) + "\n")
        
        count = len(self._records)
        self._records.clear()
        self._last_flush = datetime.now(timezone.utc)
        
        logger.info("Flushed %d usage records to %s", count, filepath)
        return count
    
    def flush(self) -> int:
        """Flush records to storage."""
        with self._lock:
            return self._flush_unsafe()
    
    def get_client_stats(self, client_id: str) -> dict[str, Any]:
        """Get usage statistics for a client."""
        with self._lock:
            if client_id not in self._stats:
                return {"error": "Client not found"}
            
            stats = self._stats[client_id]
            
            avg_latency = 0.0
            if stats["request_count"] > 0:
                avg_latency = stats["total_latency_ms"] / stats["request_count"]
            
            return {
                "client_id": client_id,
                "request_count": stats["request_count"],
                "error_count": stats["error_count"],
                "error_rate": stats["error_count"] / stats["request_count"] if stats["request_count"] > 0 else 0,
                "avg_latency_ms": round(avg_latency, 2),
                "endpoints": dict(stats["endpoints"]),
                "first_request": stats["first_request"],
                "last_request": stats["last_request"],
            }
    
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get usage statistics for all clients."""
        with self._lock:
            return {
                client_id: self.get_client_stats(client_id)
                for client_id in self._stats.keys()
            }
    
    def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get usage summary across all clients.
        
        Args:
            start_time: Filter records after this time
            end_time: Filter records before this time
        """
        with self._lock:
            filtered = self._records
            
            if start_time:
                filtered = [
                    r for r in filtered
                    if datetime.fromisoformat(r.timestamp.replace("Z", "+00:00")) >= start_time
                ]
            
            if end_time:
                filtered = [
                    r for r in filtered
                    if datetime.fromisoformat(r.timestamp.replace("Z", "+00:00")) <= end_time
                ]
            
            if not filtered:
                return {"total_requests": 0, "period": "no data"}
            
            total_requests = len(filtered)
            total_latency = sum(r.latency_ms for r in filtered)
            errors = sum(1 for r in filtered if r.status_code >= 400)
            
            unique_clients = len(set(r.client_id for r in filtered))
            
            endpoint_counts: dict[str, int] = defaultdict(int)
            for r in filtered:
                endpoint_counts[r.endpoint] += 1
            
            return {
                "total_requests": total_requests,
                "unique_clients": unique_clients,
                "total_errors": errors,
                "error_rate": round(errors / total_requests, 4) if total_requests > 0 else 0,
                "avg_latency_ms": round(total_latency / total_requests, 2) if total_requests > 0 else 0,
                "endpoints": dict(endpoint_counts),
                "records_in_memory": len(self._records),
                "last_flush": self._last_flush.isoformat(),
            }
    
    def export_records(
        self,
        client_id: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Export usage records."""
        with self._lock:
            records = self._records
            
            if client_id:
                records = [r for r in records if r.client_id == client_id]
            
            paginated = records[offset:offset + limit]
            
            return [r.to_dict() for r in paginated]
    
    def reset_stats(self, client_id: str | None = None) -> None:
        """Reset usage statistics."""
        with self._lock:
            if client_id:
                if client_id in self._stats:
                    del self._stats[client_id]
            else:
                self._stats.clear()
        
        logger.info("Reset usage stats for %s", client_id or "all clients")

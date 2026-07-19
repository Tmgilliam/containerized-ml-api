"""Online feature store for real-time feature serving."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from features.registry import FeatureRegistry, FeatureGroup

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Online feature store for low-latency feature retrieval.
    
    Provides:
    - In-memory caching with TTL
    - Redis backend support
    - Feature freshness tracking
    - Batch retrieval
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        backend: str = "memory",
        redis_url: str | None = None,
        default_ttl_seconds: int = 3600,
    ) -> None:
        """
        Initialize online feature store.
        
        Args:
            registry: Feature registry for schema validation
            backend: 'memory' or 'redis'
            redis_url: Redis connection URL (required if backend='redis')
            default_ttl_seconds: Default TTL for cached features
        """
        self.registry = registry
        self.backend = backend
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()
        
        self._redis = None
        if backend == "redis":
            self._init_redis(redis_url or os.getenv("REDIS_URL"))
    
    def _init_redis(self, redis_url: str | None) -> None:
        """Initialize Redis connection."""
        if not redis_url:
            logger.warning("Redis URL not provided, falling back to memory backend")
            self.backend = "memory"
            return
        
        try:
            import redis
            self._redis = redis.from_url(redis_url)
            self._redis.ping()
            logger.info("Connected to Redis at %s", redis_url.split("@")[-1])
        except ImportError:
            logger.warning("redis package not installed, falling back to memory")
            self.backend = "memory"
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.backend = "memory"
    
    def _make_key(self, group_name: str, entity_id: str) -> str:
        """Generate cache key."""
        return f"features:{group_name}:{entity_id}"
    
    def set_features(
        self,
        group_name: str,
        entity_id: str,
        features: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Store features for an entity.
        
        Args:
            group_name: Feature group name
            entity_id: Entity identifier
            features: Feature values
            ttl_seconds: Optional TTL override
        """
        key = self._make_key(group_name, entity_id)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        data = {
            "features": features,
            "timestamp": time.time(),
            "group": group_name,
        }
        
        if self.backend == "redis" and self._redis:
            self._redis.setex(key, ttl, json.dumps(data))
        else:
            with self._lock:
                self._cache[key] = data
                self._timestamps[key] = time.time() + ttl
        
        logger.debug("Stored features for %s/%s", group_name, entity_id)
    
    def get_features(
        self,
        group_name: str,
        entity_id: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve features for an entity.
        
        Args:
            group_name: Feature group name
            entity_id: Entity identifier
            feature_names: Optional subset of features to retrieve
        
        Returns:
            Feature dictionary or None if not found/expired
        """
        key = self._make_key(group_name, entity_id)
        
        if self.backend == "redis" and self._redis:
            raw = self._redis.get(key)
            if not raw:
                return None
            data = json.loads(raw)
        else:
            with self._lock:
                if key not in self._cache:
                    return None
                
                if time.time() > self._timestamps.get(key, 0):
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                
                data = self._cache[key]
        
        features = data["features"]
        
        if feature_names:
            features = {k: v for k, v in features.items() if k in feature_names}
        
        return features
    
    def get_features_batch(
        self,
        group_name: str,
        entity_ids: list[str],
        feature_names: list[str] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        """
        Retrieve features for multiple entities.
        
        Args:
            group_name: Feature group name
            entity_ids: List of entity identifiers
            feature_names: Optional subset of features to retrieve
        
        Returns:
            Dict mapping entity_id to features (or None if not found)
        """
        if self.backend == "redis" and self._redis:
            keys = [self._make_key(group_name, eid) for eid in entity_ids]
            values = self._redis.mget(keys)
            
            result = {}
            for entity_id, raw in zip(entity_ids, values):
                if raw:
                    data = json.loads(raw)
                    features = data["features"]
                    if feature_names:
                        features = {k: v for k, v in features.items() if k in feature_names}
                    result[entity_id] = features
                else:
                    result[entity_id] = None
            
            return result
        else:
            return {
                entity_id: self.get_features(group_name, entity_id, feature_names)
                for entity_id in entity_ids
            }
    
    def delete_features(self, group_name: str, entity_id: str) -> bool:
        """Delete features for an entity."""
        key = self._make_key(group_name, entity_id)
        
        if self.backend == "redis" and self._redis:
            return bool(self._redis.delete(key))
        else:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    del self._timestamps[key]
                    return True
                return False
    
    def get_feature_freshness(
        self,
        group_name: str,
        entity_id: str,
    ) -> float | None:
        """
        Get age of cached features in seconds.
        
        Returns None if features not found.
        """
        key = self._make_key(group_name, entity_id)
        
        if self.backend == "redis" and self._redis:
            raw = self._redis.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            return time.time() - data["timestamp"]
        else:
            with self._lock:
                if key not in self._cache:
                    return None
                return time.time() - self._cache[key]["timestamp"]
    
    def get_online_features(
        self,
        group_name: str,
        entity_id: str,
        request_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get features for inference, combining cached and request-time features.
        
        Args:
            group_name: Feature group name
            entity_id: Entity identifier
            request_features: Features provided at request time
        
        Returns:
            Complete feature dictionary for model inference
        """
        cached = self.get_features(group_name, entity_id) or {}
        
        if request_features:
            cached.update(request_features)
        
        group = self.registry.get_group(group_name)
        if group:
            for feature in group.features:
                if feature.name not in cached and feature.default_value is not None:
                    cached[feature.name] = feature.default_value
        
        return cached
    
    def clear_expired(self) -> int:
        """Clear expired entries from memory cache. Returns count cleared."""
        if self.backend != "memory":
            return 0
        
        cleared = 0
        now = time.time()
        
        with self._lock:
            expired_keys = [
                k for k, exp in self._timestamps.items()
                if now > exp
            ]
            
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
                cleared += 1
        
        if cleared:
            logger.info("Cleared %d expired cache entries", cleared)
        
        return cleared
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.backend == "redis" and self._redis:
            info = self._redis.info()
            return {
                "backend": "redis",
                "keys": self._redis.dbsize(),
                "memory_used": info.get("used_memory_human"),
            }
        else:
            with self._lock:
                return {
                    "backend": "memory",
                    "keys": len(self._cache),
                    "expired_pending": sum(
                        1 for exp in self._timestamps.values()
                        if time.time() > exp
                    ),
                }

"""Rate limiting using token bucket algorithm."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    enable_sliding_window: bool = True
    block_duration_seconds: float = 60.0


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: float
    tokens: float
    refill_rate: float
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Return seconds until tokens become available."""
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_seconds: float
    limit: int
    
    def to_headers(self) -> dict[str, str]:
        """Generate rate limit response headers."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(time.time() + self.reset_seconds)),
        }


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    
    Features:
    - Per-client rate limiting
    - Burst handling
    - Configurable limits
    - Header generation for responses
    """
    
    def __init__(
        self,
        default_config: RateLimitConfig | None = None,
    ) -> None:
        """
        Initialize rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        
        self._buckets: dict[str, TokenBucket] = {}
        self._client_configs: dict[str, RateLimitConfig] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()
    
    def _get_or_create_bucket(
        self,
        client_id: str,
        config: RateLimitConfig,
    ) -> TokenBucket:
        """Get or create a token bucket for a client."""
        if client_id not in self._buckets:
            refill_rate = config.requests_per_minute / 60.0
            self._buckets[client_id] = TokenBucket(
                capacity=float(config.burst_size),
                tokens=float(config.burst_size),
                refill_rate=refill_rate,
            )
        
        return self._buckets[client_id]
    
    def set_client_limit(
        self,
        client_id: str,
        config: RateLimitConfig,
    ) -> None:
        """Set custom rate limit for a client."""
        with self._lock:
            self._client_configs[client_id] = config
            
            if client_id in self._buckets:
                del self._buckets[client_id]
        
        logger.info(
            "Set rate limit for %s: %d req/min",
            client_id,
            config.requests_per_minute,
        )
    
    def check(
        self,
        client_id: str,
        tokens: int = 1,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.
        
        Args:
            client_id: Client identifier
            tokens: Number of tokens to consume
        
        Returns:
            RateLimitResult with allow/deny and metadata
        """
        with self._lock:
            if client_id in self._blocked_until:
                if time.time() < self._blocked_until[client_id]:
                    config = self._client_configs.get(client_id, self.default_config)
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_seconds=self._blocked_until[client_id] - time.time(),
                        limit=config.requests_per_minute,
                    )
                else:
                    del self._blocked_until[client_id]
            
            config = self._client_configs.get(client_id, self.default_config)
            bucket = self._get_or_create_bucket(client_id, config)
            
            allowed = bucket.consume(tokens)
            
            if not allowed:
                self._blocked_until[client_id] = time.time() + config.block_duration_seconds
            
            return RateLimitResult(
                allowed=allowed,
                remaining=int(bucket.tokens),
                reset_seconds=bucket.time_until_available(tokens),
                limit=config.requests_per_minute,
            )
    
    def reset(self, client_id: str) -> bool:
        """Reset rate limit for a client."""
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]
            if client_id in self._blocked_until:
                del self._blocked_until[client_id]
            return True
        return False
    
    def get_status(self, client_id: str) -> dict[str, Any]:
        """Get rate limit status for a client."""
        with self._lock:
            config = self._client_configs.get(client_id, self.default_config)
            bucket = self._buckets.get(client_id)
            
            blocked_until = self._blocked_until.get(client_id)
            is_blocked = blocked_until and time.time() < blocked_until
            
            return {
                "client_id": client_id,
                "limit": config.requests_per_minute,
                "burst_size": config.burst_size,
                "remaining": int(bucket.tokens) if bucket else config.burst_size,
                "is_blocked": is_blocked,
                "blocked_seconds_remaining": max(0, blocked_until - time.time()) if is_blocked else 0,
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired blocks and stale buckets. Returns count cleaned."""
        now = time.time()
        cleaned = 0
        
        with self._lock:
            expired_blocks = [
                cid for cid, until in self._blocked_until.items()
                if now >= until
            ]
            for cid in expired_blocks:
                del self._blocked_until[cid]
                cleaned += 1
            
            stale_buckets = [
                cid for cid, bucket in self._buckets.items()
                if now - bucket.last_refill > 3600
            ]
            for cid in stale_buckets:
                del self._buckets[cid]
                cleaned += 1
        
        if cleaned:
            logger.debug("Cleaned %d expired rate limit entries", cleaned)
        
        return cleaned

"""API Key authentication."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key with metadata."""
    key_id: str
    hashed_key: str
    client_name: str
    created_at: str
    expires_at: str | None = None
    rate_limit: int | None = None
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if not self.expires_at:
            return False
        expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > expires
    
    def has_scope(self, scope: str) -> bool:
        """Check if key has a specific scope."""
        return "*" in self.scopes or scope in self.scopes
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "client_name": self.client_name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "rate_limit": self.rate_limit,
            "scopes": self.scopes,
            "is_active": self.is_active,
            "is_expired": self.is_expired(),
        }


class APIKeyAuth:
    """
    API Key authentication manager.
    
    Features:
    - Secure key generation and hashing
    - Scope-based access control
    - Key expiration
    - Per-key rate limits
    """
    
    KEY_PREFIX = "mlapi_"
    KEY_LENGTH = 32
    
    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}
        self._key_id_by_hash: dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def generate_key(
        self,
        client_name: str,
        expires_in_days: int | None = None,
        rate_limit: int | None = None,
        scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            client_name: Name of the client/application
            expires_in_days: Days until expiration (None = never)
            rate_limit: Requests per minute (None = default)
            scopes: List of allowed scopes
            metadata: Additional metadata
        
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        raw_key = f"{self.KEY_PREFIX}{secrets.token_urlsafe(self.KEY_LENGTH)}"
        hashed = self._hash_key(raw_key)
        key_id = secrets.token_hex(8)
        
        now = datetime.now(timezone.utc)
        
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()
        
        api_key = APIKey(
            key_id=key_id,
            hashed_key=hashed,
            client_name=client_name,
            created_at=now.isoformat(),
            expires_at=expires_at,
            rate_limit=rate_limit,
            scopes=scopes or ["*"],
            metadata=metadata or {},
        )
        
        with self._lock:
            self._keys[key_id] = api_key
            self._key_id_by_hash[hashed] = key_id
        
        logger.info(
            "Generated API key for %s (id=%s, expires=%s)",
            client_name,
            key_id,
            expires_at,
        )
        
        return raw_key, api_key
    
    def validate(
        self,
        raw_key: str,
        required_scope: str | None = None,
    ) -> tuple[bool, APIKey | None, str | None]:
        """
        Validate an API key.
        
        Args:
            raw_key: The raw API key to validate
            required_scope: Optional scope requirement
        
        Returns:
            Tuple of (is_valid, APIKey or None, error_message or None)
        """
        if not raw_key:
            return False, None, "API key required"
        
        if not raw_key.startswith(self.KEY_PREFIX):
            return False, None, "Invalid API key format"
        
        hashed = self._hash_key(raw_key)
        
        with self._lock:
            key_id = self._key_id_by_hash.get(hashed)
            if not key_id:
                return False, None, "Invalid API key"
            
            api_key = self._keys.get(key_id)
            if not api_key:
                return False, None, "API key not found"
        
        if not api_key.is_active:
            return False, api_key, "API key is deactivated"
        
        if api_key.is_expired():
            return False, api_key, "API key has expired"
        
        if required_scope and not api_key.has_scope(required_scope):
            return False, api_key, f"Missing required scope: {required_scope}"
        
        return True, api_key, None
    
    def revoke(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._lock:
            if key_id not in self._keys:
                return False
            
            api_key = self._keys[key_id]
            api_key.is_active = False
            
            if api_key.hashed_key in self._key_id_by_hash:
                del self._key_id_by_hash[api_key.hashed_key]
        
        logger.info("Revoked API key: %s", key_id)
        return True
    
    def rotate(
        self,
        key_id: str,
        expires_in_days: int | None = None,
    ) -> tuple[str, APIKey] | None:
        """
        Rotate an API key (generate new, revoke old).
        
        Args:
            key_id: ID of key to rotate
            expires_in_days: New expiration (inherits from old if not specified)
        
        Returns:
            Tuple of (new_raw_key, new_APIKey) or None if key not found
        """
        with self._lock:
            old_key = self._keys.get(key_id)
            if not old_key:
                return None
        
        new_raw, new_key = self.generate_key(
            client_name=old_key.client_name,
            expires_in_days=expires_in_days,
            rate_limit=old_key.rate_limit,
            scopes=old_key.scopes,
            metadata=old_key.metadata,
        )
        
        self.revoke(key_id)
        
        logger.info(
            "Rotated API key %s -> %s for %s",
            key_id,
            new_key.key_id,
            old_key.client_name,
        )
        
        return new_raw, new_key
    
    def get_key(self, key_id: str) -> APIKey | None:
        """Get API key metadata by ID."""
        with self._lock:
            return self._keys.get(key_id)
    
    def list_keys(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        """List all API keys (without hashes)."""
        with self._lock:
            keys = list(self._keys.values())
        
        if not include_inactive:
            keys = [k for k in keys if k.is_active]
        
        return [k.to_dict() for k in keys]
    
    def load_keys(self, keys_data: list[dict[str, Any]]) -> int:
        """Load API keys from storage."""
        count = 0
        
        with self._lock:
            for data in keys_data:
                api_key = APIKey(
                    key_id=data["key_id"],
                    hashed_key=data["hashed_key"],
                    client_name=data["client_name"],
                    created_at=data["created_at"],
                    expires_at=data.get("expires_at"),
                    rate_limit=data.get("rate_limit"),
                    scopes=data.get("scopes", ["*"]),
                    metadata=data.get("metadata", {}),
                    is_active=data.get("is_active", True),
                )
                
                self._keys[api_key.key_id] = api_key
                self._key_id_by_hash[api_key.hashed_key] = api_key.key_id
                count += 1
        
        logger.info("Loaded %d API keys", count)
        return count
    
    def export_keys(self) -> list[dict[str, Any]]:
        """Export API keys for storage (includes hashes)."""
        with self._lock:
            return [
                {
                    "key_id": k.key_id,
                    "hashed_key": k.hashed_key,
                    "client_name": k.client_name,
                    "created_at": k.created_at,
                    "expires_at": k.expires_at,
                    "rate_limit": k.rate_limit,
                    "scopes": k.scopes,
                    "metadata": k.metadata,
                    "is_active": k.is_active,
                }
                for k in self._keys.values()
            ]

"""JWT validation for OAuth2/OIDC authentication."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


@dataclass
class JWTClaims:
    """Validated JWT claims."""
    subject: str
    issuer: str
    audience: str | list[str]
    expires_at: int
    issued_at: int
    scopes: list[str]
    email: str | None = None
    name: str | None = None
    custom_claims: dict[str, Any] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.subject,
            "iss": self.issuer,
            "aud": self.audience,
            "exp": self.expires_at,
            "iat": self.issued_at,
            "scopes": self.scopes,
            "email": self.email,
            "name": self.name,
        }


class JWTValidator:
    """
    JWT token validator for OAuth2/OIDC.
    
    Features:
    - JWKS (JSON Web Key Set) fetching and caching
    - Token signature verification
    - Claims validation
    - Scope checking
    """
    
    JWKS_CACHE_TTL = 3600
    
    def __init__(
        self,
        issuer: str,
        audience: str | list[str],
        jwks_uri: str | None = None,
        algorithms: list[str] | None = None,
    ) -> None:
        """
        Initialize JWT validator.
        
        Args:
            issuer: Expected token issuer (iss claim)
            audience: Expected audience (aud claim)
            jwks_uri: URI to fetch JSON Web Key Set
            algorithms: Allowed signing algorithms
        """
        self.issuer = issuer
        self.audience = audience if isinstance(audience, list) else [audience]
        self.jwks_uri = jwks_uri or f"{issuer}/.well-known/jwks.json"
        self.algorithms = algorithms or ["RS256", "RS384", "RS512"]
        
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
    
    def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch JWKS from the issuer."""
        now = time.time()
        
        if self._jwks_cache and (now - self._jwks_cache_time) < self.JWKS_CACHE_TTL:
            return self._jwks_cache
        
        try:
            with urlopen(self.jwks_uri, timeout=10) as response:
                jwks = json.loads(response.read().decode())
                self._jwks_cache = jwks
                self._jwks_cache_time = now
                logger.info("Fetched JWKS from %s", self.jwks_uri)
                return jwks
                
        except URLError as e:
            logger.error("Failed to fetch JWKS: %s", e)
            if self._jwks_cache:
                return self._jwks_cache
            raise
    
    def _get_signing_key(self, token: str) -> Any:
        """Get the signing key for a token from JWKS."""
        try:
            import jwt
            from jwt import PyJWKClient
            
            jwks_client = PyJWKClient(self.jwks_uri)
            return jwks_client.get_signing_key_from_jwt(token)
            
        except ImportError:
            logger.warning("PyJWT not installed with crypto support")
            return None
    
    def validate(
        self,
        token: str,
        required_scopes: list[str] | None = None,
    ) -> tuple[bool, JWTClaims | None, str | None]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token string
            required_scopes: Optional list of required scopes
        
        Returns:
            Tuple of (is_valid, JWTClaims or None, error_message or None)
        """
        if not token:
            return False, None, "Token required"
        
        if token.startswith("Bearer "):
            token = token[7:]
        
        try:
            import jwt
            
            signing_key = self._get_signing_key(token)
            
            if signing_key:
                payload = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=self.algorithms,
                    audience=self.audience,
                    issuer=self.issuer,
                )
            else:
                payload = jwt.decode(
                    token,
                    options={"verify_signature": False},
                )
                
                if payload.get("iss") != self.issuer:
                    return False, None, f"Invalid issuer: {payload.get('iss')}"
                
                token_aud = payload.get("aud")
                if isinstance(token_aud, str):
                    token_aud = [token_aud]
                if not any(a in self.audience for a in (token_aud or [])):
                    return False, None, f"Invalid audience: {token_aud}"
                
                if payload.get("exp", 0) < time.time():
                    return False, None, "Token expired"
            
            scope_str = payload.get("scope", payload.get("scp", ""))
            if isinstance(scope_str, str):
                scopes = scope_str.split() if scope_str else []
            else:
                scopes = scope_str or []
            
            claims = JWTClaims(
                subject=payload.get("sub", ""),
                issuer=payload.get("iss", ""),
                audience=payload.get("aud", ""),
                expires_at=payload.get("exp", 0),
                issued_at=payload.get("iat", 0),
                scopes=scopes,
                email=payload.get("email"),
                name=payload.get("name"),
                custom_claims={
                    k: v for k, v in payload.items()
                    if k not in ("sub", "iss", "aud", "exp", "iat", "scope", "scp", "email", "name")
                },
            )
            
            if required_scopes:
                missing = [s for s in required_scopes if s not in scopes]
                if missing:
                    return False, claims, f"Missing required scopes: {missing}"
            
            return True, claims, None
            
        except ImportError:
            return False, None, "JWT library not installed"
            
        except Exception as e:
            logger.warning("JWT validation failed: %s", e)
            return False, None, str(e)
    
    def decode_without_verification(self, token: str) -> dict[str, Any] | None:
        """Decode a JWT without verifying (for debugging)."""
        try:
            import jwt
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None

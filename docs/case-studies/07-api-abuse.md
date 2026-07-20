# Case Study 7: API Abuse by Third-Party Integrator

## Company Profile

**Industry:** Automotive parts supplier  
**API Consumers:** 12 internal systems, 8 external partners  
**Challenge:** Uncontrolled API access causing instability and cost overruns

---

## The Problem

### Situation

An automotive parts supplier exposed their delay risk scoring API to internal systems (ERP, planning tools) and external partners (distributors, logistics providers). The API was protected only by a shared API key per partner.

### The Incident

During a routine cost review, the infrastructure team discovered:

| Consumer | Expected Calls/Day | Actual Calls/Day | Cost Impact |
|----------|-------------------|------------------|-------------|
| ERP System | 15,000 | 14,800 | Normal |
| Planning Tool | 5,000 | 5,200 | Normal |
| Partner A | 2,000 | 2,100 | Normal |
| Partner B | 1,000 | **47,000** | +$12,000/month |
| Partner C | 500 | 480 | Normal |

### Investigation

Partner B was a mid-sized distributor with a new developer who:

1. **Implemented polling** — Called the API every 5 seconds for 3,000 orders instead of event-driven updates
2. **No caching** — Ignored Cache-Control headers
3. **Retry storm** — On any 500 error, immediately retried 10 times
4. **Batch inefficiency** — Made individual calls instead of using batch endpoint

### Business Impact

- **$144K/year** in unexpected API costs
- **15% latency increase** for all consumers during peak
- **3 outages** triggered by Partner B's retry storms
- **SLA breach** with another partner during outage

---

## The Solution

### Phase 1: API Key Management

```python
from gateway.auth import APIKeyManager, APIKeyConfig
from datetime import datetime, timedelta

# Initialize key manager
key_manager = APIKeyManager()

# Create differentiated keys by partner tier
def provision_partner_keys():
    """Create API keys with appropriate permissions."""
    
    # Internal systems - high limits
    key_manager.create_key(
        name="erp-system",
        config=APIKeyConfig(
            scopes=["predict", "batch", "explain"],
            rate_limit=1000,  # requests per minute
            burst_limit=100,  # max burst
            expires_at=None,  # No expiration for internal
            metadata={
                "type": "internal",
                "system": "ERP",
                "cost_center": "IT-001",
            }
        )
    )
    
    # Tier 1 partners - high limits
    key_manager.create_key(
        name="partner-a-production",
        config=APIKeyConfig(
            scopes=["predict", "batch"],
            rate_limit=200,
            burst_limit=50,
            expires_at=datetime.now() + timedelta(days=365),
            metadata={
                "type": "partner",
                "tier": "1",
                "company": "Partner A Inc.",
            }
        )
    )
    
    # Tier 2 partners - moderate limits
    key_manager.create_key(
        name="partner-b-production",
        config=APIKeyConfig(
            scopes=["predict"],  # No batch access
            rate_limit=50,  # Much lower limit
            burst_limit=10,
            expires_at=datetime.now() + timedelta(days=365),
            metadata={
                "type": "partner",
                "tier": "2",
                "company": "Partner B Corp.",
            }
        )
    )
    
    return key_manager.list_keys()
```

### Phase 2: Rate Limiting with Token Bucket

```python
from gateway import RateLimiter, RateLimitConfig
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Initialize rate limiter
rate_limiter = RateLimiter()

# Configure per-client limits
def get_rate_limit_for_key(api_key: str) -> RateLimitConfig:
    """Get rate limit configuration based on API key."""
    
    key_info = key_manager.get_key_info(api_key)
    
    if not key_info:
        # Unknown key - very restrictive
        return RateLimitConfig(
            requests_per_minute=10,
            burst_size=5,
        )
    
    return RateLimitConfig(
        requests_per_minute=key_info.config.rate_limit,
        burst_size=key_info.config.burst_limit,
    )

# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting based on API key."""
    
    # Extract API key
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": "API key required"}
        )
    
    # Get rate limit config
    config = get_rate_limit_for_key(api_key)
    
    # Check rate limit
    result = rate_limiter.check(
        client_id=api_key,
        config=config,
    )
    
    if not result.allowed:
        # Return 429 with retry-after header
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after_seconds": result.retry_after_seconds,
                "limit": config.requests_per_minute,
                "remaining": 0,
            },
            headers={
                "Retry-After": str(int(result.retry_after_seconds)),
                "X-RateLimit-Limit": str(config.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(result.reset_time.timestamp())),
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(int(result.reset_time.timestamp()))
    
    return response
```

### Phase 3: Usage Tracking and Analytics

```python
from gateway import UsageTracker, UsageRecord
from datetime import datetime, timedelta

# Initialize tracker
tracker = UsageTracker(
    storage_path="./data/usage",
    flush_interval_seconds=60,
)

# Track every request
@app.middleware("http")
async def usage_tracking_middleware(request: Request, call_next):
    """Track API usage for analytics and billing."""
    
    start_time = datetime.now()
    api_key = request.headers.get("X-API-Key", "anonymous")
    endpoint = f"{request.method} {request.url.path}"
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    # Record usage
    tracker.record(UsageRecord(
        api_key=api_key,
        endpoint=endpoint,
        timestamp=start_time,
        latency_ms=latency_ms,
        status_code=response.status_code,
        request_size_bytes=int(request.headers.get("Content-Length", 0)),
        response_size_bytes=int(response.headers.get("Content-Length", 0)),
    ))
    
    return response

# Generate usage reports
def generate_partner_usage_report(partner_key: str, days: int = 30) -> dict:
    """Generate detailed usage report for a partner."""
    
    start_date = datetime.now() - timedelta(days=days)
    
    stats = tracker.get_stats(
        api_key=partner_key,
        start_time=start_date,
    )
    
    return {
        "partner_key": partner_key,
        "period_days": days,
        "total_requests": stats.total_requests,
        "successful_requests": stats.successful_requests,
        "failed_requests": stats.failed_requests,
        "total_latency_ms": stats.total_latency_ms,
        "avg_latency_ms": stats.avg_latency_ms,
        "p95_latency_ms": stats.p95_latency_ms,
        "requests_by_endpoint": stats.requests_by_endpoint,
        "requests_by_status": stats.requests_by_status,
        "daily_breakdown": stats.daily_breakdown,
        "estimated_cost": calculate_cost(stats.total_requests),
    }

def calculate_cost(request_count: int) -> float:
    """Calculate API usage cost."""
    
    # Pricing tiers
    if request_count <= 10000:
        return request_count * 0.001  # $0.001 per request
    elif request_count <= 100000:
        return 10 + (request_count - 10000) * 0.0008
    else:
        return 82 + (request_count - 100000) * 0.0005
```

### Phase 4: JWT Validation for OAuth2 Partners

```python
from gateway.auth import JWTValidator
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize JWT validator
jwt_validator = JWTValidator(
    issuer="https://auth.example.com",
    audience="delay-risk-api",
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
)

security = HTTPBearer()

async def validate_jwt(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """Validate JWT token and extract claims."""
    
    token = credentials.credentials
    
    result = jwt_validator.validate(token)
    
    if not result.is_valid:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {result.error}",
        )
    
    return result.claims

# Protected endpoint with JWT
@app.post("/v2/predict")
async def predict_v2(
    payload: PredictRequest,
    claims: dict = Depends(validate_jwt),
):
    """OAuth2-protected prediction endpoint."""
    
    # Extract client info from JWT claims
    client_id = claims.get("client_id")
    scopes = claims.get("scope", "").split()
    
    # Verify required scope
    if "predict" not in scopes:
        raise HTTPException(
            status_code=403,
            detail="Missing required scope: predict",
        )
    
    # Process prediction
    result = delay_risk_model.predict(payload.model_dump())
    
    return {
        **result,
        "client_id": client_id,
    }
```

### Phase 5: Abuse Detection and Alerting

```python
from collections import defaultdict
from datetime import datetime, timedelta
import threading

class AbuseDetector:
    """Detect and alert on API abuse patterns."""
    
    def __init__(self):
        self.request_counts = defaultdict(list)
        self.error_counts = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_request(
        self,
        client_id: str,
        status_code: int,
        timestamp: datetime = None,
    ):
        """Record request for abuse detection."""
        
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            self.request_counts[client_id].append(timestamp)
            
            if status_code >= 400:
                self.error_counts[client_id].append(timestamp)
            
            # Cleanup old records (keep last hour)
            cutoff = timestamp - timedelta(hours=1)
            self.request_counts[client_id] = [
                t for t in self.request_counts[client_id] if t > cutoff
            ]
            self.error_counts[client_id] = [
                t for t in self.error_counts[client_id] if t > cutoff
            ]
    
    def check_abuse(self, client_id: str) -> dict:
        """Check for abuse patterns."""
        
        alerts = []
        
        with self._lock:
            # Check request velocity
            recent_requests = len([
                t for t in self.request_counts[client_id]
                if t > datetime.now() - timedelta(minutes=1)
            ])
            
            if recent_requests > 100:
                alerts.append({
                    "type": "high_velocity",
                    "message": f"High request velocity: {recent_requests}/min",
                    "severity": "WARNING",
                })
            
            # Check error rate
            recent_errors = len([
                t for t in self.error_counts[client_id]
                if t > datetime.now() - timedelta(minutes=5)
            ])
            total_recent = len([
                t for t in self.request_counts[client_id]
                if t > datetime.now() - timedelta(minutes=5)
            ])
            
            if total_recent > 0:
                error_rate = recent_errors / total_recent
                
                if error_rate > 0.5:
                    alerts.append({
                        "type": "high_error_rate",
                        "message": f"High error rate: {error_rate:.1%}",
                        "severity": "CRITICAL",
                    })
            
            # Check for retry storms
            if recent_errors > 50:
                # Check if errors are clustered
                error_times = sorted(self.error_counts[client_id])[-50:]
                if len(error_times) >= 2:
                    time_span = (error_times[-1] - error_times[0]).total_seconds()
                    
                    if time_span < 60:  # 50+ errors in < 1 minute
                        alerts.append({
                            "type": "retry_storm",
                            "message": f"Retry storm detected: {len(error_times)} errors in {time_span:.0f}s",
                            "severity": "CRITICAL",
                        })
        
        return {
            "client_id": client_id,
            "has_alerts": len(alerts) > 0,
            "alerts": alerts,
        }

# Integration with middleware
abuse_detector = AbuseDetector()

@app.middleware("http")
async def abuse_detection_middleware(request: Request, call_next):
    """Detect and respond to abuse patterns."""
    
    api_key = request.headers.get("X-API-Key", "anonymous")
    
    # Process request
    response = await call_next(request)
    
    # Record for abuse detection
    abuse_detector.record_request(
        client_id=api_key,
        status_code=response.status_code,
    )
    
    # Check for abuse
    abuse_result = abuse_detector.check_abuse(api_key)
    
    if abuse_result["has_alerts"]:
        for alert in abuse_result["alerts"]:
            if alert["severity"] == "CRITICAL":
                send_alert(
                    channel="pagerduty",
                    title=f"API Abuse: {alert['type']}",
                    message=f"Client {api_key}: {alert['message']}",
                )
            else:
                send_alert(
                    channel="slack",
                    title=f"API Abuse Warning: {alert['type']}",
                    message=f"Client {api_key}: {alert['message']}",
                )
    
    return response
```

### Phase 6: Partner Dashboard and Self-Service

```python
# API endpoints for partner self-service
@app.get("/v2/usage/summary")
async def get_usage_summary(
    claims: dict = Depends(validate_jwt),
    days: int = 30,
):
    """Get usage summary for authenticated partner."""
    
    client_id = claims.get("client_id")
    
    return generate_partner_usage_report(client_id, days)

@app.get("/v2/rate-limit/status")
async def get_rate_limit_status(
    claims: dict = Depends(validate_jwt),
):
    """Get current rate limit status."""
    
    client_id = claims.get("client_id")
    config = get_rate_limit_for_key(client_id)
    status = rate_limiter.get_status(client_id)
    
    return {
        "limit_per_minute": config.requests_per_minute,
        "burst_size": config.burst_size,
        "current_tokens": status.current_tokens,
        "next_refill_at": status.next_refill_at.isoformat(),
    }

@app.post("/v2/api-keys/rotate")
async def rotate_api_key(
    claims: dict = Depends(validate_jwt),
):
    """Rotate API key (self-service)."""
    
    client_id = claims.get("client_id")
    
    # Generate new key
    new_key = key_manager.rotate_key(client_id)
    
    return {
        "message": "API key rotated successfully",
        "new_key": new_key,
        "old_key_valid_until": (datetime.now() + timedelta(hours=24)).isoformat(),
    }
```

---

## Results

### Before Implementation

| Issue | Impact |
|-------|--------|
| Shared API keys | No accountability |
| No rate limiting | Cost overruns, instability |
| No usage tracking | No visibility |
| Manual key management | Security risk |

### After Implementation

| Metric | Before | After |
|--------|--------|-------|
| API cost/month | $24,000 | $12,000 |
| Unplanned outages | 3/quarter | 0/quarter |
| Time to identify abuse | Days | Real-time |
| Partner onboarding time | 2 days | 30 minutes |

### Partner B Behavior Change

| Metric | Before | After |
|--------|--------|-------|
| Requests/day | 47,000 | 1,200 |
| Retry attempts on error | 10 | 3 (with backoff) |
| Batch vs individual calls | 0% batch | 80% batch |
| Cache utilization | 0% | 85% |

---

## Key Learnings

1. **Rate limiting is essential** — Token bucket algorithm handles bursts while enforcing limits
2. **Per-client tracking enables accountability** — Usage data identifies abusers and informs pricing
3. **Self-service reduces friction** — Partners can monitor usage and rotate keys independently
4. **Abuse detection prevents outages** — Real-time detection catches retry storms before they cascade
5. **Tiered access aligns incentives** — Partners optimize their integrations when costs are visible

---

## API Gateway Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       API Gateway                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ API Key      │  │ JWT          │  │ Rate         │      │
│  │ Validation   │  │ Validation   │  │ Limiting     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐      │
│  │ Usage        │  │ Abuse        │  │ Request      │      │
│  │ Tracking     │  │ Detection    │  │ Logging      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    Backend Services                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ /predict    │  │ /batch      │  │ /explain    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## Related Modules

- `gateway/auth/api_keys.py` — API key management
- `gateway/auth/jwt_validator.py` — JWT/OAuth2 validation
- `gateway/rate_limiter.py` — Token bucket rate limiting
- `gateway/usage_tracker.py` — Usage analytics and billing

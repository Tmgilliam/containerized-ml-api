"""Pydantic schemas for ERP delay risk inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """ERP order features used for delay risk scoring."""

    model_config = ConfigDict(extra="forbid")

    order_qty: int = Field(ge=1, description="Order quantity in units")
    lead_time_days: float = Field(ge=0.0, description="Expected supplier lead time in days")
    vendor_reliability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Historical on-time delivery rate for vendor (0-1)",
    )
    days_until_due: float = Field(ge=0.0, description="Days remaining until promised delivery")
    historical_delay_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Historical delay rate for similar orders (0-1)",
    )
    inventory_buffer_days: float = Field(
        ge=0.0,
        description="Days of inventory coverage available as buffer",
    )


class PredictResponse(BaseModel):
    """Delay risk prediction with confidence and model traceability."""

    delay_risk: int = Field(description="Binary delay risk prediction (0=low, 1=high)")
    risk_score: float = Field(ge=0.0, le=1.0, description="Probability of delay")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in prediction")
    model_version: str = Field(description="Deployed model version identifier")


class HealthResponse(BaseModel):
    """Operational health payload for load balancers and deployment smoke tests."""

    status: str
    model_version: str
    uptime_seconds: float
    last_prediction_at: str | None
    environment: str

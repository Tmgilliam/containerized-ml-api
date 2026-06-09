"""Production-ready FastAPI application for containerized ML inference."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model import delay_risk_model
from app.schemas import HealthResponse, PredictRequest, PredictResponse

APP_START_TIME = time.time()
LAST_PREDICTION_AT: str | None = None
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


class JsonFormatter(logging.Formatter):
    """Emit structured JSON log lines for observability pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "model_version", "input_schema", "prediction", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger("app.main")


def _log_request_event(
    request_id: str,
    model_version: str,
    input_schema: str,
    prediction: dict | None,
    latency_ms: float,
    message: str,
) -> None:
    logger.info(
        message,
        extra={
            "request_id": request_id,
            "model_version": model_version,
            "input_schema": input_schema,
            "prediction": prediction,
            "latency_ms": round(latency_ms, 2),
        },
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Ensure model is loaded before serving traffic."""
    delay_risk_model.load()
    logger.info("Application startup complete model_version=%s", delay_risk_model.version)
    yield


app = FastAPI(
    title="Containerized ML API — ERP Delay Risk",
    description=(
        "Containerized inference API enforcing identical runtime from development "
        "through production for ERP-adjacent delay risk scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Generate a traceable request ID for every HTTP request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "request_id": request_id,
        },
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return model version, uptime, and last prediction timestamp."""
    request_id = request.state.request_id
    start = time.perf_counter()
    uptime_seconds = time.time() - APP_START_TIME

    response = HealthResponse(
        status="healthy",
        model_version=delay_risk_model.version,
        uptime_seconds=round(uptime_seconds, 2),
        last_prediction_at=LAST_PREDICTION_AT,
        environment=ENVIRONMENT,
    )

    _log_request_event(
        request_id=request_id,
        model_version=delay_risk_model.version,
        input_schema="health",
        prediction={"status": response.status},
        latency_ms=(time.perf_counter() - start) * 1000,
        message="Health check completed",
    )
    return response


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score ERP order features and return delay risk with confidence."""
    global LAST_PREDICTION_AT

    request_id = request.state.request_id
    start = time.perf_counter()
    input_schema = PredictRequest.__name__

    try:
        result = delay_risk_model.predict(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    LAST_PREDICTION_AT = datetime.now(timezone.utc).isoformat()
    response = PredictResponse(**result)
    latency_ms = (time.perf_counter() - start) * 1000

    _log_request_event(
        request_id=request_id,
        model_version=response.model_version,
        input_schema=input_schema,
        prediction=response.model_dump(),
        latency_ms=latency_ms,
        message="Prediction completed",
    )
    return response

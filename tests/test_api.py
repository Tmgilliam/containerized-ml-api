"""Unit tests for containerized ML inference API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import delay_risk_model


@pytest.fixture(scope="module", autouse=True)
def load_model_once():
    delay_risk_model.load()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_model_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_version"]
    assert payload["uptime_seconds"] >= 0
    assert "environment" in payload
    assert "X-Request-ID" in response.headers


def test_predict_returns_risk_score(client):
    response = client.post(
        "/predict",
        json={
            "order_qty": 250,
            "lead_time_days": 12.0,
            "vendor_reliability_score": 0.82,
            "days_until_due": 8.0,
            "historical_delay_rate": 0.15,
            "inventory_buffer_days": 4.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["delay_risk"] in (0, 1)
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["model_version"]


def test_predict_rejects_missing_features(client):
    response = client.post(
        "/predict",
        json={
            "order_qty": 100,
            "lead_time_days": 10.0,
        },
    )
    assert response.status_code == 422

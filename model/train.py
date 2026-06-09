"""Train ERP delay risk classifier and persist serialized artifact with metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "order_qty",
    "lead_time_days",
    "vendor_reliability_score",
    "days_until_due",
    "historical_delay_rate",
    "inventory_buffer_days",
]
TARGET_COLUMN = "delay_risk"
MODEL_VERSION = "1.0.0"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.pkl"
RANDOM_STATE = 42


def generate_mock_erp_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate realistic mock ERP delay risk training data."""
    rng = np.random.default_rng(RANDOM_STATE)

    order_qty = rng.integers(10, 5000, size=n_samples)
    lead_time_days = rng.normal(loc=14, scale=5, size=n_samples).clip(1, 60)
    vendor_reliability_score = rng.beta(5, 2, size=n_samples)
    days_until_due = rng.normal(loc=10, scale=4, size=n_samples).clip(0.5, 45)
    historical_delay_rate = rng.beta(2, 5, size=n_samples)
    inventory_buffer_days = rng.normal(loc=5, scale=2, size=n_samples).clip(0, 30)

    # Delay risk increases with long lead times, low vendor reliability,
    # tight due dates, high historical delays, and thin inventory buffers.
    risk_logit = (
        0.002 * order_qty
        + 0.08 * lead_time_days
        - 2.5 * vendor_reliability_score
        - 0.12 * days_until_due
        + 2.0 * historical_delay_rate
        - 0.15 * inventory_buffer_days
        + rng.normal(0, 0.5, size=n_samples)
    )
    delay_probability = 1 / (1 + np.exp(-risk_logit))
    delay_risk = rng.binomial(1, delay_probability)

    return pd.DataFrame(
        {
            "order_qty": order_qty,
            "lead_time_days": lead_time_days,
            "vendor_reliability_score": vendor_reliability_score,
            "days_until_due": days_until_due,
            "historical_delay_rate": historical_delay_rate,
            "inventory_buffer_days": inventory_buffer_days,
            "delay_risk": delay_risk,
        }
    )


def train_and_evaluate() -> dict:
    """Train GradientBoostingClassifier and evaluate with classification metrics."""
    df = generate_mock_erp_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    y_prob = classifier.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }

    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"],
    }

    artifact = {"model": classifier, "metadata": metadata}
    joblib.dump(artifact, OUTPUT_PATH)

    print("=== ERP Delay Risk Model — Training Summary ===")
    print(f"Samples: {len(df)} | Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Artifact saved: {OUTPUT_PATH}")
    print("--- Evaluation ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("--- Metadata ---")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()

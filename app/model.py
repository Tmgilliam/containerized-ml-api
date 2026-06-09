"""Model loading and inference for ERP delay risk prediction."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"


class DelayRiskModel:
    """Loads serialized model once and serves inference with validation."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        self._classifier = None
        self._metadata: dict[str, Any] = {}
        self._feature_names: list[str] = []
        self._version = "unknown"

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def load(self) -> None:
        """Load model artifact and metadata at startup."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict):
            self._classifier = artifact["model"]
            self._metadata = artifact.get("metadata", {})
        else:
            self._classifier = artifact
            self._metadata = {}

        self._feature_names = list(
            self._metadata.get(
                "features",
                [
                    "order_qty",
                    "lead_time_days",
                    "vendor_reliability_score",
                    "days_until_due",
                    "historical_delay_rate",
                    "inventory_buffer_days",
                ],
            )
        )
        self._version = os.getenv(
            "MODEL_VERSION",
            str(self._metadata.get("version", "unknown")),
        )
        logger.info(
            "Model loaded path=%s version=%s features=%s",
            self.model_path,
            self._version,
            self._feature_names,
        )

    def predict(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate input features, run inference, return prediction and confidence."""
        if self._classifier is None:
            raise RuntimeError("Model is not loaded")

        missing = [name for name in self._feature_names if name not in features_dict]
        if missing:
            raise ValueError(
                f"Missing required features: {', '.join(missing)}. "
                f"Expected: {', '.join(self._feature_names)}"
            )

        start = time.perf_counter()
        feature_vector = pd.DataFrame(
            [{name: float(features_dict[name]) for name in self._feature_names}]
        )
        probabilities = self._classifier.predict_proba(feature_vector)[0]
        delay_risk = int(self._classifier.predict(feature_vector)[0])
        risk_score = float(probabilities[1])
        confidence = float(max(probabilities))
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Inference completed latency_ms=%.2f delay_risk=%s risk_score=%.4f",
            latency_ms,
            delay_risk,
            risk_score,
        )

        return {
            "delay_risk": delay_risk,
            "risk_score": risk_score,
            "confidence": confidence,
            "model_version": self._version,
        }


delay_risk_model = DelayRiskModel()

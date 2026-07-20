"""Metrics collection for A/B experiments."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single prediction record for experiment tracking."""
    timestamp: str
    experiment_name: str | None
    variant_name: str
    user_id: str | None
    features: dict[str, Any]
    prediction: dict[str, Any]
    latency_ms: float
    outcome: Any | None = None
    outcome_timestamp: str | None = None


@dataclass
class VariantStats:
    """Aggregated statistics for a variant."""
    variant_name: str
    prediction_count: int = 0
    total_latency_ms: float = 0.0
    positive_predictions: int = 0
    outcomes_recorded: int = 0
    positive_outcomes: int = 0
    sum_risk_score: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.prediction_count == 0:
            return 0.0
        return self.total_latency_ms / self.prediction_count

    @property
    def positive_rate(self) -> float:
        if self.prediction_count == 0:
            return 0.0
        return self.positive_predictions / self.prediction_count

    @property
    def outcome_rate(self) -> float:
        if self.outcomes_recorded == 0:
            return 0.0
        return self.positive_outcomes / self.outcomes_recorded

    @property
    def avg_risk_score(self) -> float:
        if self.prediction_count == 0:
            return 0.0
        return self.sum_risk_score / self.prediction_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_name": self.variant_name,
            "prediction_count": self.prediction_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "positive_rate": round(self.positive_rate, 4),
            "avg_risk_score": round(self.avg_risk_score, 4),
            "outcomes_recorded": self.outcomes_recorded,
            "outcome_rate": round(self.outcome_rate, 4) if self.outcomes_recorded > 0 else None,
        }


class ExperimentMetrics:
    """
    Collects and aggregates metrics for A/B experiments.

    Thread-safe metric collection with support for:
    - Prediction logging
    - Outcome recording (feedback loop)
    - Per-variant statistics
    - Export for analysis
    """

    def __init__(self, max_records: int = 100000) -> None:
        """
        Initialize metrics collector.

        Args:
            max_records: Maximum prediction records to keep in memory
        """
        self.max_records = max_records
        self._records: list[PredictionRecord] = []
        self._stats: dict[str, dict[str, VariantStats]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._record_index: dict[str, int] = {}

    def record_prediction(
        self,
        experiment_name: str | None,
        variant_name: str,
        user_id: str | None,
        features: dict[str, Any],
        prediction: dict[str, Any],
        latency_ms: float,
        prediction_id: str | None = None,
    ) -> str:
        """
        Record a prediction event.

        Args:
            experiment_name: Name of experiment (None for control)
            variant_name: Name of the variant used
            user_id: Optional user identifier
            features: Input features
            prediction: Model output
            latency_ms: Inference latency
            prediction_id: Optional ID for outcome matching

        Returns:
            Prediction ID for outcome matching
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        record = PredictionRecord(
            timestamp=timestamp,
            experiment_name=experiment_name,
            variant_name=variant_name,
            user_id=user_id,
            features=features,
            prediction=prediction,
            latency_ms=latency_ms,
        )

        with self._lock:
            if len(self._records) >= self.max_records:
                self._records.pop(0)
                for key, idx in list(self._record_index.items()):
                    if idx == 0:
                        del self._record_index[key]
                    else:
                        self._record_index[key] = idx - 1

            record_idx = len(self._records)
            self._records.append(record)

            if prediction_id:
                self._record_index[prediction_id] = record_idx

            exp_key = experiment_name or "_control"
            if variant_name not in self._stats[exp_key]:
                self._stats[exp_key][variant_name] = VariantStats(variant_name=variant_name)

            stats = self._stats[exp_key][variant_name]
            stats.prediction_count += 1
            stats.total_latency_ms += latency_ms

            if prediction.get("delay_risk", 0) == 1:
                stats.positive_predictions += 1

            if "risk_score" in prediction:
                stats.sum_risk_score += prediction["risk_score"]

        logger.debug(
            "Recorded prediction experiment=%s variant=%s",
            experiment_name,
            variant_name,
        )

        return prediction_id or f"{timestamp}-{record_idx}"

    def record_outcome(
        self,
        prediction_id: str,
        outcome: Any,
    ) -> bool:
        """
        Record the actual outcome for a prediction (feedback loop).

        Args:
            prediction_id: ID returned from record_prediction
            outcome: Actual outcome (e.g., did delay actually occur)

        Returns:
            True if outcome was matched to a prediction
        """
        with self._lock:
            if prediction_id not in self._record_index:
                logger.warning("No prediction found for outcome: %s", prediction_id)
                return False

            idx = self._record_index[prediction_id]
            record = self._records[idx]
            record.outcome = outcome
            record.outcome_timestamp = datetime.now(timezone.utc).isoformat()

            exp_key = record.experiment_name or "_control"
            stats = self._stats[exp_key].get(record.variant_name)
            if stats:
                stats.outcomes_recorded += 1
                if outcome:
                    stats.positive_outcomes += 1

        logger.debug("Recorded outcome for prediction %s", prediction_id)
        return True

    def get_variant_stats(
        self,
        experiment_name: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Get aggregated statistics per variant.

        Args:
            experiment_name: Filter to specific experiment (None for all)

        Returns:
            Dict mapping experiment names to list of variant stats
        """
        with self._lock:
            if experiment_name:
                exp_key = experiment_name
                if exp_key in self._stats:
                    return {
                        exp_key: [s.to_dict() for s in self._stats[exp_key].values()]
                    }
                return {}

            return {
                exp: [s.to_dict() for s in variants.values()]
                for exp, variants in self._stats.items()
            }

    def get_records(
        self,
        experiment_name: str | None = None,
        variant_name: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get raw prediction records for analysis.

        Args:
            experiment_name: Filter by experiment
            variant_name: Filter by variant
            limit: Maximum records to return
            offset: Pagination offset

        Returns:
            List of prediction records as dicts
        """
        with self._lock:
            filtered = self._records

            if experiment_name:
                filtered = [r for r in filtered if r.experiment_name == experiment_name]
            if variant_name:
                filtered = [r for r in filtered if r.variant_name == variant_name]

            paginated = filtered[offset:offset + limit]

            return [
                {
                    "timestamp": r.timestamp,
                    "experiment_name": r.experiment_name,
                    "variant_name": r.variant_name,
                    "user_id": r.user_id,
                    "prediction": r.prediction,
                    "latency_ms": r.latency_ms,
                    "outcome": r.outcome,
                    "outcome_timestamp": r.outcome_timestamp,
                }
                for r in paginated
            ]

    def export_for_analysis(
        self,
        experiment_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Export experiment data grouped by variant for statistical analysis.

        Args:
            experiment_name: Experiment to export

        Returns:
            Dict mapping variant names to their prediction records
        """
        with self._lock:
            result: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for record in self._records:
                if record.experiment_name == experiment_name:
                    result[record.variant_name].append({
                        "risk_score": record.prediction.get("risk_score"),
                        "delay_risk": record.prediction.get("delay_risk"),
                        "outcome": record.outcome,
                        "latency_ms": record.latency_ms,
                    })

            return dict(result)

    def reset(self, experiment_name: str | None = None) -> None:
        """Reset metrics (optionally for a specific experiment)."""
        with self._lock:
            if experiment_name:
                exp_key = experiment_name
                self._records = [
                    r for r in self._records
                    if r.experiment_name != experiment_name
                ]
                if exp_key in self._stats:
                    del self._stats[exp_key]
            else:
                self._records.clear()
                self._stats.clear()
                self._record_index.clear()

        logger.info("Reset metrics for experiment=%s", experiment_name or "all")

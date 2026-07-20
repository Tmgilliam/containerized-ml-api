"""Shadow result comparison and analysis."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Aggregated comparison results."""
    total_comparisons: int
    prediction_agreement_rate: float
    mean_probability_difference: float
    std_probability_difference: float
    max_probability_difference: float
    production_positive_rate: float
    shadow_positive_rate: float
    confusion_matrix: dict[str, int]
    percentile_diffs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_comparisons": self.total_comparisons,
            "prediction_agreement_rate": round(self.prediction_agreement_rate, 4),
            "mean_probability_difference": round(self.mean_probability_difference, 4),
            "std_probability_difference": round(self.std_probability_difference, 4),
            "max_probability_difference": round(self.max_probability_difference, 4),
            "production_positive_rate": round(self.production_positive_rate, 4),
            "shadow_positive_rate": round(self.shadow_positive_rate, 4),
            "confusion_matrix": self.confusion_matrix,
            "percentile_diffs": {k: round(v, 4) for k, v in self.percentile_diffs.items()},
        }


class ShadowComparator:
    """
    Compares production and shadow model predictions.

    Features:
    - Agreement rate calculation
    - Distribution comparison
    - Statistical analysis
    - Drift detection between models
    """

    def __init__(
        self,
        agreement_threshold: float = 0.95,
        probability_diff_threshold: float = 0.1,
    ) -> None:
        """
        Initialize comparator.

        Args:
            agreement_threshold: Minimum agreement rate to consider models equivalent
            probability_diff_threshold: Max probability difference to consider acceptable
        """
        self.agreement_threshold = agreement_threshold
        self.probability_diff_threshold = probability_diff_threshold

        self._comparisons: list[dict[str, Any]] = []

    def add_comparison(
        self,
        production_result: dict[str, Any],
        shadow_result: dict[str, Any],
        features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a single comparison result.

        Args:
            production_result: Production model output
            shadow_result: Shadow model output
            features: Optional input features for analysis

        Returns:
            Comparison details
        """
        prod_class = production_result.get("delay_risk", production_result.get("prediction"))
        shadow_class = shadow_result.get("delay_risk", shadow_result.get("prediction"))

        prod_prob = production_result.get("risk_score", production_result.get("probability", 0))
        shadow_prob = shadow_result.get("risk_score", shadow_result.get("probability", 0))

        comparison = {
            "production_class": prod_class,
            "shadow_class": shadow_class,
            "predictions_match": prod_class == shadow_class,
            "production_probability": prod_prob,
            "shadow_probability": shadow_prob,
            "probability_difference": abs(prod_prob - shadow_prob),
            "features": features,
        }

        self._comparisons.append(comparison)
        return comparison

    def analyze(self) -> ComparisonResult:
        """
        Analyze all collected comparisons.

        Returns:
            ComparisonResult with aggregate statistics
        """
        if not self._comparisons:
            return ComparisonResult(
                total_comparisons=0,
                prediction_agreement_rate=0.0,
                mean_probability_difference=0.0,
                std_probability_difference=0.0,
                max_probability_difference=0.0,
                production_positive_rate=0.0,
                shadow_positive_rate=0.0,
                confusion_matrix={},
                percentile_diffs={},
            )

        n = len(self._comparisons)

        agreements = sum(1 for c in self._comparisons if c["predictions_match"])
        agreement_rate = agreements / n

        prob_diffs = [c["probability_difference"] for c in self._comparisons]
        mean_diff = np.mean(prob_diffs)
        std_diff = np.std(prob_diffs)
        max_diff = max(prob_diffs)

        prod_positives = sum(1 for c in self._comparisons if c["production_class"] == 1)
        shadow_positives = sum(1 for c in self._comparisons if c["shadow_class"] == 1)

        confusion = {
            "both_positive": 0,
            "both_negative": 0,
            "prod_pos_shadow_neg": 0,
            "prod_neg_shadow_pos": 0,
        }

        for c in self._comparisons:
            if c["production_class"] == 1 and c["shadow_class"] == 1:
                confusion["both_positive"] += 1
            elif c["production_class"] == 0 and c["shadow_class"] == 0:
                confusion["both_negative"] += 1
            elif c["production_class"] == 1 and c["shadow_class"] == 0:
                confusion["prod_pos_shadow_neg"] += 1
            else:
                confusion["prod_neg_shadow_pos"] += 1

        percentiles = {
            "p50": np.percentile(prob_diffs, 50),
            "p90": np.percentile(prob_diffs, 90),
            "p95": np.percentile(prob_diffs, 95),
            "p99": np.percentile(prob_diffs, 99),
        }

        return ComparisonResult(
            total_comparisons=n,
            prediction_agreement_rate=agreement_rate,
            mean_probability_difference=float(mean_diff),
            std_probability_difference=float(std_diff),
            max_probability_difference=float(max_diff),
            production_positive_rate=prod_positives / n,
            shadow_positive_rate=shadow_positives / n,
            confusion_matrix=confusion,
            percentile_diffs=percentiles,
        )

    def is_shadow_ready_for_promotion(self) -> tuple[bool, list[str]]:
        """
        Determine if shadow model is ready for promotion to production.

        Returns:
            Tuple of (is_ready, list of issues)
        """
        result = self.analyze()
        issues = []

        if result.total_comparisons < 100:
            issues.append(f"Insufficient comparisons: {result.total_comparisons} < 100")

        if result.prediction_agreement_rate < self.agreement_threshold:
            rate = result.prediction_agreement_rate
            thresh = self.agreement_threshold
            issues.append(f"Agreement rate too low: {rate:.2%} < {thresh:.2%}")

        if result.mean_probability_difference > self.probability_diff_threshold:
            diff = result.mean_probability_difference
            thresh = self.probability_diff_threshold
            issues.append(f"Mean probability difference too high: {diff:.4f} > {thresh}")

        rate_diff = abs(result.production_positive_rate - result.shadow_positive_rate)
        if rate_diff > 0.05:
            issues.append(
                f"Positive rate differs significantly: {rate_diff:.2%} difference"
            )

        is_ready = len(issues) == 0

        logger.info(
            "Shadow promotion check: ready=%s issues=%d",
            is_ready,
            len(issues),
        )

        return is_ready, issues

    def get_disagreements(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get examples where models disagreed."""
        disagreements = [
            c for c in self._comparisons
            if not c["predictions_match"]
        ]

        disagreements.sort(key=lambda x: x["probability_difference"], reverse=True)

        return disagreements[:limit]

    def get_high_difference_examples(
        self,
        threshold: float = 0.2,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get examples with high probability difference."""
        high_diff = [
            c for c in self._comparisons
            if c["probability_difference"] > threshold
        ]

        high_diff.sort(key=lambda x: x["probability_difference"], reverse=True)

        return high_diff[:limit]

    def segment_analysis(
        self,
        segment_feature: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze comparisons by a feature segment.

        Args:
            segment_feature: Feature name to segment by

        Returns:
            Dict mapping segment values to statistics
        """
        segments: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for c in self._comparisons:
            if c.get("features") and segment_feature in c["features"]:
                value = c["features"][segment_feature]
                if isinstance(value, (int, float)):
                    if value < 0.33:
                        bucket = "low"
                    elif value < 0.66:
                        bucket = "medium"
                    else:
                        bucket = "high"
                else:
                    bucket = str(value)

                segments[bucket].append(c)

        results = {}
        for segment, comparisons in segments.items():
            n = len(comparisons)
            if n == 0:
                continue

            agreements = sum(1 for c in comparisons if c["predictions_match"])
            prob_diffs = [c["probability_difference"] for c in comparisons]

            results[segment] = {
                "count": n,
                "agreement_rate": round(agreements / n, 4),
                "mean_prob_diff": round(np.mean(prob_diffs), 4),
                "max_prob_diff": round(max(prob_diffs), 4),
            }

        return results

    def clear(self) -> None:
        """Clear all stored comparisons."""
        self._comparisons.clear()
        logger.info("Cleared comparison data")

    def export(self) -> list[dict[str, Any]]:
        """Export all comparison data."""
        return list(self._comparisons)

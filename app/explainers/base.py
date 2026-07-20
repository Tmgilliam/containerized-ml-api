"""Base classes for model explainability."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureContribution:
    """Contribution of a single feature to prediction."""
    feature_name: str
    feature_value: Any
    contribution: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    """Complete explanation for a prediction."""
    prediction: int | float
    probability: float
    base_value: float
    feature_contributions: list[FeatureContribution] = field(default_factory=list)
    model_version: str = "unknown"
    explanation_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_positive_features(self) -> list[FeatureContribution]:
        """Get features that increased the prediction."""
        return sorted(
            [f for f in self.feature_contributions if f.contribution > 0],
            key=lambda x: abs(x.contribution),
            reverse=True,
        )

    @property
    def top_negative_features(self) -> list[FeatureContribution]:
        """Get features that decreased the prediction."""
        return sorted(
            [f for f in self.feature_contributions if f.contribution < 0],
            key=lambda x: abs(x.contribution),
            reverse=True,
        )

    def summary(self, top_n: int = 3) -> str:
        """Generate human-readable summary."""
        lines = []

        if self.prediction == 1:
            lines.append(f"High delay risk ({self.probability:.1%} probability)")
        else:
            lines.append(f"Low delay risk ({1-self.probability:.1%} confidence)")

        positive = self.top_positive_features[:top_n]
        if positive:
            lines.append("\nFactors increasing risk:")
            for f in positive:
                lines.append(f"  - {f.feature_name}: {f.feature_value} (+{f.contribution:.3f})")

        negative = self.top_negative_features[:top_n]
        if negative:
            lines.append("\nFactors decreasing risk:")
            for f in negative:
                lines.append(f"  - {f.feature_name}: {f.feature_value} ({f.contribution:.3f})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "base_value": round(self.base_value, 4),
            "feature_contributions": [f.to_dict() for f in self.feature_contributions],
            "model_version": self.model_version,
            "explanation_type": self.explanation_type,
            "top_positive": [f.to_dict() for f in self.top_positive_features[:3]],
            "top_negative": [f.to_dict() for f in self.top_negative_features[:3]],
            "metadata": self.metadata,
        }


class BaseExplainer(ABC):
    """Base class for model explainers."""

    @abstractmethod
    def explain(
        self,
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> ExplanationResult:
        """Generate explanation for a prediction."""
        pass

    @abstractmethod
    def explain_batch(
        self,
        records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[ExplanationResult]:
        """Generate explanations for multiple predictions."""
        pass

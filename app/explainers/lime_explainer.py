"""LIME-based model explanations."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.explainers.base import (
    BaseExplainer,
    ExplanationResult,
    FeatureContribution,
)

logger = logging.getLogger(__name__)


class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer.

    Creates local linear approximations to explain individual predictions.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        training_data: pd.DataFrame | None = None,
        model_version: str = "unknown",
        mode: str = "classification",
    ) -> None:
        """
        Initialize LIME explainer.

        Args:
            predict_fn: Function that takes (n_samples, n_features) and returns probabilities
            feature_names: List of feature names in order
            training_data: Training data for computing statistics
            model_version: Model version for tracking
            mode: 'classification' or 'regression'
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.training_data = training_data
        self.model_version = model_version
        self.mode = mode
        self._explainer = None

    def _get_explainer(self):
        """Lazily initialize LIME explainer."""
        if self._explainer is not None:
            return self._explainer

        try:
            from lime.lime_tabular import LimeTabularExplainer

            if self.training_data is not None:
                training_array = self.training_data[self.feature_names].values
            else:
                training_array = np.zeros((10, len(self.feature_names)))

            self._explainer = LimeTabularExplainer(
                training_array,
                feature_names=self.feature_names,
                mode=self.mode,
                discretize_continuous=True,
            )

            logger.info("Initialized LIME explainer")
            return self._explainer

        except ImportError:
            logger.warning("LIME not installed, explanations will be limited")
            return None

    def _fallback_explain(
        self,
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> ExplanationResult:
        """Fallback when LIME is not available."""
        probability = prediction.get("risk_score", prediction.get("probability", 0.5))
        pred_class = prediction.get("delay_risk", prediction.get("prediction", 0))

        contributions = []
        for name in self.feature_names:
            value = features.get(name, 0)
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=value,
                contribution=0.0,
                direction="neutral",
            ))

        return ExplanationResult(
            prediction=pred_class,
            probability=probability,
            base_value=0.5,
            feature_contributions=contributions,
            model_version=self.model_version,
            explanation_type="lime_unavailable",
        )

    def explain(
        self,
        features: dict[str, Any],
        prediction: dict[str, Any],
        num_features: int = 6,
        num_samples: int = 5000,
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a single prediction.

        Args:
            features: Input feature dictionary
            prediction: Model prediction result
            num_features: Number of top features to include
            num_samples: Number of samples for local approximation

        Returns:
            ExplanationResult with LIME coefficients
        """
        explainer = self._get_explainer()

        if explainer is None:
            return self._fallback_explain(features, prediction)

        try:
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names])

            exp = explainer.explain_instance(
                feature_vector,
                self.predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

            lime_weights = dict(exp.as_list())

            contributions = []
            for name in self.feature_names:
                value = features.get(name, 0)

                weight = 0.0
                for lime_name, lime_weight in lime_weights.items():
                    if name in lime_name:
                        weight = lime_weight
                        break

                contributions.append(FeatureContribution(
                    feature_name=name,
                    feature_value=value,
                    contribution=weight,
                    direction="positive" if weight > 0 else "negative",
                ))

            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

            probability = prediction.get("risk_score", prediction.get("probability", 0.5))
            pred_class = prediction.get("delay_risk", prediction.get("prediction", 0))

            local_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') else probability

            return ExplanationResult(
                prediction=pred_class,
                probability=probability,
                base_value=exp.intercept[1] if hasattr(exp, 'intercept') else 0.5,
                feature_contributions=contributions,
                model_version=self.model_version,
                explanation_type="lime",
                metadata={
                    "local_prediction": float(local_pred),
                    "r2_score": float(exp.score) if hasattr(exp, 'score') else None,
                    "num_samples": num_samples,
                },
            )

        except Exception as e:
            logger.error("LIME explanation failed: %s", e)
            return self._fallback_explain(features, prediction)

    def explain_batch(
        self,
        records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        num_features: int = 6,
    ) -> list[ExplanationResult]:
        """Generate LIME explanations for multiple predictions."""
        return [
            self.explain(features, pred, num_features=num_features)
            for features, pred in zip(records, predictions)
        ]

    def counterfactual_analysis(
        self,
        features: dict[str, Any],
        target_class: int = 0,
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        """
        Analyze what feature changes would flip the prediction.

        Args:
            features: Current feature values
            target_class: Desired prediction class
            num_samples: Samples for analysis

        Returns:
            Dict with suggested changes and their effects
        """
        feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])

        current_prob = self.predict_fn(feature_vector)[0]
        current_class = 1 if current_prob[1] > 0.5 else 0

        if current_class == target_class:
            return {
                "status": "already_at_target",
                "current_class": current_class,
                "current_probability": float(current_prob[1]),
            }

        suggestions = []

        for i, name in enumerate(self.feature_names):
            original = features.get(name, 0)

            for delta_pct in [-50, -25, 25, 50]:
                modified = original * (1 + delta_pct / 100)
                modified_vector = feature_vector.copy()
                modified_vector[0, i] = modified

                new_prob = self.predict_fn(modified_vector)[0]
                new_class = 1 if new_prob[1] > 0.5 else 0

                if new_class == target_class:
                    suggestions.append({
                        "feature": name,
                        "original_value": original,
                        "suggested_value": modified,
                        "change_percent": delta_pct,
                        "new_probability": float(new_prob[1]),
                    })

        suggestions.sort(key=lambda x: abs(x["change_percent"]))

        return {
            "status": "suggestions_found" if suggestions else "no_simple_changes",
            "current_class": current_class,
            "current_probability": float(current_prob[1]),
            "target_class": target_class,
            "suggestions": suggestions[:5],
        }

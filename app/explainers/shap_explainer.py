"""SHAP-based model explanations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.explainers.base import (
    BaseExplainer,
    ExplanationResult,
    FeatureContribution,
)

logger = logging.getLogger(__name__)


class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) explainer.
    
    Provides consistent, game-theoretic feature attributions
    that sum to the difference between prediction and base value.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pd.DataFrame | None = None,
        model_version: str = "unknown",
    ) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict/predict_proba
            feature_names: List of feature names in order
            background_data: Optional background dataset for KernelSHAP
            model_version: Model version for tracking
        """
        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version
        self._explainer = None
        self._background_data = background_data
        
    def _get_explainer(self):
        """Lazily initialize SHAP explainer."""
        if self._explainer is not None:
            return self._explainer
        
        try:
            import shap
            
            model_type = type(self.model).__name__.lower()
            
            if "tree" in model_type or "forest" in model_type or "gradient" in model_type:
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer for %s", model_type)
            elif self._background_data is not None:
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    self._background_data,
                )
                logger.info("Using KernelExplainer with background data")
            else:
                background = np.zeros((1, len(self.feature_names)))
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background,
                )
                logger.info("Using KernelExplainer with zero background")
            
            return self._explainer
            
        except ImportError:
            logger.warning("SHAP not installed, using fallback explainer")
            return None
    
    def _fallback_explain(
        self,
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> ExplanationResult:
        """
        Fallback explanation using feature importance approximation.
        
        Uses model coefficients or feature importances if available.
        """
        probability = prediction.get("risk_score", prediction.get("probability", 0.5))
        pred_class = prediction.get("delay_risk", prediction.get("prediction", 0))
        
        contributions = []
        
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for i, name in enumerate(self.feature_names):
                value = features.get(name, 0)
                importance = importances[i] if i < len(importances) else 0
                
                sign = 1 if probability > 0.5 else -1
                contribution = importance * sign * (probability - 0.5)
                
                contributions.append(FeatureContribution(
                    feature_name=name,
                    feature_value=value,
                    contribution=contribution,
                    direction="positive" if contribution > 0 else "negative",
                ))
        else:
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
            explanation_type="feature_importance_fallback",
        )
    
    def explain(
        self,
        features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            features: Input feature dictionary
            prediction: Model prediction result
        
        Returns:
            ExplanationResult with SHAP values
        """
        explainer = self._get_explainer()
        
        if explainer is None:
            return self._fallback_explain(features, prediction)
        
        try:
            import shap
            
            feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])
            
            shap_values = explainer.shap_values(feature_vector)
            
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]
            
            if hasattr(explainer, "expected_value"):
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_value = float(explainer.expected_value[1])
                else:
                    base_value = float(explainer.expected_value)
            else:
                base_value = 0.5
            
            contributions = []
            for i, name in enumerate(self.feature_names):
                value = features.get(name, 0)
                shap_value = float(values[i]) if i < len(values) else 0
                
                contributions.append(FeatureContribution(
                    feature_name=name,
                    feature_value=value,
                    contribution=shap_value,
                    direction="positive" if shap_value > 0 else "negative",
                ))
            
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
            
            probability = prediction.get("risk_score", prediction.get("probability", 0.5))
            pred_class = prediction.get("delay_risk", prediction.get("prediction", 0))
            
            return ExplanationResult(
                prediction=pred_class,
                probability=probability,
                base_value=base_value,
                feature_contributions=contributions,
                model_version=self.model_version,
                explanation_type="shap",
                metadata={
                    "shap_sum": sum(c.contribution for c in contributions),
                },
            )
            
        except Exception as e:
            logger.error("SHAP explanation failed: %s", e)
            return self._fallback_explain(features, prediction)
    
    def explain_batch(
        self,
        records: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[ExplanationResult]:
        """Generate explanations for multiple predictions."""
        return [
            self.explain(features, pred)
            for features, pred in zip(records, predictions)
        ]
    
    def feature_importance_global(
        self,
        data: pd.DataFrame,
        max_samples: int = 1000,
    ) -> dict[str, float]:
        """
        Calculate global feature importance using SHAP.
        
        Args:
            data: Dataset to compute importance over
            max_samples: Maximum samples to use
        
        Returns:
            Dict mapping feature names to mean absolute SHAP values
        """
        explainer = self._get_explainer()
        
        if explainer is None:
            if hasattr(self.model, "feature_importances_"):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return {name: 0.0 for name in self.feature_names}
        
        try:
            if len(data) > max_samples:
                data = data.sample(n=max_samples, random_state=42)
            
            shap_values = explainer.shap_values(data[self.feature_names])
            
            if isinstance(shap_values, list):
                values = np.abs(shap_values[1]).mean(axis=0)
            else:
                values = np.abs(shap_values).mean(axis=0)
            
            return dict(zip(self.feature_names, values))
            
        except Exception as e:
            logger.error("Global importance calculation failed: %s", e)
            return {name: 0.0 for name in self.feature_names}

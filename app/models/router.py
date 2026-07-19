"""Multi-model routing for inference."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.models.registry import ModelRegistry, LoadedModel

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes prediction requests to appropriate models.
    
    Features:
    - Model selection by name/version
    - Automatic version resolution
    - Request routing based on input features
    - Latency tracking
    """
    
    def __init__(self, registry: ModelRegistry) -> None:
        """
        Initialize model router.
        
        Args:
            registry: Model registry containing available models
        """
        self.registry = registry
        self._routing_rules: list[dict[str, Any]] = []
    
    def add_routing_rule(
        self,
        condition: dict[str, Any],
        model_name: str,
        model_version: str | None = None,
        priority: int = 0,
    ) -> None:
        """
        Add a routing rule for automatic model selection.
        
        Args:
            condition: Dict of feature conditions (e.g., {"order_qty": {"gt": 1000}})
            model_name: Target model name
            model_version: Target version (uses default if not specified)
            priority: Rule priority (higher = checked first)
        """
        self._routing_rules.append({
            "condition": condition,
            "model_name": model_name,
            "model_version": model_version,
            "priority": priority,
        })
        
        self._routing_rules.sort(key=lambda r: r["priority"], reverse=True)
        
        logger.info(
            "Added routing rule for %s:%s with priority %d",
            model_name,
            model_version or "default",
            priority,
        )
    
    def _evaluate_condition(
        self,
        condition: dict[str, Any],
        features: dict[str, Any],
    ) -> bool:
        """Evaluate a routing condition against features."""
        for feature_name, operators in condition.items():
            if feature_name not in features:
                return False
            
            value = features[feature_name]
            
            if isinstance(operators, dict):
                for op, threshold in operators.items():
                    if op == "eq" and value != threshold:
                        return False
                    if op == "ne" and value == threshold:
                        return False
                    if op == "gt" and value <= threshold:
                        return False
                    if op == "gte" and value < threshold:
                        return False
                    if op == "lt" and value >= threshold:
                        return False
                    if op == "lte" and value > threshold:
                        return False
                    if op == "in" and value not in threshold:
                        return False
            else:
                if value != operators:
                    return False
        
        return True
    
    def _select_model(
        self,
        features: dict[str, Any],
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> tuple[str, str | None]:
        """Select model based on routing rules or explicit specification."""
        if model_name:
            return model_name, model_version
        
        for rule in self._routing_rules:
            if self._evaluate_condition(rule["condition"], features):
                logger.debug(
                    "Routing rule matched: %s -> %s:%s",
                    rule["condition"],
                    rule["model_name"],
                    rule["model_version"],
                )
                return rule["model_name"], rule["model_version"]
        
        models = self.registry.list_models()
        if models:
            default = next(
                (m for m in models if m["is_default"]),
                models[0],
            )
            return default["name"], None
        
        raise ValueError("No models available for routing")
    
    def predict(
        self,
        features: dict[str, Any],
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Route prediction request to appropriate model.
        
        Args:
            features: Input feature dictionary
            model_name: Optional explicit model name
            model_version: Optional explicit version
        
        Returns:
            Prediction result with model info and latency
        """
        start = time.perf_counter()
        
        selected_name, selected_version = self._select_model(
            features, model_name, model_version
        )
        
        model = self.registry.load(selected_name, selected_version)
        
        result = model.predict(features)
        
        latency_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        
        logger.info(
            "Prediction routed to %s:%s latency=%.2fms",
            selected_name,
            selected_version or "default",
            latency_ms,
        )
        
        return result
    
    def predict_batch(
        self,
        records: list[dict[str, Any]],
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run batch predictions through the router.
        
        Args:
            records: List of feature dictionaries
            model_name: Optional explicit model name (applied to all)
            model_version: Optional explicit version
        
        Returns:
            List of prediction results
        """
        results = []
        
        for features in records:
            try:
                result = self.predict(features, model_name, model_version)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "features": features,
                })
        
        return results
    
    def predict_multi(
        self,
        features: dict[str, Any],
        model_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Run prediction on multiple models for comparison.
        
        Args:
            features: Input feature dictionary
            model_names: List of model names to query
        
        Returns:
            Dict mapping model name to prediction result
        """
        results = {}
        
        for name in model_names:
            try:
                results[name] = self.predict(features, model_name=name)
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    
    def get_routing_rules(self) -> list[dict[str, Any]]:
        """Get all configured routing rules."""
        return [
            {
                "condition": r["condition"],
                "model_name": r["model_name"],
                "model_version": r["model_version"],
                "priority": r["priority"],
            }
            for r in self._routing_rules
        ]
    
    def clear_routing_rules(self) -> None:
        """Clear all routing rules."""
        self._routing_rules.clear()
        logger.info("Cleared all routing rules")

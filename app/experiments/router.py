"""Experiment routing for A/B testing model variants."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ModelVariant:
    """A model variant in an experiment."""
    name: str
    model_version: str
    predict_fn: Callable[[dict[str, Any]], dict[str, Any]]
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """An A/B experiment configuration."""
    name: str
    variants: list[ModelVariant]
    start_time: datetime
    end_time: datetime | None = None
    enabled: bool = True
    sticky_assignment: bool = True
    
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        if not self.enabled:
            return False
        now = datetime.now(timezone.utc)
        if now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False
        return True
    
    @property
    def total_weight(self) -> float:
        return sum(v.weight for v in self.variants)


class ExperimentRouter:
    """
    Routes requests to model variants for A/B testing.
    
    Supports:
    - Weighted random assignment
    - Sticky assignment (same user gets same variant)
    - Multiple concurrent experiments
    - Graceful fallback to control
    """
    
    def __init__(self, default_model: ModelVariant) -> None:
        """
        Initialize router with a default (control) model.
        
        Args:
            default_model: The control model used when no experiment matches
        """
        self.default_model = default_model
        self._experiments: dict[str, Experiment] = {}
        self._assignments: dict[str, dict[str, str]] = {}
        
    def register_experiment(self, experiment: Experiment) -> None:
        """Register an experiment."""
        self._experiments[experiment.name] = experiment
        logger.info(
            "Registered experiment %s with %d variants",
            experiment.name,
            len(experiment.variants),
        )
        
    def deactivate_experiment(self, experiment_name: str) -> None:
        """Deactivate an experiment."""
        if experiment_name in self._experiments:
            self._experiments[experiment_name].enabled = False
            logger.info("Deactivated experiment %s", experiment_name)
            
    def _generate_assignment_key(self, user_id: str, experiment_name: str) -> str:
        """Generate a deterministic hash for sticky assignment."""
        combined = f"{user_id}:{experiment_name}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _select_variant(
        self,
        experiment: Experiment,
        user_id: str | None,
    ) -> ModelVariant:
        """Select a variant based on weights and sticky assignment."""
        if experiment.sticky_assignment and user_id:
            cached = self._assignments.get(user_id, {}).get(experiment.name)
            if cached:
                for variant in experiment.variants:
                    if variant.name == cached:
                        return variant
            
            hash_key = self._generate_assignment_key(user_id, experiment.name)
            hash_value = int(hash_key, 16) / (16 ** len(hash_key))
            
            cumulative = 0.0
            for variant in experiment.variants:
                cumulative += variant.weight / experiment.total_weight
                if hash_value < cumulative:
                    if user_id not in self._assignments:
                        self._assignments[user_id] = {}
                    self._assignments[user_id][experiment.name] = variant.name
                    return variant
        
        weights = [v.weight for v in experiment.variants]
        return random.choices(experiment.variants, weights=weights)[0]
    
    def route(
        self,
        features: dict[str, Any],
        user_id: str | None = None,
        experiment_name: str | None = None,
    ) -> tuple[dict[str, Any], str, str | None]:
        """
        Route a prediction request to the appropriate model variant.
        
        Args:
            features: Input features for prediction
            user_id: Optional user ID for sticky assignment
            experiment_name: Optional specific experiment to use
        
        Returns:
            Tuple of (prediction_result, variant_name, experiment_name)
        """
        active_experiments = [
            (name, exp) for name, exp in self._experiments.items()
            if exp.is_active() and (experiment_name is None or name == experiment_name)
        ]
        
        if not active_experiments:
            result = self.default_model.predict_fn(features)
            return result, self.default_model.name, None
        
        exp_name, experiment = active_experiments[0]
        variant = self._select_variant(experiment, user_id)
        
        try:
            result = variant.predict_fn(features)
            logger.info(
                "Routed to experiment=%s variant=%s user=%s",
                exp_name,
                variant.name,
                user_id or "anonymous",
            )
            return result, variant.name, exp_name
            
        except Exception as e:
            logger.error(
                "Variant %s failed, falling back to control: %s",
                variant.name,
                e,
            )
            result = self.default_model.predict_fn(features)
            return result, self.default_model.name, None
    
    def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get information about all active experiments."""
        return [
            {
                "name": name,
                "variants": [v.name for v in exp.variants],
                "weights": [v.weight for v in exp.variants],
                "start_time": exp.start_time.isoformat(),
                "end_time": exp.end_time.isoformat() if exp.end_time else None,
            }
            for name, exp in self._experiments.items()
            if exp.is_active()
        ]
    
    def get_assignment(self, user_id: str) -> dict[str, str]:
        """Get all experiment assignments for a user."""
        return dict(self._assignments.get(user_id, {}))

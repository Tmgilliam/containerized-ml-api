"""Feature transformations and derived feature computation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for a feature transformation."""
    name: str
    input_features: list[str]
    output_feature: str
    transform_fn: Callable[..., Any]
    description: str = ""


class FeatureTransformer:
    """
    Applies transformations to compute derived features.
    
    Supports:
    - Mathematical transformations
    - Aggregations
    - Normalization
    - Custom functions
    """
    
    def __init__(self) -> None:
        self._transforms: dict[str, TransformConfig] = {}
        self._register_builtin_transforms()
    
    def _register_builtin_transforms(self) -> None:
        """Register common built-in transformations."""
        
        self.register_transform(TransformConfig(
            name="log1p",
            input_features=["value"],
            output_feature="log_value",
            transform_fn=lambda x: math.log1p(x) if x >= 0 else 0,
            description="Natural log of (1 + x)",
        ))
        
        self.register_transform(TransformConfig(
            name="normalize",
            input_features=["value", "min", "max"],
            output_feature="normalized",
            transform_fn=lambda v, mn, mx: (v - mn) / (mx - mn) if mx != mn else 0,
            description="Min-max normalization to [0, 1]",
        ))
        
        self.register_transform(TransformConfig(
            name="ratio",
            input_features=["numerator", "denominator"],
            output_feature="ratio",
            transform_fn=lambda n, d: n / d if d != 0 else 0,
            description="Safe division (returns 0 if denominator is 0)",
        ))
        
        self.register_transform(TransformConfig(
            name="clip",
            input_features=["value", "min", "max"],
            output_feature="clipped",
            transform_fn=lambda v, mn, mx: max(mn, min(mx, v)),
            description="Clip value to [min, max] range",
        ))
        
        self.register_transform(TransformConfig(
            name="bucket",
            input_features=["value", "boundaries"],
            output_feature="bucket",
            transform_fn=self._bucket_fn,
            description="Assign value to bucket based on boundaries",
        ))
    
    @staticmethod
    def _bucket_fn(value: float, boundaries: list[float]) -> int:
        """Assign value to bucket."""
        for i, boundary in enumerate(boundaries):
            if value < boundary:
                return i
        return len(boundaries)
    
    def register_transform(self, config: TransformConfig) -> None:
        """Register a transformation."""
        self._transforms[config.name] = config
        logger.debug("Registered transform: %s", config.name)
    
    def apply_transform(
        self,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """
        Apply a registered transformation.
        
        Args:
            name: Transform name
            **kwargs: Input feature values
        
        Returns:
            Transformed value
        """
        if name not in self._transforms:
            raise ValueError(f"Unknown transform: {name}")
        
        config = self._transforms[name]
        
        args = []
        for feature in config.input_features:
            if feature not in kwargs:
                raise ValueError(f"Missing input feature: {feature}")
            args.append(kwargs[feature])
        
        return config.transform_fn(*args)
    
    def compute_derived_features(
        self,
        features: dict[str, Any],
        derived_specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compute multiple derived features.
        
        Args:
            features: Input feature dictionary
            derived_specs: List of derivation specs, each with:
                - output: output feature name
                - transform: transform name
                - inputs: dict mapping transform inputs to feature names
        
        Returns:
            Features dict with derived features added
        """
        result = dict(features)
        
        for spec in derived_specs:
            output_name = spec["output"]
            transform_name = spec["transform"]
            input_mapping = spec["inputs"]
            
            kwargs = {}
            for param_name, feature_name in input_mapping.items():
                if feature_name in result:
                    kwargs[param_name] = result[feature_name]
                else:
                    logger.warning(
                        "Missing feature %s for derived feature %s",
                        feature_name,
                        output_name,
                    )
                    kwargs[param_name] = 0
            
            try:
                result[output_name] = self.apply_transform(transform_name, **kwargs)
            except Exception as e:
                logger.error(
                    "Failed to compute derived feature %s: %s",
                    output_name,
                    e,
                )
                result[output_name] = None
        
        return result
    
    def list_transforms(self) -> list[dict[str, Any]]:
        """List all registered transforms."""
        return [
            {
                "name": config.name,
                "input_features": config.input_features,
                "output_feature": config.output_feature,
                "description": config.description,
            }
            for config in self._transforms.values()
        ]


def create_erp_feature_transformer() -> FeatureTransformer:
    """Create transformer with ERP-specific derived features."""
    transformer = FeatureTransformer()
    
    transformer.register_transform(TransformConfig(
        name="buffer_ratio",
        input_features=["buffer_days", "lead_time"],
        output_feature="buffer_lead_ratio",
        transform_fn=lambda b, l: b / l if l > 0 else 0,
        description="Ratio of buffer days to lead time",
    ))
    
    transformer.register_transform(TransformConfig(
        name="urgency_score",
        input_features=["days_until_due", "lead_time"],
        output_feature="urgency",
        transform_fn=lambda due, lead: max(0, 1 - (due / lead)) if lead > 0 else 1,
        description="Urgency score based on time remaining vs lead time",
    ))
    
    transformer.register_transform(TransformConfig(
        name="risk_adjusted_qty",
        input_features=["qty", "delay_rate"],
        output_feature="risk_adj_qty",
        transform_fn=lambda q, r: q * (1 + r),
        description="Order quantity adjusted for historical delay rate",
    ))
    
    return transformer

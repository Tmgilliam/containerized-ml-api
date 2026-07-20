"""Feature validation for schema enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from features.registry import FeatureRegistry, FeatureGroup, FeatureDefinition

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation error."""
    feature_name: str
    error_type: str
    message: str
    value: Any = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "error_type": self.error_type,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
        }


@dataclass
class ValidationResult:
    """Result of feature validation."""
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    transformed_values: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


class FeatureValidator:
    """
    Validates feature values against registry definitions.
    
    Provides:
    - Schema validation
    - Type coercion
    - Range checking
    - Missing feature detection
    - Training/serving skew detection
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        strict_mode: bool = False,
        coerce_types: bool = True,
    ) -> None:
        """
        Initialize validator.
        
        Args:
            registry: Feature registry with definitions
            strict_mode: If True, treat warnings as errors
            coerce_types: If True, attempt to convert types
        """
        self.registry = registry
        self.strict_mode = strict_mode
        self.coerce_types = coerce_types
    
    def _coerce_value(
        self,
        value: Any,
        feature: FeatureDefinition,
    ) -> tuple[Any, ValidationError | None]:
        """Attempt to coerce a value to the expected type."""
        if value is None:
            return None, None
        
        try:
            from features.registry import FeatureType
            
            if feature.dtype == FeatureType.INTEGER:
                return int(float(value)), None
            elif feature.dtype == FeatureType.FLOAT:
                return float(value), None
            elif feature.dtype == FeatureType.STRING:
                return str(value), None
            elif feature.dtype == FeatureType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes"), None
                return bool(value), None
            
            return value, None
            
        except (ValueError, TypeError) as e:
            return value, ValidationError(
                feature_name=feature.name,
                error_type="type_coercion_failed",
                message=f"Cannot convert {type(value).__name__} to {feature.dtype.value}: {e}",
                value=value,
            )
    
    def validate_feature(
        self,
        feature: FeatureDefinition,
        value: Any,
    ) -> tuple[Any, list[ValidationError], list[ValidationError]]:
        """
        Validate a single feature value.
        
        Returns tuple of (transformed_value, errors, warnings).
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        
        if value is None:
            if feature.default_value is not None:
                return feature.default_value, errors, warnings
            else:
                errors.append(ValidationError(
                    feature_name=feature.name,
                    error_type="missing_required",
                    message=f"Required feature '{feature.name}' is missing",
                ))
                return None, errors, warnings
        
        if self.coerce_types:
            value, coerce_error = self._coerce_value(value, feature)
            if coerce_error:
                errors.append(coerce_error)
                return value, errors, warnings
        
        is_valid, error_msg = feature.validate(value)
        if not is_valid and error_msg:
            errors.append(ValidationError(
                feature_name=feature.name,
                error_type="validation_failed",
                message=error_msg,
                value=value,
            ))
            return value, errors, warnings
        
        transformed = feature.transform(value)
        
        return transformed, errors, warnings
    
    def validate(
        self,
        group_name: str,
        features: dict[str, Any],
        allow_extra: bool = True,
    ) -> ValidationResult:
        """
        Validate a feature dictionary against a group schema.
        
        Args:
            group_name: Name of the feature group
            features: Dictionary of feature name -> value
            allow_extra: Whether to allow features not in the schema
        
        Returns:
            ValidationResult with errors, warnings, and transformed values
        """
        group = self.registry.get_group(group_name)
        if not group:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    feature_name="_group",
                    error_type="unknown_group",
                    message=f"Unknown feature group: {group_name}",
                )],
            )
        
        all_errors: list[ValidationError] = []
        all_warnings: list[ValidationError] = []
        transformed: dict[str, Any] = {}
        
        for feature_def in group.features:
            value = features.get(feature_def.name)
            trans_value, errors, warnings = self.validate_feature(feature_def, value)
            
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            
            if trans_value is not None:
                transformed[feature_def.name] = trans_value
        
        if not allow_extra:
            expected_names = {f.name for f in group.features}
            extra_names = set(features.keys()) - expected_names
            
            for name in extra_names:
                warning = ValidationError(
                    feature_name=name,
                    error_type="unexpected_feature",
                    message=f"Feature '{name}' not in schema for group '{group_name}'",
                    value=features[name],
                )
                if self.strict_mode:
                    all_errors.append(warning)
                else:
                    all_warnings.append(warning)
        
        is_valid = len(all_errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(all_warnings) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            transformed_values=transformed,
        )
    
    def validate_batch(
        self,
        group_name: str,
        records: list[dict[str, Any]],
        allow_extra: bool = True,
    ) -> list[ValidationResult]:
        """
        Validate a batch of feature records.
        
        Args:
            group_name: Name of the feature group
            records: List of feature dictionaries
            allow_extra: Whether to allow features not in the schema
        
        Returns:
            List of ValidationResult, one per record
        """
        return [
            self.validate(group_name, record, allow_extra)
            for record in records
        ]
    
    def detect_training_serving_skew(
        self,
        group_name: str,
        training_stats: dict[str, dict[str, float]],
        serving_stats: dict[str, dict[str, float]],
        threshold: float = 0.1,
    ) -> list[ValidationError]:
        """
        Detect potential training/serving skew in feature distributions.
        
        Args:
            group_name: Name of the feature group
            training_stats: Dict of feature name -> {mean, std, min, max}
            serving_stats: Dict of feature name -> {mean, std, min, max}
            threshold: Relative difference threshold for warnings
        
        Returns:
            List of warnings for features with potential skew
        """
        warnings: list[ValidationError] = []
        
        group = self.registry.get_group(group_name)
        if not group:
            return warnings
        
        for feature in group.features:
            name = feature.name
            
            if name not in training_stats or name not in serving_stats:
                continue
            
            train = training_stats[name]
            serve = serving_stats[name]
            
            if train.get("mean", 0) != 0:
                mean_diff = abs(train["mean"] - serve.get("mean", 0)) / abs(train["mean"])
                if mean_diff > threshold:
                    warnings.append(ValidationError(
                        feature_name=name,
                        error_type="training_serving_skew",
                        message=f"Mean differs by {mean_diff:.1%}: training={train['mean']:.4f}, serving={serve.get('mean', 0):.4f}",
                    ))
            
            if train.get("std", 0) != 0:
                std_diff = abs(train["std"] - serve.get("std", 0)) / train["std"]
                if std_diff > threshold:
                    warnings.append(ValidationError(
                        feature_name=name,
                        error_type="training_serving_skew",
                        message=f"Std differs by {std_diff:.1%}: training={train['std']:.4f}, serving={serve.get('std', 0):.4f}",
                    ))
        
        return warnings

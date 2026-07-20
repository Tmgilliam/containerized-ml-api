"""Drift detection engine for ML models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.drift.metrics import (
    calculate_psi,
    calculate_ks_statistic,
    calculate_jensen_shannon_divergence,
)

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Drift severity levels for alerting."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature."""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    jsd: float
    severity: DriftSeverity
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "jsd": round(self.jsd, 4),
            "severity": self.severity.value,
        }


@dataclass
class DriftResult:
    """Complete drift analysis result."""
    timestamp: str
    model_version: str
    reference_size: int
    current_size: int
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: FeatureDriftResult | None = None
    overall_severity: DriftSeverity = DriftSeverity.NONE
    alert_triggered: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "feature_results": [f.to_dict() for f in self.feature_results],
            "prediction_drift": self.prediction_drift.to_dict() if self.prediction_drift else None,
            "overall_severity": self.overall_severity.value,
            "alert_triggered": self.alert_triggered,
        }


class DriftDetector:
    """
    Detects feature and prediction drift using statistical tests.
    
    Compares production data against reference (training) distributions
    to identify when retraining or investigation is needed.
    """
    
    PSI_THRESHOLDS = {
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MODERATE: 0.2,
        DriftSeverity.HIGH: 0.3,
        DriftSeverity.CRITICAL: 0.5,
    }
    
    KS_PVALUE_THRESHOLD = 0.01
    
    def __init__(
        self,
        reference_data: dict[str, np.ndarray],
        model_version: str,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize drift detector with reference distributions.
        
        Args:
            reference_data: Dict mapping feature names to numpy arrays
            model_version: Version identifier for the model being monitored
            feature_names: Optional subset of features to monitor
        """
        self.reference_data = reference_data
        self.model_version = model_version
        self.feature_names = feature_names or list(reference_data.keys())
        
        self._reference_predictions: np.ndarray | None = None
        
    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Set reference prediction distribution for output drift detection."""
        self._reference_predictions = predictions
        
    def _classify_severity(self, psi: float, ks_pvalue: float) -> DriftSeverity:
        """Determine drift severity based on PSI and KS test."""
        if psi >= self.PSI_THRESHOLDS[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif psi >= self.PSI_THRESHOLDS[DriftSeverity.HIGH]:
            return DriftSeverity.HIGH
        elif psi >= self.PSI_THRESHOLDS[DriftSeverity.MODERATE]:
            return DriftSeverity.MODERATE
        elif psi >= self.PSI_THRESHOLDS[DriftSeverity.LOW] or ks_pvalue < self.KS_PVALUE_THRESHOLD:
            return DriftSeverity.LOW
        return DriftSeverity.NONE
    
    def analyze_feature(
        self,
        feature_name: str,
        current_data: np.ndarray,
    ) -> FeatureDriftResult:
        """Analyze drift for a single feature."""
        if feature_name not in self.reference_data:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        reference = self.reference_data[feature_name]
        
        psi = calculate_psi(reference, current_data)
        ks_stat, ks_pvalue = calculate_ks_statistic(reference, current_data)
        jsd = calculate_jensen_shannon_divergence(reference, current_data)
        severity = self._classify_severity(psi, ks_pvalue)
        
        return FeatureDriftResult(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            jsd=jsd,
            severity=severity,
        )
    
    def analyze(
        self,
        current_data: dict[str, np.ndarray],
        current_predictions: np.ndarray | None = None,
        alert_threshold: DriftSeverity = DriftSeverity.MODERATE,
    ) -> DriftResult:
        """
        Run complete drift analysis on current data.
        
        Args:
            current_data: Dict mapping feature names to current numpy arrays
            current_predictions: Optional array of current predictions for output drift
            alert_threshold: Minimum severity to trigger alert
        
        Returns:
            DriftResult with analysis details
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        feature_results = []
        max_severity = DriftSeverity.NONE
        
        for feature_name in self.feature_names:
            if feature_name not in current_data:
                logger.warning("Feature %s missing from current data", feature_name)
                continue
                
            result = self.analyze_feature(feature_name, current_data[feature_name])
            feature_results.append(result)
            
            if list(DriftSeverity).index(result.severity) > list(DriftSeverity).index(max_severity):
                max_severity = result.severity
        
        prediction_drift = None
        if current_predictions is not None and self._reference_predictions is not None:
            psi = calculate_psi(self._reference_predictions, current_predictions)
            ks_stat, ks_pvalue = calculate_ks_statistic(self._reference_predictions, current_predictions)
            jsd = calculate_jensen_shannon_divergence(self._reference_predictions, current_predictions)
            severity = self._classify_severity(psi, ks_pvalue)
            
            prediction_drift = FeatureDriftResult(
                feature_name="prediction_score",
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                jsd=jsd,
                severity=severity,
            )
            
            if list(DriftSeverity).index(severity) > list(DriftSeverity).index(max_severity):
                max_severity = severity
        
        alert_triggered = list(DriftSeverity).index(max_severity) >= list(DriftSeverity).index(alert_threshold)
        
        reference_size = len(next(iter(self.reference_data.values())))
        current_size = len(next(iter(current_data.values()))) if current_data else 0
        
        result = DriftResult(
            timestamp=timestamp,
            model_version=self.model_version,
            reference_size=reference_size,
            current_size=current_size,
            feature_results=feature_results,
            prediction_drift=prediction_drift,
            overall_severity=max_severity,
            alert_triggered=alert_triggered,
        )
        
        logger.info(
            "Drift analysis complete model=%s severity=%s alert=%s",
            self.model_version,
            max_severity.value,
            alert_triggered,
        )
        
        return result
    
    def save_reference(self, path: Path) -> None:
        """Save reference distributions to disk."""
        data = {
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "reference_data": {k: v.tolist() for k, v in self.reference_data.items()},
        }
        if self._reference_predictions is not None:
            data["reference_predictions"] = self._reference_predictions.tolist()
        
        path.write_text(json.dumps(data, indent=2))
        logger.info("Reference data saved to %s", path)
    
    @classmethod
    def load_reference(cls, path: Path) -> "DriftDetector":
        """Load drift detector from saved reference."""
        data = json.loads(path.read_text())
        
        reference_data = {k: np.array(v) for k, v in data["reference_data"].items()}
        detector = cls(
            reference_data=reference_data,
            model_version=data["model_version"],
            feature_names=data.get("feature_names"),
        )
        
        if "reference_predictions" in data:
            detector.set_reference_predictions(np.array(data["reference_predictions"]))
        
        return detector

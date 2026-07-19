"""Automated retraining triggers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of retraining triggers."""
    DRIFT = "drift"
    ACCURACY = "accuracy"
    SCHEDULE = "schedule"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"


@dataclass
class TriggerConfig:
    """Configuration for retraining triggers."""
    drift_psi_threshold: float = 0.2
    drift_ks_pvalue_threshold: float = 0.01
    accuracy_threshold: float = 0.8
    accuracy_window_days: int = 7
    schedule_days: int = 30
    min_new_samples: int = 1000
    min_samples_for_accuracy: int = 100
    cooldown_hours: float = 24.0


@dataclass
class TriggerResult:
    """Result of a trigger evaluation."""
    should_retrain: bool
    trigger_type: TriggerType
    reason: str
    details: dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "trigger_type": self.trigger_type.value,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class RetrainTrigger:
    """
    Determines when model retraining should be triggered.
    
    Supports multiple trigger conditions:
    - Drift detection (PSI/KS thresholds)
    - Accuracy degradation
    - Scheduled retraining
    - New data volume
    """
    
    def __init__(
        self,
        config: TriggerConfig | None = None,
        on_trigger: Callable[[TriggerResult], None] | None = None,
    ) -> None:
        """
        Initialize retrain trigger.
        
        Args:
            config: Trigger configuration
            on_trigger: Callback when retraining is triggered
        """
        self.config = config or TriggerConfig()
        self.on_trigger = on_trigger
        
        self._last_retrain: datetime | None = None
        self._last_check: datetime | None = None
    
    def check_drift_trigger(
        self,
        drift_result: dict[str, Any],
    ) -> TriggerResult:
        """
        Check if drift levels warrant retraining.
        
        Args:
            drift_result: Result from DriftDetector.analyze()
        
        Returns:
            TriggerResult indicating if retraining needed
        """
        max_psi = 0.0
        min_ks_pvalue = 1.0
        drifted_features = []
        
        for feature in drift_result.get("feature_results", []):
            psi = feature.get("psi", 0)
            ks_pvalue = feature.get("ks_pvalue", 1)
            
            if psi > max_psi:
                max_psi = psi
            if ks_pvalue < min_ks_pvalue:
                min_ks_pvalue = ks_pvalue
            
            if psi >= self.config.drift_psi_threshold:
                drifted_features.append(feature["feature_name"])
        
        should_retrain = (
            max_psi >= self.config.drift_psi_threshold or
            min_ks_pvalue < self.config.drift_ks_pvalue_threshold
        )
        
        result = TriggerResult(
            should_retrain=should_retrain,
            trigger_type=TriggerType.DRIFT,
            reason=f"Drift detected: max_psi={max_psi:.3f}, drifted_features={drifted_features}" if should_retrain else "Drift within acceptable limits",
            details={
                "max_psi": max_psi,
                "min_ks_pvalue": min_ks_pvalue,
                "drifted_features": drifted_features,
                "threshold_psi": self.config.drift_psi_threshold,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        if should_retrain and self.on_trigger:
            self.on_trigger(result)
        
        return result
    
    def check_accuracy_trigger(
        self,
        accuracy_report: dict[str, Any],
    ) -> TriggerResult:
        """
        Check if accuracy degradation warrants retraining.
        
        Args:
            accuracy_report: Result from LabelMatcher.accuracy_report()
        
        Returns:
            TriggerResult indicating if retraining needed
        """
        total_samples = accuracy_report.get("total_matched", 0)
        accuracy = accuracy_report.get("accuracy", 1.0)
        
        if total_samples < self.config.min_samples_for_accuracy:
            return TriggerResult(
                should_retrain=False,
                trigger_type=TriggerType.ACCURACY,
                reason=f"Insufficient samples for accuracy check ({total_samples} < {self.config.min_samples_for_accuracy})",
                details={"total_samples": total_samples, "accuracy": accuracy},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        should_retrain = accuracy < self.config.accuracy_threshold
        
        result = TriggerResult(
            should_retrain=should_retrain,
            trigger_type=TriggerType.ACCURACY,
            reason=f"Accuracy degraded to {accuracy:.1%} (threshold: {self.config.accuracy_threshold:.1%})" if should_retrain else f"Accuracy acceptable at {accuracy:.1%}",
            details={
                "accuracy": accuracy,
                "threshold": self.config.accuracy_threshold,
                "total_samples": total_samples,
                "precision": accuracy_report.get("precision"),
                "recall": accuracy_report.get("recall"),
                "f1_score": accuracy_report.get("f1_score"),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        if should_retrain and self.on_trigger:
            self.on_trigger(result)
        
        return result
    
    def check_schedule_trigger(
        self,
        last_training_date: datetime | None = None,
    ) -> TriggerResult:
        """
        Check if scheduled retraining is due.
        
        Args:
            last_training_date: When model was last trained
        
        Returns:
            TriggerResult indicating if retraining needed
        """
        if last_training_date is None:
            last_training_date = self._last_retrain
        
        if last_training_date is None:
            return TriggerResult(
                should_retrain=True,
                trigger_type=TriggerType.SCHEDULE,
                reason="No previous training date recorded",
                details={"schedule_days": self.config.schedule_days},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        days_since = (datetime.now(timezone.utc) - last_training_date).days
        should_retrain = days_since >= self.config.schedule_days
        
        result = TriggerResult(
            should_retrain=should_retrain,
            trigger_type=TriggerType.SCHEDULE,
            reason=f"Scheduled retraining due ({days_since} days since last training)" if should_retrain else f"Next scheduled retraining in {self.config.schedule_days - days_since} days",
            details={
                "days_since_training": days_since,
                "schedule_days": self.config.schedule_days,
                "last_training": last_training_date.isoformat(),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        if should_retrain and self.on_trigger:
            self.on_trigger(result)
        
        return result
    
    def check_data_volume_trigger(
        self,
        new_sample_count: int,
    ) -> TriggerResult:
        """
        Check if sufficient new data is available for retraining.
        
        Args:
            new_sample_count: Number of new labeled samples
        
        Returns:
            TriggerResult indicating if retraining needed
        """
        should_retrain = new_sample_count >= self.config.min_new_samples
        
        result = TriggerResult(
            should_retrain=should_retrain,
            trigger_type=TriggerType.DATA_VOLUME,
            reason=f"Sufficient new data available ({new_sample_count} samples)" if should_retrain else f"Insufficient new data ({new_sample_count} < {self.config.min_new_samples})",
            details={
                "new_samples": new_sample_count,
                "threshold": self.config.min_new_samples,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        if should_retrain and self.on_trigger:
            self.on_trigger(result)
        
        return result
    
    def evaluate_all(
        self,
        drift_result: dict[str, Any] | None = None,
        accuracy_report: dict[str, Any] | None = None,
        last_training_date: datetime | None = None,
        new_sample_count: int = 0,
    ) -> list[TriggerResult]:
        """
        Evaluate all trigger conditions.
        
        Returns:
            List of TriggerResult for each condition checked
        """
        results = []
        
        if self._last_retrain:
            cooldown_end = self._last_retrain + timedelta(hours=self.config.cooldown_hours)
            if datetime.now(timezone.utc) < cooldown_end:
                logger.info("In cooldown period, skipping trigger evaluation")
                return []
        
        if drift_result:
            results.append(self.check_drift_trigger(drift_result))
        
        if accuracy_report:
            results.append(self.check_accuracy_trigger(accuracy_report))
        
        results.append(self.check_schedule_trigger(last_training_date))
        results.append(self.check_data_volume_trigger(new_sample_count))
        
        self._last_check = datetime.now(timezone.utc)
        
        return results
    
    def should_retrain(
        self,
        drift_result: dict[str, Any] | None = None,
        accuracy_report: dict[str, Any] | None = None,
        last_training_date: datetime | None = None,
        new_sample_count: int = 0,
    ) -> tuple[bool, list[TriggerResult]]:
        """
        Check if any trigger condition indicates retraining needed.
        
        Returns:
            Tuple of (should_retrain, list of trigger results)
        """
        results = self.evaluate_all(
            drift_result=drift_result,
            accuracy_report=accuracy_report,
            last_training_date=last_training_date,
            new_sample_count=new_sample_count,
        )
        
        should_retrain = any(r.should_retrain for r in results)
        
        return should_retrain, results
    
    def record_retrain(self) -> None:
        """Record that retraining occurred."""
        self._last_retrain = datetime.now(timezone.utc)
        logger.info("Recorded retraining at %s", self._last_retrain.isoformat())

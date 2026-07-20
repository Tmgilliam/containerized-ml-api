"""Match predictions with actual outcomes for training data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MatchedRecord:
    """A prediction matched with its actual outcome."""
    entity_id: str
    prediction_timestamp: str
    outcome_timestamp: str
    features: dict[str, Any]
    predicted_value: int | float
    predicted_probability: float
    actual_outcome: bool
    is_correct: bool
    model_version: str
    latency_to_outcome_hours: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "prediction_timestamp": self.prediction_timestamp,
            "outcome_timestamp": self.outcome_timestamp,
            "features": self.features,
            "predicted_value": self.predicted_value,
            "predicted_probability": self.predicted_probability,
            "actual_outcome": self.actual_outcome,
            "is_correct": self.is_correct,
            "model_version": self.model_version,
            "latency_to_outcome_hours": round(self.latency_to_outcome_hours, 2),
            "metadata": self.metadata,
        }
    
    def to_training_example(self) -> dict[str, Any]:
        """Convert to format suitable for model training."""
        return {
            **self.features,
            "label": 1 if self.actual_outcome else 0,
        }


class LabelMatcher:
    """
    Matches predictions with actual outcomes.
    
    Joins prediction logs with outcome events to create
    labeled training data for retraining.
    """
    
    def __init__(
        self,
        predictions_path: Path | None = None,
        outcomes_path: Path | None = None,
        match_window_hours: float = 168.0,
    ) -> None:
        """
        Initialize label matcher.
        
        Args:
            predictions_path: Path to prediction logs
            outcomes_path: Path to outcome records
            match_window_hours: Maximum time between prediction and outcome
        """
        self.predictions_path = predictions_path
        self.outcomes_path = outcomes_path
        self.match_window_hours = match_window_hours
        
        self._predictions: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, dict[str, Any]] = {}
    
    def load_predictions(
        self,
        records: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Load prediction records.
        
        Args:
            records: List of prediction dicts, or load from path
        
        Returns:
            Number of predictions loaded
        """
        if records:
            for record in records:
                entity_id = record.get("entity_id") or record.get("_id")
                if entity_id:
                    self._predictions[str(entity_id)] = record
            return len(records)
        
        if self.predictions_path and self.predictions_path.exists():
            count = 0
            for filepath in self.predictions_path.glob("*.jsonl"):
                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            entity_id = record.get("entity_id") or record.get("_id")
                            if entity_id:
                                self._predictions[str(entity_id)] = record
                                count += 1
            
            logger.info("Loaded %d predictions", count)
            return count
        
        return 0
    
    def load_outcomes(
        self,
        records: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Load outcome records.
        
        Args:
            records: List of outcome dicts, or load from path
        
        Returns:
            Number of outcomes loaded
        """
        if records:
            for record in records:
                entity_id = record.get("entity_id")
                if entity_id:
                    self._outcomes[str(entity_id)] = record
            return len(records)
        
        if self.outcomes_path and self.outcomes_path.exists():
            count = 0
            for filepath in self.outcomes_path.glob("*.jsonl"):
                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            entity_id = record.get("entity_id")
                            if entity_id:
                                self._outcomes[str(entity_id)] = record
                                count += 1
            
            logger.info("Loaded %d outcomes", count)
            return count
        
        return 0
    
    def match(self) -> list[MatchedRecord]:
        """
        Match predictions with outcomes.
        
        Returns:
            List of MatchedRecord with prediction/outcome pairs
        """
        matched = []
        
        for entity_id, prediction in self._predictions.items():
            if entity_id not in self._outcomes:
                continue
            
            outcome = self._outcomes[entity_id]
            
            pred_ts = prediction.get("timestamp") or prediction.get("_predicted_at")
            outcome_ts = outcome.get("outcome_timestamp") or outcome.get("timestamp")
            
            if not pred_ts or not outcome_ts:
                continue
            
            if isinstance(pred_ts, str):
                pred_dt = datetime.fromisoformat(pred_ts.replace("Z", "+00:00"))
            else:
                pred_dt = pred_ts
            
            if isinstance(outcome_ts, str):
                outcome_dt = datetime.fromisoformat(outcome_ts.replace("Z", "+00:00"))
            else:
                outcome_dt = outcome_ts
            
            latency_hours = (outcome_dt - pred_dt).total_seconds() / 3600
            
            if latency_hours < 0 or latency_hours > self.match_window_hours:
                continue
            
            predicted_value = prediction.get("delay_risk") or prediction.get("prediction", 0)
            predicted_prob = prediction.get("risk_score") or prediction.get("probability", 0.5)
            actual_outcome = outcome.get("outcome", False)
            
            is_correct = (predicted_value == 1) == actual_outcome
            
            features = {
                k: v for k, v in prediction.items()
                if k not in ("timestamp", "_predicted_at", "delay_risk", "risk_score",
                           "prediction", "probability", "confidence", "model_version",
                           "entity_id", "_id", "latency_ms")
            }
            
            record = MatchedRecord(
                entity_id=entity_id,
                prediction_timestamp=pred_ts if isinstance(pred_ts, str) else pred_ts.isoformat(),
                outcome_timestamp=outcome_ts if isinstance(outcome_ts, str) else outcome_ts.isoformat(),
                features=features,
                predicted_value=predicted_value,
                predicted_probability=predicted_prob,
                actual_outcome=actual_outcome,
                is_correct=is_correct,
                model_version=prediction.get("model_version", "unknown"),
                latency_to_outcome_hours=latency_hours,
            )
            
            matched.append(record)
        
        logger.info(
            "Matched %d prediction-outcome pairs (%.1f%% accuracy)",
            len(matched),
            sum(1 for m in matched if m.is_correct) / len(matched) * 100 if matched else 0,
        )
        
        return matched
    
    def generate_training_data(
        self,
        min_samples: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Generate training dataset from matched records.
        
        Args:
            min_samples: Minimum samples required
        
        Returns:
            List of training examples with features and labels
        """
        matched = self.match()
        
        if len(matched) < min_samples:
            logger.warning(
                "Only %d matched samples, minimum is %d",
                len(matched),
                min_samples,
            )
        
        return [m.to_training_example() for m in matched]
    
    def accuracy_report(self) -> dict[str, Any]:
        """Generate accuracy report from matched data."""
        matched = self.match()
        
        if not matched:
            return {"error": "No matched records"}
        
        total = len(matched)
        correct = sum(1 for m in matched if m.is_correct)
        
        true_positives = sum(1 for m in matched if m.predicted_value == 1 and m.actual_outcome)
        false_positives = sum(1 for m in matched if m.predicted_value == 1 and not m.actual_outcome)
        true_negatives = sum(1 for m in matched if m.predicted_value == 0 and not m.actual_outcome)
        false_negatives = sum(1 for m in matched if m.predicted_value == 0 and m.actual_outcome)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        by_version: dict[str, dict[str, int]] = {}
        for m in matched:
            v = m.model_version
            if v not in by_version:
                by_version[v] = {"total": 0, "correct": 0}
            by_version[v]["total"] += 1
            if m.is_correct:
                by_version[v]["correct"] += 1
        
        return {
            "total_matched": total,
            "accuracy": round(correct / total, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
            },
            "by_model_version": {
                v: {
                    "total": d["total"],
                    "accuracy": round(d["correct"] / d["total"], 4),
                }
                for v, d in by_version.items()
            },
            "avg_latency_hours": round(
                sum(m.latency_to_outcome_hours for m in matched) / total, 2
            ),
        }
    
    def clear(self) -> None:
        """Clear loaded data."""
        self._predictions.clear()
        self._outcomes.clear()

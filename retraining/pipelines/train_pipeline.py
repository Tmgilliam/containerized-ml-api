"""Training pipeline for model retraining."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    min_training_samples: int = 500
    artifact_path: Path = field(default_factory=lambda: Path("./model"))
    model_type: str = "gradient_boosting"
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Results from a training run."""
    model_version: str
    training_timestamp: str
    metrics: dict[str, float]
    cv_scores: list[float]
    feature_importance: dict[str, float]
    training_samples: int
    test_samples: int
    artifact_path: str
    hyperparameters: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "cv_scores": [round(s, 4) for s in self.cv_scores],
            "cv_mean": round(np.mean(self.cv_scores), 4),
            "cv_std": round(np.std(self.cv_scores), 4),
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "artifact_path": self.artifact_path,
            "hyperparameters": self.hyperparameters,
        }


class TrainingPipeline:
    """
    Training pipeline for model retraining.
    
    Handles:
    - Data validation
    - Train/test splitting
    - Model training with cross-validation
    - Metric evaluation
    - Artifact saving
    """
    
    DEFAULT_HYPERPARAMETERS = {
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
    }
    
    def __init__(
        self,
        config: PipelineConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration
            feature_names: Expected feature names
        """
        self.config = config or PipelineConfig()
        self.feature_names = feature_names or [
            "order_qty",
            "lead_time_days",
            "vendor_reliability_score",
            "days_until_due",
            "historical_delay_rate",
            "inventory_buffer_days",
        ]
        
        self.config.artifact_path.mkdir(parents=True, exist_ok=True)
    
    def _create_model(self):
        """Create a fresh model instance."""
        hyperparams = {
            **self.DEFAULT_HYPERPARAMETERS.get(self.config.model_type, {}),
            **self.config.hyperparameters,
        }
        hyperparams["random_state"] = self.config.random_state
        
        if self.config.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**hyperparams)
        elif self.config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**hyperparams)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**hyperparams)
    
    def validate_data(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
    ) -> tuple[bool, list[str]]:
        """
        Validate training data.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if len(data) < self.config.min_training_samples:
            issues.append(
                f"Insufficient samples: {len(data)} < {self.config.min_training_samples}"
            )
        
        missing_features = [f for f in self.feature_names if f not in data.columns]
        if missing_features:
            issues.append(f"Missing features: {missing_features}")
        
        if label_column not in data.columns:
            issues.append(f"Missing label column: {label_column}")
        
        if label_column in data.columns:
            label_values = data[label_column].unique()
            if len(label_values) < 2:
                issues.append(f"Only one class in labels: {label_values}")
            
            class_counts = data[label_column].value_counts()
            min_class_pct = class_counts.min() / len(data)
            if min_class_pct < 0.1:
                issues.append(
                    f"Severe class imbalance: minority class is {min_class_pct:.1%}"
                )
        
        for feature in self.feature_names:
            if feature in data.columns:
                null_pct = data[feature].isnull().sum() / len(data)
                if null_pct > 0.1:
                    issues.append(f"High null rate in {feature}: {null_pct:.1%}")
        
        return len(issues) == 0, issues
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        label_column: str = "label",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = data[self.feature_names].fillna(0).values
        y = data[label_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        label_column: str = "label",
        version_suffix: str | None = None,
    ) -> TrainingResult:
        """
        Train a new model.
        
        Args:
            data: Training data (DataFrame or list of dicts)
            label_column: Name of label column
            version_suffix: Optional suffix for model version
        
        Returns:
            TrainingResult with metrics and artifact info
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        is_valid, issues = self.validate_data(data, label_column)
        if not is_valid:
            raise ValueError(f"Data validation failed: {issues}")
        
        X_train, X_test, y_train, y_test = self.prepare_data(data, label_column)
        
        logger.info(
            "Training on %d samples, testing on %d samples",
            len(X_train),
            len(X_test),
        )
        
        model = self._create_model()
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring="roc_auc",
        )
        
        logger.info("CV AUC: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_prob),
        }
        
        logger.info("Test metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
        
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(self.feature_names, model.feature_importances_))
        else:
            importance = {f: 0.0 for f in self.feature_names}
        
        timestamp = datetime.now(timezone.utc)
        version = timestamp.strftime("%Y%m%d_%H%M%S")
        if version_suffix:
            version = f"{version}_{version_suffix}"
        
        artifact_filename = f"model_{version}.pkl"
        artifact_path = self.config.artifact_path / artifact_filename
        
        artifact = {
            "model": model,
            "metadata": {
                "version": version,
                "trained_at": timestamp.isoformat(),
                "features": self.feature_names,
                "metrics": metrics,
                "cv_scores": cv_scores.tolist(),
                "model_type": self.config.model_type,
                "hyperparameters": self.config.hyperparameters,
            },
        }
        
        joblib.dump(artifact, artifact_path)
        logger.info("Model saved to %s", artifact_path)
        
        latest_path = self.config.artifact_path / "model.pkl"
        joblib.dump(artifact, latest_path)
        
        return TrainingResult(
            model_version=version,
            training_timestamp=timestamp.isoformat(),
            metrics=metrics,
            cv_scores=cv_scores.tolist(),
            feature_importance=importance,
            training_samples=len(X_train),
            test_samples=len(X_test),
            artifact_path=str(artifact_path),
            hyperparameters=self.config.hyperparameters,
        )
    
    def evaluate_model(
        self,
        model_path: Path,
        test_data: pd.DataFrame,
        label_column: str = "label",
    ) -> dict[str, Any]:
        """
        Evaluate an existing model on new data.
        
        Args:
            model_path: Path to model artifact
            test_data: Test dataset
            label_column: Name of label column
        
        Returns:
            Dict with evaluation metrics
        """
        artifact = joblib.load(model_path)
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        
        X = test_data[self.feature_names].fillna(0).values
        y = test_data[label_column].values
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_prob),
            "samples": len(y),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }
    
    def compare_models(
        self,
        model_paths: list[Path],
        test_data: pd.DataFrame,
        label_column: str = "label",
    ) -> dict[str, dict[str, Any]]:
        """
        Compare multiple model versions.
        
        Returns:
            Dict mapping model path to evaluation metrics
        """
        results = {}
        
        for path in model_paths:
            try:
                metrics = self.evaluate_model(path, test_data, label_column)
                results[str(path)] = metrics
            except Exception as e:
                results[str(path)] = {"error": str(e)}
        
        return results
    
    def list_models(self) -> list[dict[str, Any]]:
        """List all saved model artifacts."""
        models = []
        
        for path in self.config.artifact_path.glob("model_*.pkl"):
            try:
                artifact = joblib.load(path)
                metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
                models.append({
                    "path": str(path),
                    "version": metadata.get("version", "unknown"),
                    "trained_at": metadata.get("trained_at"),
                    "auc": metadata.get("metrics", {}).get("auc_roc"),
                })
            except Exception as e:
                models.append({
                    "path": str(path),
                    "error": str(e),
                })
        
        return sorted(models, key=lambda m: m.get("trained_at", ""), reverse=True)

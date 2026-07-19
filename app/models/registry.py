"""Model registry for multi-model serving."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import joblib

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status."""
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    version: str
    model_type: str
    description: str
    artifact_path: Path
    feature_names: list[str]
    status: ModelStatus = ModelStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "description": self.description,
            "artifact_path": str(self.artifact_path),
            "feature_names": self.feature_names,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        return cls(
            name=data["name"],
            version=data["version"],
            model_type=data["model_type"],
            description=data["description"],
            artifact_path=Path(data["artifact_path"]),
            feature_names=data["feature_names"],
            status=ModelStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
            metrics=data.get("metrics", {}),
            tags=data.get("tags", []),
        )


class LoadedModel:
    """A loaded model ready for inference."""
    
    def __init__(self, info: ModelInfo, classifier: Any) -> None:
        self.info = info
        self._classifier = classifier
    
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference on the model."""
        import pandas as pd
        
        feature_vector = pd.DataFrame(
            [{name: float(features[name]) for name in self.info.feature_names}]
        )
        
        probabilities = self._classifier.predict_proba(feature_vector)[0]
        prediction = int(self._classifier.predict(feature_vector)[0])
        
        return {
            "prediction": prediction,
            "probability": float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]),
            "confidence": float(max(probabilities)),
            "model_name": self.info.name,
            "model_version": self.info.version,
        }


class ModelRegistry:
    """
    Central registry for multiple ML models.
    
    Features:
    - Model registration and versioning
    - Lazy loading of model artifacts
    - Health checking
    - Model lifecycle management
    """
    
    def __init__(self, registry_path: Path | None = None) -> None:
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to persist registry state
        """
        self.registry_path = registry_path
        self._models: dict[str, dict[str, ModelInfo]] = {}
        self._loaded: dict[str, LoadedModel] = {}
        self._default_versions: dict[str, str] = {}
        
        if registry_path and registry_path.exists():
            self._load_registry()
    
    def _model_key(self, name: str, version: str) -> str:
        """Generate key for loaded model cache."""
        return f"{name}:{version}"
    
    def register(
        self,
        name: str,
        version: str,
        model_type: str,
        description: str,
        artifact_path: Path | str,
        feature_names: list[str],
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
        set_default: bool = False,
    ) -> ModelInfo:
        """
        Register a new model version.
        
        Args:
            name: Model name
            version: Version string
            model_type: Type of model (e.g., 'delay_risk', 'demand_forecast')
            description: Human-readable description
            artifact_path: Path to model artifact (.pkl, .joblib)
            feature_names: List of required feature names
            metadata: Optional metadata dict
            metrics: Optional metrics dict (e.g., {'auc': 0.87})
            tags: Optional tags for filtering
            set_default: Whether to set as default version
        
        Returns:
            ModelInfo for the registered model
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            logger.warning("Model artifact not found: %s", artifact_path)
        
        info = ModelInfo(
            name=name,
            version=version,
            model_type=model_type,
            description=description,
            artifact_path=artifact_path,
            feature_names=feature_names,
            metadata=metadata or {},
            metrics=metrics or {},
            tags=tags or [],
        )
        
        if name not in self._models:
            self._models[name] = {}
        
        self._models[name][version] = info
        
        if set_default or name not in self._default_versions:
            self._default_versions[name] = version
        
        logger.info(
            "Registered model %s version %s (default=%s)",
            name,
            version,
            set_default or name not in self._default_versions,
        )
        
        if self.registry_path:
            self._save_registry()
        
        return info
    
    def load(self, name: str, version: str | None = None) -> LoadedModel:
        """
        Load a model for inference.
        
        Args:
            name: Model name
            version: Version string (uses default if not specified)
        
        Returns:
            LoadedModel ready for inference
        """
        version = version or self._default_versions.get(name)
        if not version:
            raise ValueError(f"No default version for model: {name}")
        
        key = self._model_key(name, version)
        
        if key in self._loaded:
            return self._loaded[key]
        
        if name not in self._models or version not in self._models[name]:
            raise ValueError(f"Model not found: {name}:{version}")
        
        info = self._models[name][version]
        info.status = ModelStatus.LOADING
        
        try:
            artifact = joblib.load(info.artifact_path)
            
            if isinstance(artifact, dict):
                classifier = artifact["model"]
            else:
                classifier = artifact
            
            loaded = LoadedModel(info, classifier)
            self._loaded[key] = loaded
            info.status = ModelStatus.READY
            
            logger.info("Loaded model %s version %s", name, version)
            return loaded
            
        except Exception as e:
            info.status = ModelStatus.FAILED
            logger.error("Failed to load model %s:%s - %s", name, version, e)
            raise
    
    def get_info(self, name: str, version: str | None = None) -> ModelInfo | None:
        """Get model info without loading."""
        version = version or self._default_versions.get(name)
        if not version:
            return None
        
        return self._models.get(name, {}).get(version)
    
    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        results = []
        
        for name, versions in self._models.items():
            for version, info in versions.items():
                is_default = self._default_versions.get(name) == version
                results.append({
                    **info.to_dict(),
                    "is_default": is_default,
                    "is_loaded": self._model_key(name, version) in self._loaded,
                })
        
        return results
    
    def list_versions(self, name: str) -> list[str]:
        """List all versions for a model."""
        return list(self._models.get(name, {}).keys())
    
    def set_default(self, name: str, version: str) -> None:
        """Set the default version for a model."""
        if name not in self._models or version not in self._models[name]:
            raise ValueError(f"Model not found: {name}:{version}")
        
        self._default_versions[name] = version
        logger.info("Set default version for %s to %s", name, version)
        
        if self.registry_path:
            self._save_registry()
    
    def deprecate(self, name: str, version: str) -> None:
        """Mark a model version as deprecated."""
        if name in self._models and version in self._models[name]:
            self._models[name][version].status = ModelStatus.DEPRECATED
            logger.info("Deprecated model %s version %s", name, version)
            
            if self.registry_path:
                self._save_registry()
    
    def unload(self, name: str, version: str | None = None) -> bool:
        """Unload a model from memory."""
        version = version or self._default_versions.get(name)
        if not version:
            return False
        
        key = self._model_key(name, version)
        if key in self._loaded:
            del self._loaded[key]
            logger.info("Unloaded model %s version %s", name, version)
            return True
        
        return False
    
    def health_check(self) -> dict[str, Any]:
        """Check health of all loaded models."""
        results = {
            "total_registered": sum(len(v) for v in self._models.values()),
            "total_loaded": len(self._loaded),
            "models": {},
        }
        
        for name, versions in self._models.items():
            results["models"][name] = {
                "versions": len(versions),
                "default": self._default_versions.get(name),
                "status": {
                    v: info.status.value
                    for v, info in versions.items()
                },
            }
        
        return results
    
    def _save_registry(self) -> None:
        """Save registry state to disk."""
        if not self.registry_path:
            return
        
        data = {
            "models": {
                name: {v: info.to_dict() for v, info in versions.items()}
                for name, versions in self._models.items()
            },
            "default_versions": self._default_versions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(data, indent=2))
    
    def _load_registry(self) -> None:
        """Load registry state from disk."""
        if not self.registry_path or not self.registry_path.exists():
            return
        
        data = json.loads(self.registry_path.read_text())
        
        for name, versions in data.get("models", {}).items():
            self._models[name] = {}
            for version, info_data in versions.items():
                self._models[name][version] = ModelInfo.from_dict(info_data)
        
        self._default_versions = data.get("default_versions", {})
        
        logger.info(
            "Loaded registry with %d models",
            sum(len(v) for v in self._models.values()),
        )

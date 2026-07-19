"""Feature registry for managing feature definitions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Supported feature data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    EMBEDDING = "embedding"


class FeatureSource(str, Enum):
    """Feature data source types."""
    BATCH = "batch"
    STREAMING = "streaming"
    REQUEST = "request"
    DERIVED = "derived"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    dtype: FeatureType
    description: str
    source: FeatureSource = FeatureSource.BATCH
    default_value: Any = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None
    transform_fn: Callable[[Any], Any] | None = None
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a value against this feature definition."""
        if value is None:
            if self.default_value is not None:
                return True, None
            return False, f"Feature '{self.name}' is required"
        
        try:
            if self.dtype == FeatureType.INTEGER:
                value = int(value)
            elif self.dtype == FeatureType.FLOAT:
                value = float(value)
            elif self.dtype == FeatureType.STRING:
                value = str(value)
            elif self.dtype == FeatureType.BOOLEAN:
                value = bool(value)
        except (ValueError, TypeError):
            return False, f"Feature '{self.name}' must be {self.dtype.value}"
        
        if self.dtype in (FeatureType.INTEGER, FeatureType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return False, f"Feature '{self.name}' below minimum {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Feature '{self.name}' above maximum {self.max_value}"
        
        if self.allowed_values is not None and value not in self.allowed_values:
            return False, f"Feature '{self.name}' must be one of {self.allowed_values}"
        
        return True, None
    
    def transform(self, value: Any) -> Any:
        """Apply transformation to feature value."""
        if value is None and self.default_value is not None:
            value = self.default_value
        
        if self.transform_fn is not None:
            value = self.transform_fn(value)
        
        return value
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "description": self.description,
            "source": self.source.value,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
            "version": self.version,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureDefinition":
        return cls(
            name=data["name"],
            dtype=FeatureType(data["dtype"]),
            description=data["description"],
            source=FeatureSource(data.get("source", "batch")),
            default_value=data.get("default_value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FeatureGroup:
    """A group of related features."""
    name: str
    description: str
    features: list[FeatureDefinition] = field(default_factory=list)
    entity_key: str = "entity_id"
    timestamp_key: str = "event_timestamp"
    ttl_seconds: int | None = None
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    
    def add_feature(self, feature: FeatureDefinition) -> None:
        """Add a feature to the group."""
        if any(f.name == feature.name for f in self.features):
            raise ValueError(f"Feature '{feature.name}' already exists in group")
        self.features.append(feature)
    
    def get_feature(self, name: str) -> FeatureDefinition | None:
        """Get a feature by name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None
    
    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "features": [f.to_dict() for f in self.features],
            "entity_key": self.entity_key,
            "timestamp_key": self.timestamp_key,
            "ttl_seconds": self.ttl_seconds,
            "version": self.version,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureGroup":
        group = cls(
            name=data["name"],
            description=data["description"],
            entity_key=data.get("entity_key", "entity_id"),
            timestamp_key=data.get("timestamp_key", "event_timestamp"),
            ttl_seconds=data.get("ttl_seconds"),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
        )
        for f_data in data.get("features", []):
            group.add_feature(FeatureDefinition.from_dict(f_data))
        return group


class FeatureRegistry:
    """
    Central registry for feature definitions.
    
    Manages feature groups and their definitions, providing:
    - Feature discovery and documentation
    - Schema validation
    - Version tracking
    - Serialization/deserialization
    """
    
    def __init__(self) -> None:
        self._groups: dict[str, FeatureGroup] = {}
        self._created_at = datetime.now(timezone.utc).isoformat()
    
    def register_group(self, group: FeatureGroup) -> None:
        """Register a feature group."""
        if group.name in self._groups:
            logger.warning("Overwriting existing feature group: %s", group.name)
        self._groups[group.name] = group
        logger.info(
            "Registered feature group '%s' with %d features",
            group.name,
            len(group.features),
        )
    
    def get_group(self, name: str) -> FeatureGroup | None:
        """Get a feature group by name."""
        return self._groups.get(name)
    
    def get_feature(self, group_name: str, feature_name: str) -> FeatureDefinition | None:
        """Get a specific feature from a group."""
        group = self.get_group(group_name)
        if group:
            return group.get_feature(feature_name)
        return None
    
    def list_groups(self) -> list[str]:
        """List all registered feature groups."""
        return list(self._groups.keys())
    
    def list_features(self, group_name: str | None = None) -> list[dict[str, str]]:
        """List features, optionally filtered by group."""
        results = []
        groups = [self._groups[group_name]] if group_name else self._groups.values()
        
        for group in groups:
            for feature in group.features:
                results.append({
                    "group": group.name,
                    "name": feature.name,
                    "dtype": feature.dtype.value,
                    "description": feature.description,
                })
        
        return results
    
    def search_features(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        dtype: FeatureType | None = None,
    ) -> list[dict[str, Any]]:
        """Search features by name, tags, or type."""
        results = []
        
        for group in self._groups.values():
            for feature in group.features:
                matches = True
                
                if query and query.lower() not in feature.name.lower():
                    if query.lower() not in feature.description.lower():
                        matches = False
                
                if tags and not any(t in feature.tags for t in tags):
                    matches = False
                
                if dtype and feature.dtype != dtype:
                    matches = False
                
                if matches:
                    results.append({
                        "group": group.name,
                        **feature.to_dict(),
                    })
        
        return results
    
    def save(self, path: Path) -> None:
        """Save registry to a JSON file."""
        data = {
            "created_at": self._created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "groups": {name: group.to_dict() for name, group in self._groups.items()},
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved feature registry to %s", path)
    
    @classmethod
    def load(cls, path: Path) -> "FeatureRegistry":
        """Load registry from a JSON file."""
        data = json.loads(path.read_text())
        
        registry = cls()
        registry._created_at = data.get("created_at", registry._created_at)
        
        for group_data in data.get("groups", {}).values():
            group = FeatureGroup.from_dict(group_data)
            registry.register_group(group)
        
        logger.info("Loaded feature registry from %s", path)
        return registry
    
    def to_schema(self, group_name: str) -> dict[str, Any]:
        """Generate JSON schema for a feature group."""
        group = self.get_group(group_name)
        if not group:
            raise ValueError(f"Unknown feature group: {group_name}")
        
        properties = {}
        required = []
        
        for feature in group.features:
            prop: dict[str, Any] = {
                "description": feature.description,
            }
            
            if feature.dtype == FeatureType.INTEGER:
                prop["type"] = "integer"
            elif feature.dtype == FeatureType.FLOAT:
                prop["type"] = "number"
            elif feature.dtype == FeatureType.STRING:
                prop["type"] = "string"
            elif feature.dtype == FeatureType.BOOLEAN:
                prop["type"] = "boolean"
            elif feature.dtype == FeatureType.ARRAY:
                prop["type"] = "array"
            else:
                prop["type"] = "string"
            
            if feature.min_value is not None:
                prop["minimum"] = feature.min_value
            if feature.max_value is not None:
                prop["maximum"] = feature.max_value
            if feature.allowed_values is not None:
                prop["enum"] = feature.allowed_values
            if feature.default_value is not None:
                prop["default"] = feature.default_value
            
            properties[feature.name] = prop
            
            if feature.default_value is None:
                required.append(feature.name)
        
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": group.name,
            "description": group.description,
            "type": "object",
            "properties": properties,
            "required": required,
        }


def create_erp_delay_risk_registry() -> FeatureRegistry:
    """Create the default feature registry for ERP delay risk model."""
    registry = FeatureRegistry()
    
    order_features = FeatureGroup(
        name="erp_delay_risk",
        description="Features for ERP order delay risk prediction",
        entity_key="order_id",
    )
    
    order_features.add_feature(FeatureDefinition(
        name="order_qty",
        dtype=FeatureType.INTEGER,
        description="Order quantity in units",
        source=FeatureSource.REQUEST,
        min_value=1,
        tags=["order", "quantity"],
    ))
    
    order_features.add_feature(FeatureDefinition(
        name="lead_time_days",
        dtype=FeatureType.FLOAT,
        description="Expected supplier lead time in days",
        source=FeatureSource.BATCH,
        min_value=0.0,
        tags=["supplier", "time"],
    ))
    
    order_features.add_feature(FeatureDefinition(
        name="vendor_reliability_score",
        dtype=FeatureType.FLOAT,
        description="Historical on-time delivery rate for vendor (0-1)",
        source=FeatureSource.BATCH,
        min_value=0.0,
        max_value=1.0,
        tags=["vendor", "reliability"],
    ))
    
    order_features.add_feature(FeatureDefinition(
        name="days_until_due",
        dtype=FeatureType.FLOAT,
        description="Days remaining until promised delivery",
        source=FeatureSource.REQUEST,
        min_value=0.0,
        tags=["time", "deadline"],
    ))
    
    order_features.add_feature(FeatureDefinition(
        name="historical_delay_rate",
        dtype=FeatureType.FLOAT,
        description="Historical delay rate for similar orders (0-1)",
        source=FeatureSource.BATCH,
        min_value=0.0,
        max_value=1.0,
        tags=["historical", "rate"],
    ))
    
    order_features.add_feature(FeatureDefinition(
        name="inventory_buffer_days",
        dtype=FeatureType.FLOAT,
        description="Days of inventory coverage available as buffer",
        source=FeatureSource.STREAMING,
        min_value=0.0,
        tags=["inventory", "buffer"],
    ))
    
    registry.register_group(order_features)
    
    return registry

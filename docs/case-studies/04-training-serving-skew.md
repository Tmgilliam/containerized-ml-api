# Case Study 4: Training/Serving Skew Causes 15% Accuracy Drop

## Company Profile

**Industry:** Aerospace parts manufacturer  
**Scale:** 3,000 orders/day, 500+ part SKUs  
**Challenge:** Complex feature engineering across multiple data sources

---

## The Problem

### Situation

An aerospace parts manufacturer built a sophisticated delay risk model using 47 features derived from ERP, supplier management, inventory, and logistics systems. The model achieved excellent offline metrics during development.

### Model Performance

| Metric | Training | Validation | Production (Month 1) |
|--------|----------|------------|---------------------|
| AUC | 0.91 | 0.89 | 0.74 |
| Precision | 0.85 | 0.82 | 0.68 |
| Recall | 0.88 | 0.84 | 0.71 |

### Investigation Timeline

**Week 1:** Data science team reviews model code — no issues found  
**Week 2:** Infrastructure team reviews deployment — API working correctly  
**Week 3:** Feature engineering audit reveals the problem

### Root Cause: Feature Engineering Mismatch

The data science team used pandas in Jupyter notebooks for feature engineering:

```python
# Training pipeline (notebooks/feature_engineering.ipynb)
df["reliability_score"] = df["on_time_deliveries"] / df["total_deliveries"]
df["reliability_score"] = df["reliability_score"].fillna(0.5)  # Default for new suppliers
```

The API team reimplemented in FastAPI:

```python
# API implementation (app/features.py)
def calculate_reliability_score(on_time: int, total: int) -> float:
    if total == 0:
        return 0.0  # Different default!
    return on_time / total
```

### The Discrepancies Found

| Feature | Training Logic | Serving Logic | Impact |
|---------|---------------|---------------|--------|
| `reliability_score` | NaN → 0.5 | 0 deliveries → 0.0 | High |
| `lead_time_days` | Clipped to [1, 90] | No clipping | Medium |
| `historical_delay_rate` | 2-year window | All-time window | High |
| `inventory_buffer_days` | Max(0, buffer) | Raw value (negative allowed) | Medium |

---

## The Solution

### Phase 1: Centralized Feature Registry

```python
from features import FeatureRegistry, FeatureDefinition, FeatureType

# Create central registry
registry = FeatureRegistry()

# Define each feature with its computation rules
registry.register(FeatureDefinition(
    name="reliability_score",
    dtype=FeatureType.FLOAT,
    source="supplier_metrics",
    description="Supplier on-time delivery rate (2-year window)",
    nullable=False,
    default_value=0.5,  # Explicit default
    min_value=0.0,
    max_value=1.0,
    computation_logic="on_time_deliveries_2y / total_deliveries_2y",
    tags=["supplier", "risk_factor"],
))

registry.register(FeatureDefinition(
    name="lead_time_days",
    dtype=FeatureType.FLOAT,
    source="order_data",
    description="Expected lead time in days (clipped 1-90)",
    nullable=False,
    default_value=14.0,
    min_value=1.0,
    max_value=90.0,  # Explicit clipping bounds
    tags=["order", "timing"],
))

registry.register(FeatureDefinition(
    name="historical_delay_rate",
    dtype=FeatureType.FLOAT,
    source="supplier_metrics",
    description="Supplier's historical delay rate (2-year window)",
    nullable=False,
    default_value=0.0,
    min_value=0.0,
    max_value=1.0,
    computation_logic="delayed_orders_2y / total_orders_2y",
    tags=["supplier", "risk_factor"],
))

registry.register(FeatureDefinition(
    name="inventory_buffer_days",
    dtype=FeatureType.FLOAT,
    source="inventory_system",
    description="Days of buffer inventory (floored at 0)",
    nullable=False,
    default_value=0.0,
    min_value=0.0,  # Floor at zero
    max_value=None,
    tags=["inventory", "protective"],
))

# Create feature group for delay risk model
registry.create_feature_group(
    name="delay_risk_features",
    feature_names=[
        "order_qty",
        "lead_time_days", 
        "vendor_reliability_score",
        "days_until_due",
        "historical_delay_rate",
        "inventory_buffer_days",
    ],
    description="Features for delay risk scoring model",
)

# Save registry
registry.save(Path("./model/feature_registry.json"))
```

### Phase 2: Feature Validation

```python
from features import FeatureValidator, FeatureRegistry
from pathlib import Path

# Load registry
registry = FeatureRegistry()
registry.load(Path("./model/feature_registry.json"))

# Create validator
validator = FeatureValidator(
    registry=registry,
    strict_mode=True,  # Fail on any violation
)

def validate_features(features: dict) -> dict:
    """Validate and fix features before prediction."""
    
    result = validator.validate(features, "delay_risk_features")
    
    if not result.is_valid:
        # Log violations for monitoring
        for violation in result.violations:
            logger.warning(f"Feature violation: {violation.feature_name} - "
                          f"{violation.violation_type}: {violation.message}")
        
        # Use corrected features
        return result.corrected_features
    
    return features

# Example usage in API
@app.post("/predict")
def predict(payload: PredictRequest):
    features = payload.model_dump()
    
    # Validate features (will apply defaults, clipping, etc.)
    validated_features = validate_features(features)
    
    # Predict with validated features
    result = delay_risk_model.predict(validated_features)
    
    return result
```

### Phase 3: Training/Serving Skew Detection

```python
from features import FeatureValidator

# During training, record feature distributions
def record_training_distributions(training_data: pd.DataFrame):
    """Record feature statistics during training for skew detection."""
    
    stats = {}
    for feature in training_data.columns:
        stats[feature] = {
            "mean": float(training_data[feature].mean()),
            "std": float(training_data[feature].std()),
            "min": float(training_data[feature].min()),
            "max": float(training_data[feature].max()),
            "null_rate": float(training_data[feature].isnull().mean()),
            "percentiles": {
                "p1": float(training_data[feature].quantile(0.01)),
                "p5": float(training_data[feature].quantile(0.05)),
                "p50": float(training_data[feature].quantile(0.50)),
                "p95": float(training_data[feature].quantile(0.95)),
                "p99": float(training_data[feature].quantile(0.99)),
            }
        }
    
    with open("./model/training_feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats

# During serving, detect skew
def detect_skew(serving_features: dict, training_stats: dict) -> dict:
    """Detect training/serving skew for a single prediction."""
    
    skew_alerts = []
    
    for feature, value in serving_features.items():
        if feature not in training_stats:
            skew_alerts.append({
                "feature": feature,
                "type": "unknown_feature",
                "message": f"Feature not in training data",
            })
            continue
        
        stats = training_stats[feature]
        
        # Check if value is outside training range
        if value < stats["min"] or value > stats["max"]:
            skew_alerts.append({
                "feature": feature,
                "type": "out_of_range",
                "value": value,
                "training_range": [stats["min"], stats["max"]],
                "message": f"Value {value} outside training range",
            })
        
        # Check if value is extreme (outside p1-p99)
        elif value < stats["percentiles"]["p1"] or value > stats["percentiles"]["p99"]:
            skew_alerts.append({
                "feature": feature,
                "type": "extreme_value",
                "value": value,
                "training_p1_p99": [
                    stats["percentiles"]["p1"],
                    stats["percentiles"]["p99"]
                ],
                "message": f"Value {value} is extreme (outside p1-p99)",
            })
    
    return {
        "has_skew": len(skew_alerts) > 0,
        "alerts": skew_alerts,
    }
```

### Phase 4: Unified Transform Pipeline

```python
from features.compute import FeatureTransformer

# Define transforms that apply to both training AND serving
transformer = FeatureTransformer()

# Register transform functions
@transformer.register("reliability_score")
def transform_reliability_score(value, context=None):
    """Transform reliability score with consistent logic."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.5  # Default for missing
    return max(0.0, min(1.0, value))  # Clip to [0, 1]

@transformer.register("lead_time_days")
def transform_lead_time(value, context=None):
    """Transform lead time with consistent clipping."""
    if value is None:
        return 14.0  # Default
    return max(1.0, min(90.0, float(value)))  # Clip to [1, 90]

@transformer.register("inventory_buffer_days")
def transform_buffer_days(value, context=None):
    """Transform buffer days with floor at zero."""
    if value is None:
        return 0.0
    return max(0.0, float(value))  # Floor at 0

@transformer.register("historical_delay_rate")
def transform_delay_rate(value, context=None):
    """Transform delay rate with bounds."""
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))  # Clip to [0, 1]

# Usage in training
def prepare_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent transforms during training."""
    return transformer.transform_dataframe(df)

# Usage in serving
def prepare_serving_features(features: dict) -> dict:
    """Apply consistent transforms during serving."""
    return transformer.transform_dict(features)
```

### Phase 5: Integration Tests

```python
import pytest
from features import FeatureValidator, FeatureRegistry
from features.compute import FeatureTransformer

class TestFeatureConsistency:
    """Test that training and serving produce identical features."""
    
    @pytest.fixture
    def registry(self):
        registry = FeatureRegistry()
        registry.load(Path("./model/feature_registry.json"))
        return registry
    
    @pytest.fixture
    def transformer(self):
        # Load the same transformer used in production
        return load_production_transformer()
    
    def test_null_handling_consistency(self, registry, transformer):
        """Test that null values are handled identically."""
        
        test_cases = [
            {"reliability_score": None},
            {"lead_time_days": None},
            {"inventory_buffer_days": None},
            {"historical_delay_rate": None},
        ]
        
        for features in test_cases:
            # Get training behavior (pandas fillna)
            training_result = apply_training_transforms(features)
            
            # Get serving behavior
            serving_result = transformer.transform_dict(features)
            
            for key in features:
                assert training_result[key] == serving_result[key], \
                    f"Mismatch for {key}: training={training_result[key]}, serving={serving_result[key]}"
    
    def test_boundary_consistency(self, registry, transformer):
        """Test that boundary values are handled identically."""
        
        test_cases = [
            {"lead_time_days": 0},    # Below min
            {"lead_time_days": 100},  # Above max
            {"reliability_score": -0.1},  # Below min
            {"reliability_score": 1.5},   # Above max
            {"inventory_buffer_days": -5},  # Negative
        ]
        
        for features in test_cases:
            training_result = apply_training_transforms(features)
            serving_result = transformer.transform_dict(features)
            
            for key in features:
                assert training_result[key] == serving_result[key], \
                    f"Boundary mismatch for {key}"
    
    def test_realistic_data_consistency(self, registry, transformer):
        """Test on realistic order data."""
        
        # Load a sample of production data
        sample_data = load_production_sample(n=1000)
        
        for features in sample_data:
            training_result = apply_training_transforms(features)
            serving_result = transformer.transform_dict(features)
            
            for key in training_result:
                assert abs(training_result[key] - serving_result[key]) < 1e-6, \
                    f"Production data mismatch for {key}"
```

### Phase 6: Online Skew Monitoring

```python
from features import FeatureValidator
from monitoring.drift import DriftDetector

class OnlineSkewMonitor:
    """Monitor for training/serving skew in real-time."""
    
    def __init__(self, training_stats_path: Path):
        with open(training_stats_path) as f:
            self.training_stats = json.load(f)
        
        self.skew_counts = defaultdict(int)
        self.total_predictions = 0
    
    def check(self, features: dict) -> dict:
        """Check for skew on each prediction."""
        
        self.total_predictions += 1
        skew_result = detect_skew(features, self.training_stats)
        
        for alert in skew_result["alerts"]:
            self.skew_counts[alert["feature"]] += 1
        
        return skew_result
    
    def get_skew_rates(self) -> dict:
        """Get skew rates by feature."""
        
        return {
            feature: count / self.total_predictions
            for feature, count in self.skew_counts.items()
        }
    
    def report(self) -> str:
        """Generate skew report."""
        
        rates = self.get_skew_rates()
        
        report_lines = [
            f"Training/Serving Skew Report",
            f"Total predictions: {self.total_predictions}",
            f"",
            f"Skew rates by feature:",
        ]
        
        for feature, rate in sorted(rates.items(), key=lambda x: -x[1]):
            report_lines.append(f"  {feature}: {rate:.2%}")
        
        return "\n".join(report_lines)

# Usage
monitor = OnlineSkewMonitor(Path("./model/training_feature_stats.json"))

@app.post("/predict")
def predict(payload: PredictRequest):
    features = payload.model_dump()
    
    # Check for skew
    skew_result = monitor.check(features)
    
    if skew_result["has_skew"]:
        logger.warning(f"Skew detected: {skew_result['alerts']}")
    
    # Validate and predict
    validated_features = validate_features(features)
    result = delay_risk_model.predict(validated_features)
    
    return result

# Daily skew report
@app.get("/monitoring/skew-report")
def get_skew_report():
    return {"report": monitor.report()}
```

---

## Results

### Before Implementation

| Issue | Occurrences |
|-------|-------------|
| Features with different null handling | 8 |
| Features with different boundary logic | 5 |
| Features with different time windows | 3 |
| Undocumented feature transformations | 12 |

### After Implementation

| Metric | Before | After |
|--------|--------|-------|
| Training/Serving AUC difference | 0.15 | 0.02 |
| Features with skew | 16 | 0 |
| Time to detect skew issues | Days/weeks | Real-time |
| Feature definitions documented | 35% | 100% |

### Model Performance Recovery

| Metric | Training | Validation | Production |
|--------|----------|------------|------------|
| AUC | 0.91 | 0.89 | **0.88** |
| Precision | 0.85 | 0.82 | **0.81** |
| Recall | 0.88 | 0.84 | **0.83** |

---

## Key Learnings

1. **Single source of truth** — Feature definitions must be centralized, not duplicated
2. **Explicit > implicit** — Document defaults, bounds, and computation logic explicitly
3. **Test consistency** — Automated tests catch skew before production
4. **Monitor continuously** — Skew can emerge gradually as data distributions shift
5. **Unified transforms** — Same code path for training and serving eliminates discrepancies

---

## Feature Registry Schema Example

```json
{
  "features": {
    "reliability_score": {
      "name": "reliability_score",
      "dtype": "float",
      "source": "supplier_metrics",
      "description": "Supplier on-time delivery rate (2-year window)",
      "nullable": false,
      "default_value": 0.5,
      "min_value": 0.0,
      "max_value": 1.0,
      "computation_logic": "on_time_deliveries_2y / total_deliveries_2y",
      "tags": ["supplier", "risk_factor"]
    }
  },
  "feature_groups": {
    "delay_risk_features": {
      "name": "delay_risk_features",
      "feature_names": [
        "order_qty",
        "lead_time_days",
        "vendor_reliability_score",
        "days_until_due",
        "historical_delay_rate",
        "inventory_buffer_days"
      ],
      "description": "Features for delay risk scoring model"
    }
  }
}
```

---

## Related Modules

- `features/registry.py` — Centralized feature definitions
- `features/validation.py` — Feature validation and skew detection
- `features/compute/transforms.py` — Unified transformation logic
- `features/serving/online.py` — Online feature serving
- `features/serving/offline.py` — Offline feature generation

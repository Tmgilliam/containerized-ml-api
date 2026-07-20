# Case Study 5: High-Value Orders Need Different Treatment

## Company Profile

**Industry:** Pharmaceutical distribution  
**Scale:** 12,000 orders/day across 3 distribution centers  
**Challenge:** One-size-fits-all model doesn't optimize for business value

---

## The Problem

### Situation

A pharmaceutical distributor uses an ML model to flag orders at risk of delay. The model treats all orders equally, optimizing for overall accuracy across the entire order population.

### The Hidden Cost

Analysis revealed severe misalignment between model behavior and business impact:

| Order Category | Volume | Avg Value | Model Attention | Business Impact |
|---------------|--------|-----------|-----------------|-----------------|
| Standard | 75% | $2,500 | 73% | 25% of revenue |
| High-Value | 20% | $25,000 | 24% | 55% of revenue |
| Critical/Life-Saving | 5% | $50,000+ | 3% | 20% of revenue |

### The Discovery

The model optimized for accuracy, which meant:
- Missing 8 high-value orders per day (acceptable statistically)
- Each miss cost $15K+ in expedited shipping or lost business
- Total annual impact: **$4.2M in avoidable costs**

### Root Cause Analysis

```python
# The model optimized for this:
total_accuracy = correct_predictions / total_predictions  # 87%

# But business value requires optimizing for this:
weighted_accuracy = sum(
    correct * order_value for order, correct in predictions
) / sum(order_value for order in predictions)

# Result: 87% accuracy, but only 71% weighted accuracy
```

---

## The Solution

### Phase 1: Multi-Model Architecture

Deploy specialized models for different order segments:

```python
from app.models import ModelRegistry, ModelRouter, RoutingRule

# Initialize registry
registry = ModelRegistry()

# Register segment-specific models
registry.register_model(
    name="delay-risk-standard",
    model_path="./models/standard_orders_v1.pkl",
    version="1.0.0",
    metadata={
        "segment": "standard",
        "order_value_range": [0, 10000],
        "training_data": "2024_standard_orders.csv",
    }
)

registry.register_model(
    name="delay-risk-high-value",
    model_path="./models/high_value_orders_v1.pkl",
    version="1.0.0",
    metadata={
        "segment": "high_value",
        "order_value_range": [10000, 50000],
        "training_data": "2024_high_value_orders.csv",
        "custom_threshold": 0.35,  # Lower threshold = more conservative
    }
)

registry.register_model(
    name="delay-risk-critical",
    model_path="./models/critical_orders_v1.pkl",
    version="1.0.0",
    metadata={
        "segment": "critical",
        "order_value_range": [50000, None],
        "training_data": "2024_critical_orders.csv",
        "custom_threshold": 0.25,  # Very conservative
    }
)

# Load all models
registry.load_model("delay-risk-standard")
registry.load_model("delay-risk-high-value")
registry.load_model("delay-risk-critical")
```

### Phase 2: Intelligent Routing

```python
from app.models import ModelRouter, RoutingRule

# Create router with segment-based rules
router = ModelRouter(registry=registry)

# Define routing rules (order matters - first match wins)
router.add_rule(RoutingRule(
    name="critical-orders",
    condition=lambda features: features.get("order_value", 0) >= 50000,
    model_name="delay-risk-critical",
    priority=100,
))

router.add_rule(RoutingRule(
    name="high-value-orders",
    condition=lambda features: features.get("order_value", 0) >= 10000,
    model_name="delay-risk-high-value",
    priority=50,
))

router.add_rule(RoutingRule(
    name="standard-orders",
    condition=lambda features: True,  # Default
    model_name="delay-risk-standard",
    priority=0,
))

# Predict with automatic routing
def predict(features: dict) -> dict:
    """Route to appropriate model based on order value."""
    
    result, model_info = router.route_and_predict(features)
    
    return {
        **result,
        "model_used": model_info["model_name"],
        "model_version": model_info["model_version"],
        "segment": model_info["metadata"]["segment"],
    }
```

### Phase 3: Segment-Specific Thresholds

```python
def predict_with_segment_threshold(features: dict) -> dict:
    """Apply segment-specific decision thresholds."""
    
    result, model_info = router.route_and_predict(features)
    
    # Get segment-specific threshold
    metadata = model_info.get("metadata", {})
    threshold = metadata.get("custom_threshold", 0.5)
    
    # Apply threshold
    risk_score = result.get("risk_score", 0.5)
    delay_risk = 1 if risk_score >= threshold else 0
    
    return {
        "delay_risk": delay_risk,
        "risk_score": risk_score,
        "threshold_applied": threshold,
        "model_used": model_info["model_name"],
        "segment": metadata.get("segment", "unknown"),
    }
```

### Phase 4: Segment-Aware Training

Train models with segment-specific objectives:

```python
from retraining.pipelines import TrainingPipeline

def train_segment_models():
    """Train separate models for each segment."""
    
    segments = {
        "standard": {
            "filter": lambda df: df["order_value"] < 10000,
            "class_weight": "balanced",
            "threshold": 0.5,
        },
        "high_value": {
            "filter": lambda df: (df["order_value"] >= 10000) & (df["order_value"] < 50000),
            "class_weight": {0: 1, 1: 3},  # Weight delays 3x
            "threshold": 0.35,
        },
        "critical": {
            "filter": lambda df: df["order_value"] >= 50000,
            "class_weight": {0: 1, 1: 5},  # Weight delays 5x
            "threshold": 0.25,
        },
    }
    
    # Load full training data
    full_data = pd.read_csv("./data/training_orders.csv")
    
    for segment_name, config in segments.items():
        print(f"\nTraining {segment_name} model...")
        
        # Filter to segment
        segment_data = full_data[config["filter"](full_data)]
        print(f"  Samples: {len(segment_data)}")
        
        # Create pipeline with segment-specific config
        pipeline = TrainingPipeline(
            model_name=f"delay-risk-{segment_name}",
            output_dir=Path(f"./models/{segment_name}"),
            classifier_params={
                "class_weight": config["class_weight"],
            },
        )
        
        # Prepare features and labels
        X = segment_data[FEATURE_COLUMNS]
        y = segment_data["was_delayed"]
        
        # Train with validation
        result = pipeline.train(
            X=X,
            y=y,
            test_size=0.2,
            cv_folds=5,
        )
        
        print(f"  AUC: {result.metrics['auc']:.4f}")
        print(f"  Recall: {result.metrics['recall']:.4f}")
        print(f"  Precision: {result.metrics['precision']:.4f}")
        
        # Save with metadata
        pipeline.save_model(
            path=Path(f"./models/{segment_name}_v1.pkl"),
            metadata={
                "segment": segment_name,
                "custom_threshold": config["threshold"],
                "training_samples": len(segment_data),
                "class_weight": str(config["class_weight"]),
            }
        )
```

### Phase 5: Value-Weighted Metrics

```python
def calculate_value_weighted_metrics(predictions: list[dict]) -> dict:
    """Calculate business-value-weighted metrics."""
    
    total_value = sum(p["order_value"] for p in predictions)
    
    # Value-weighted accuracy
    correct_value = sum(
        p["order_value"] 
        for p in predictions 
        if p["predicted"] == p["actual"]
    )
    weighted_accuracy = correct_value / total_value
    
    # Value-weighted recall (for delays)
    delayed_orders = [p for p in predictions if p["actual"] == 1]
    total_delayed_value = sum(p["order_value"] for p in delayed_orders)
    
    caught_delayed_value = sum(
        p["order_value"] 
        for p in delayed_orders 
        if p["predicted"] == 1
    )
    weighted_recall = caught_delayed_value / total_delayed_value if total_delayed_value > 0 else 1.0
    
    # Cost of misses
    missed_value = sum(
        p["order_value"] 
        for p in delayed_orders 
        if p["predicted"] == 0
    )
    
    return {
        "weighted_accuracy": weighted_accuracy,
        "weighted_recall": weighted_recall,
        "missed_delay_value": missed_value,
        "total_predictions": len(predictions),
        "by_segment": calculate_segment_metrics(predictions),
    }

def calculate_segment_metrics(predictions: list[dict]) -> dict:
    """Break down metrics by segment."""
    
    segments = {}
    
    for segment in ["standard", "high_value", "critical"]:
        segment_preds = [p for p in predictions if p["segment"] == segment]
        
        if not segment_preds:
            continue
        
        total = len(segment_preds)
        correct = sum(1 for p in segment_preds if p["predicted"] == p["actual"])
        
        # Recall for delays
        actual_delays = [p for p in segment_preds if p["actual"] == 1]
        caught = sum(1 for p in actual_delays if p["predicted"] == 1)
        
        segments[segment] = {
            "count": total,
            "accuracy": correct / total,
            "delay_recall": caught / len(actual_delays) if actual_delays else 1.0,
            "total_value": sum(p["order_value"] for p in segment_preds),
        }
    
    return segments
```

### Phase 6: Monitoring Dashboard

```python
# Expose segment-specific metrics
@app.get("/monitoring/segment-metrics")
def get_segment_metrics():
    """Get real-time metrics by segment."""
    
    # Collect recent predictions
    recent_predictions = fetch_recent_predictions(hours=24)
    
    metrics = calculate_value_weighted_metrics(recent_predictions)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "period_hours": 24,
        "total_predictions": metrics["total_predictions"],
        "overall": {
            "weighted_accuracy": metrics["weighted_accuracy"],
            "weighted_recall": metrics["weighted_recall"],
            "missed_delay_value": metrics["missed_delay_value"],
        },
        "by_segment": metrics["by_segment"],
    }

# Alert on high-value misses
def check_high_value_misses():
    """Alert if too many high-value orders are being missed."""
    
    recent = fetch_recent_predictions(hours=1)
    high_value = [p for p in recent if p["order_value"] >= 10000]
    
    # Check recall for high-value segment
    delays = [p for p in high_value if p["actual"] == 1]
    
    if not delays:
        return
    
    caught = sum(1 for p in delays if p["predicted"] == 1)
    recall = caught / len(delays)
    
    if recall < 0.95:  # Alert if recall drops below 95%
        send_alert(
            severity="HIGH",
            title="High-Value Order Recall Dropping",
            message=f"High-value order delay recall is {recall:.1%} (threshold: 95%)",
            details={
                "missed_count": len(delays) - caught,
                "missed_value": sum(p["order_value"] for p in delays if p["predicted"] == 0),
            }
        )
```

---

## Results

### Before Implementation (Single Model)

| Segment | Volume | Delay Recall | Missed Delays/Day | Cost/Day |
|---------|--------|--------------|-------------------|----------|
| Standard | 9,000 | 88% | 36 | $5,400 |
| High-Value | 2,400 | 82% | 8 | $12,000 |
| Critical | 600 | 78% | 3 | $22,500 |
| **Total** | 12,000 | 86% | 47 | **$39,900** |

### After Implementation (Multi-Model)

| Segment | Volume | Delay Recall | Missed Delays/Day | Cost/Day |
|---------|--------|--------------|-------------------|----------|
| Standard | 9,000 | 85% | 45 | $6,750 |
| High-Value | 2,400 | 94% | 2 | $3,000 |
| Critical | 600 | 97% | 0.5 | $3,750 |
| **Total** | 12,000 | 88% | 47.5 | **$13,500** |

### Annual Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Annual missed delay cost | $14.6M | $4.9M | **-$9.7M (66%)** |
| High-value recall | 82% | 94% | +12 points |
| Critical recall | 78% | 97% | +19 points |

---

## Key Learnings

1. **Business value ≠ equal treatment** — Segment-specific models can optimize for business impact
2. **Threshold tuning by segment** — Conservative thresholds for high-value segments catch more delays
3. **Class weights matter** — Weighting positive examples higher improves recall for critical segments
4. **Monitor value-weighted metrics** — Overall accuracy can hide segment-specific problems
5. **Small volume, big impact** — 5% of orders (critical) drove 20% of revenue impact

---

## Model Selection Decision Tree

```
                    ┌─────────────────┐
                    │  Incoming Order │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Order Value ≥   │
                    │    $50,000?     │
                    └────────┬────────┘
                         Yes │ No
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Critical Model   │         │ Order Value ≥     │
    │ Threshold: 0.25   │         │    $10,000?       │
    └───────────────────┘         └─────────┬─────────┘
                                        Yes │ No
                             ┌──────────────┴──────────────┐
                             │                             │
                   ┌─────────▼─────────┐         ┌─────────▼─────────┐
                   │ High-Value Model  │         │  Standard Model   │
                   │ Threshold: 0.35   │         │ Threshold: 0.50   │
                   └───────────────────┘         └───────────────────┘
```

---

## Related Modules

- `app/models/registry.py` — Multi-model registration and management
- `app/models/router.py` — Intelligent routing based on features
- `retraining/pipelines/train_pipeline.py` — Segment-specific training
- `monitoring/drift/detector.py` — Per-segment drift monitoring

# Case Study 2: Failed Model Rollout Causes $2M in Expedited Shipping

## Company Profile

**Industry:** Consumer electronics manufacturer  
**Scale:** 15,000 orders/day during peak season  
**Challenge:** Holiday demand surge with tight delivery windows

---

## The Problem

### Situation

The data science team developed an improved delay risk model (v2.0) that showed 8% better AUC in offline evaluation. Following standard practice, they ran an A/B test with 5% of traffic for 2 weeks before full rollout.

### The A/B Test Results

```
Model v1.0 (Control):
  Samples: 10,500
  Predicted delay rate: 18.2%
  
Model v2.0 (Treatment):
  Samples: 10,500
  Predicted delay rate: 14.1%
  
Conclusion: v2.0 appears more conservative (fewer false alarms)
Decision: Promote v2.0 to 100%
```

### The Disaster

Within 72 hours of full rollout:
- 847 high-value orders missed delivery windows
- $2.1M spent on expedited shipping to recover
- Customer satisfaction scores dropped 15 points
- Procurement team lost confidence in the system

### Root Cause Analysis

1. **Insufficient sample size** — 10,500 samples during a slow period wasn't representative
2. **Missing segment analysis** — v2.0 performed well on average but failed on high-value orders (>$10K)
3. **No statistical power calculation** — The test couldn't detect a 4% difference in delay rates
4. **Holiday surge not captured** — Test ran in October; rollout happened during November peak

```python
# What the team SHOULD have calculated:
from app.experiments import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(alpha=0.05)

required_samples = analyzer.calculate_required_sample_size(
    baseline_rate=0.18,  # 18% baseline delay rate
    minimum_detectable_effect=0.04,  # Detect 4% difference
    power=0.8
)

print(f"Required samples per variant: {required_samples}")
# Output: Required samples per variant: 2,401
# But this assumes uniform distribution — high-value orders need separate analysis!
```

---

## The Solution

### Phase 1: Proper A/B Testing Infrastructure

#### Configure Experiment with Statistical Rigor

```python
from app.experiments import (
    ExperimentRouter,
    Experiment,
    ModelVariant,
    ExperimentMetrics,
    ExperimentAnalyzer,
)
from datetime import datetime, timezone, timedelta

# Define model variants
control = ModelVariant(
    name="v1.0-control",
    model_version="1.0.0",
    predict_fn=model_v1.predict,
    weight=50.0,  # 50% traffic
)

treatment = ModelVariant(
    name="v2.0-treatment", 
    model_version="2.0.0",
    predict_fn=model_v2.predict,
    weight=50.0,  # 50% traffic
)

# Create experiment with proper duration
experiment = Experiment(
    name="delay-risk-v2-rollout",
    variants=[control, treatment],
    start_time=datetime.now(timezone.utc),
    end_time=datetime.now(timezone.utc) + timedelta(days=30),  # 30 days, not 14
    sticky_assignment=True,  # Same user gets same variant
)

# Initialize router
router = ExperimentRouter(default_model=control)
router.register_experiment(experiment)

# Initialize metrics collection
metrics = ExperimentMetrics(max_records=500000)
```

#### Track Every Prediction

```python
import uuid

def predict_with_experiment(features: dict, user_id: str = None):
    """Route prediction through experiment and track metrics."""
    
    request_id = str(uuid.uuid4())
    user_id = user_id or request_id
    
    # Route to variant
    start = time.perf_counter()
    result, variant_name, exp_name = router.route(
        features=features,
        user_id=user_id,
        experiment_name="delay-risk-v2-rollout",
    )
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Record for analysis
    metrics.record_prediction(
        experiment_name=exp_name,
        variant_name=variant_name,
        user_id=user_id,
        features=features,
        prediction=result,
        latency_ms=latency_ms,
        prediction_id=request_id,
    )
    
    return result, request_id
```

#### Analyze with Statistical Rigor

```python
def analyze_experiment_results():
    """Run statistical analysis on experiment data."""
    
    analyzer = ExperimentAnalyzer(
        alpha=0.05,
        min_samples_per_variant=1000,
    )
    
    # Export data by variant
    data = metrics.export_for_analysis("delay-risk-v2-rollout")
    
    # Analyze all metrics
    results = analyzer.analyze_experiment(
        experiment_name="delay-risk-v2-rollout",
        data=data,
        control_variant="v1.0-control",
        metrics=["risk_score", "latency_ms"],
    )
    
    for result in results:
        print(f"\n{result.metric_name} Analysis:")
        print(f"  Control mean: {result.control_mean:.4f}")
        print(f"  Treatment mean: {result.treatment_mean:.4f}")
        print(f"  Difference: {result.absolute_difference:.4f} "
              f"({result.relative_difference:.2%})")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Significant: {result.is_significant}")
        print(f"  95% CI: [{result.confidence_interval[0]:.4f}, "
              f"{result.confidence_interval[1]:.4f}]")
        print(f"  Statistical Power: {result.statistical_power:.2f}")
        
        if result.statistical_power < 0.8:
            print(f"  ⚠️ WARNING: Low statistical power. "
                  f"Need more samples for reliable conclusion.")
    
    return results
```

### Phase 2: Shadow Mode Validation

Before any A/B test, run the new model in shadow mode on 100% of traffic:

```python
from app.shadow import ShadowRunner, ShadowConfig, ShadowComparator, ShadowStorage

# Configure shadow mode
config = ShadowConfig(
    enabled=True,
    sample_rate=1.0,  # 100% of traffic
    async_execution=True,  # Don't impact production latency
    log_comparisons=True,
)

# Initialize shadow runner
shadow_runner = ShadowRunner(
    production_model=model_v1.predict,
    shadow_model=model_v2.predict,
    config=config,
)

# Storage for analysis
storage = ShadowStorage(storage_path=Path("./shadow_results"))

# Comparator for promotion decision
comparator = ShadowComparator(
    agreement_threshold=0.95,  # 95% predictions must match
    probability_diff_threshold=0.1,  # Max 10% probability difference
)

def predict_with_shadow(features: dict):
    """Production prediction with shadow validation."""
    
    result, shadow_pred = shadow_runner.predict(features)
    
    # Store for analysis (async)
    if shadow_pred:
        storage.store(shadow_pred.to_dict())
        comparator.add_comparison(
            production_result=shadow_pred.production_result,
            shadow_result=shadow_pred.shadow_result,
            features=features,
        )
    
    return result
```

#### Check Promotion Readiness

```python
def check_shadow_promotion():
    """Determine if shadow model is ready for promotion."""
    
    # Analyze comparisons
    result = comparator.analyze()
    
    print(f"Shadow Analysis ({result.total_comparisons} comparisons):")
    print(f"  Agreement Rate: {result.prediction_agreement_rate:.2%}")
    print(f"  Mean Prob Diff: {result.mean_probability_difference:.4f}")
    print(f"  Production Positive Rate: {result.production_positive_rate:.2%}")
    print(f"  Shadow Positive Rate: {result.shadow_positive_rate:.2%}")
    
    # Check promotion readiness
    is_ready, issues = comparator.is_shadow_ready_for_promotion()
    
    if is_ready:
        print("\n✅ Shadow model is READY for promotion")
    else:
        print("\n❌ Shadow model is NOT ready for promotion")
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Segment analysis for high-value orders
    segments = comparator.segment_analysis("order_value_bucket")
    print("\nSegment Analysis:")
    for segment, stats in segments.items():
        print(f"  {segment}: agreement={stats['agreement_rate']:.2%}, "
              f"count={stats['count']}")
    
    return is_ready, issues
```

### Phase 3: Segment-Specific Validation

```python
def validate_high_value_segment():
    """Ensure model performs well on high-value orders."""
    
    # Get disagreements
    disagreements = comparator.get_disagreements(limit=100)
    
    # Analyze by order value
    high_value_disagreements = [
        d for d in disagreements
        if d.get("features", {}).get("order_value", 0) > 10000
    ]
    
    print(f"High-value order disagreements: {len(high_value_disagreements)}")
    
    # If high-value segment has problems, don't promote
    if len(high_value_disagreements) > 10:
        print("⚠️ Too many disagreements on high-value orders")
        print("Recommendation: Investigate before promotion")
        
        for d in high_value_disagreements[:5]:
            print(f"  Order value: ${d['features'].get('order_value', 0):,.0f}")
            print(f"    Production: {d['production_class']} "
                  f"(prob={d['production_probability']:.3f})")
            print(f"    Shadow: {d['shadow_class']} "
                  f"(prob={d['shadow_probability']:.3f})")
```

---

## Results

### Before Implementation

| Phase | Duration | Traffic | Outcome |
|-------|----------|---------|---------|
| Offline eval | 1 week | 0% | 8% AUC improvement |
| A/B test | 2 weeks | 5% | "Looks good" |
| Full rollout | 3 days | 100% | $2.1M loss |

### After Implementation

| Phase | Duration | Traffic | Outcome |
|-------|----------|---------|---------|
| Offline eval | 1 week | 0% | 8% AUC improvement |
| Shadow mode | 2 weeks | 100% (shadow) | Identified segment issues |
| Model fix | 1 week | 0% | Fixed high-value logic |
| Shadow mode v2 | 1 week | 100% (shadow) | 97% agreement |
| A/B test | 4 weeks | 50% | Statistically significant |
| Full rollout | Ongoing | 100% | +6% accuracy, no incidents |

---

## Key Learnings

1. **Shadow mode first** — Test on 100% of traffic without risk before any A/B test
2. **Calculate required sample sizes** — Don't guess at test duration
3. **Segment analysis is critical** — Overall metrics hide segment-specific failures
4. **Statistical power matters** — Low power means unreliable conclusions
5. **Gate promotion on data** — `is_shadow_ready_for_promotion()` removes subjectivity

---

## Promotion Checklist

Before promoting any model:

- [ ] Shadow mode run for 2+ weeks
- [ ] Agreement rate > 95%
- [ ] Mean probability difference < 0.1
- [ ] Segment analysis shows no degradation in critical segments
- [ ] A/B test with calculated sample size
- [ ] Statistical significance (p < 0.05)
- [ ] Statistical power > 0.8
- [ ] Outcome data confirms predictions (if available)

---

## Related Modules

- `app/experiments/router.py` — Traffic routing between variants
- `app/experiments/metrics.py` — Prediction tracking
- `app/experiments/analysis.py` — Statistical significance tests
- `app/shadow/runner.py` — Shadow mode execution
- `app/shadow/comparator.py` — Production vs shadow comparison
- `app/shadow/storage.py` — Result persistence

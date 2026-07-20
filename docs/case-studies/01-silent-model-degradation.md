# Case Study 1: Silent Model Degradation at Scale

## Company Profile

**Industry:** Global logistics provider  
**Scale:** 50,000 order predictions daily  
**Region:** North America, Europe, Asia-Pacific

---

## The Problem

### Situation

A global logistics provider deployed an ML-based delay risk scoring system to help procurement teams prioritize orders requiring intervention. The model achieved 0.87 AUC during validation and was deployed to production with confidence.

### Symptoms

Over a 6-month period, procurement teams reported:
- Increasing false positives (orders flagged as high-risk that delivered on time)
- Decreasing trust in model recommendations
- Return to manual review processes for "important" orders
- 40% increase in expedited shipping costs

### The Mystery

The model never crashed. No errors appeared in logs. Health checks passed. The API responded with sub-100ms latency. By all operational metrics, the system was healthy.

### Root Cause Discovery

After extensive investigation, the data science team discovered:

1. **Supplier lead times shifted post-pandemic** — Average lead times dropped from 14 days to 9 days as supply chains normalized
2. **The model was trained on 2019-2022 data** — It expected longer lead times and flagged "short" lead times as anomalous
3. **Feature distributions drifted silently** — No monitoring existed to detect the shift

```
Training Data (2019-2022):
  lead_time_days: mean=14.2, std=4.8

Production Data (6 months later):
  lead_time_days: mean=9.1, std=3.2

PSI Score: 0.34 (CRITICAL - would have triggered alert)
```

---

## The Solution

### Implementation with Drift Detection

#### Step 1: Establish Reference Distributions

```python
from monitoring.drift import DriftDetector
import numpy as np

# Load training data statistics (computed during model training)
reference_data = {
    "order_qty": np.array([...]),  # Training distribution
    "lead_time_days": np.array([...]),
    "vendor_reliability_score": np.array([...]),
    "days_until_due": np.array([...]),
    "historical_delay_rate": np.array([...]),
    "inventory_buffer_days": np.array([...]),
}

# Initialize detector with reference distributions
detector = DriftDetector(
    reference_data=reference_data,
    model_version="1.0.0",
    feature_names=list(reference_data.keys()),
)

# Save reference for production use
detector.save_reference(Path("./model/reference_distributions.json"))
```

#### Step 2: Schedule Daily Drift Checks

```python
from monitoring.drift import DriftDetector
from monitoring.alerts import create_default_notifier, AlertChannel
from datetime import datetime, timedelta

def daily_drift_check():
    """Run daily drift analysis on last 24 hours of predictions."""
    
    # Load reference detector
    detector = DriftDetector.load_reference(
        Path("./model/reference_distributions.json")
    )
    
    # Fetch last 24 hours of production data
    current_data = fetch_production_features(
        start_time=datetime.now() - timedelta(days=1),
        end_time=datetime.now(),
    )
    
    # Run drift analysis
    result = detector.analyze(
        current_data=current_data,
        alert_threshold=DriftSeverity.MODERATE,
    )
    
    # Log results
    print(f"Drift Analysis Results:")
    print(f"  Overall Severity: {result.overall_severity.value}")
    print(f"  Alert Triggered: {result.alert_triggered}")
    
    for feature in result.feature_results:
        print(f"  {feature.feature_name}: PSI={feature.psi:.4f}, "
              f"Severity={feature.severity.value}")
    
    # Send alerts if needed
    if result.alert_triggered:
        notifier = create_default_notifier()
        notifier.send_drift_alert(result)
    
    return result
```

#### Step 3: Configure Alert Channels

```python
from monitoring.alerts import (
    AlertNotifier, 
    AlertChannel,
    SlackNotifier,
    PagerDutyNotifier,
)
from monitoring.drift import DriftSeverity

# Create notifier with multiple channels
notifier = AlertNotifier()

# Register Slack for moderate issues
notifier.register_channel(
    AlertChannel.SLACK,
    SlackNotifier(webhook_url="https://hooks.slack.com/services/...")
)

# Register PagerDuty for critical issues
notifier.register_channel(
    AlertChannel.PAGERDUTY,
    PagerDutyNotifier(routing_key="your-pagerduty-key")
)

# Configure severity routing
notifier.configure_severity_routing(
    DriftSeverity.MODERATE,
    [AlertChannel.LOG, AlertChannel.SLACK]
)
notifier.configure_severity_routing(
    DriftSeverity.HIGH,
    [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
)
notifier.configure_severity_routing(
    DriftSeverity.CRITICAL,
    [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
)
```

#### Step 4: Integrate Feedback Loop

```python
from retraining import OutcomeIngester, LabelMatcher, RetrainTrigger

# Ingest actual outcomes when orders complete
ingester = OutcomeIngester(storage_backend="local")

def on_order_delivered(order_id: str, was_delayed: bool):
    """Called when order delivery status is confirmed."""
    ingester.ingest(
        entity_id=order_id,
        outcome=was_delayed,
        metadata={"source": "erp_system"}
    )

# Weekly accuracy check
def weekly_accuracy_check():
    matcher = LabelMatcher(
        predictions_path=Path("./predictions"),
        outcomes_path=Path("./outcomes"),
    )
    
    matcher.load_predictions()
    matcher.load_outcomes()
    
    report = matcher.accuracy_report()
    
    print(f"Weekly Accuracy Report:")
    print(f"  Total Matched: {report['total_matched']}")
    print(f"  Accuracy: {report['accuracy']:.2%}")
    print(f"  Precision: {report['precision']:.2%}")
    print(f"  Recall: {report['recall']:.2%}")
    
    # Check if retraining is needed
    trigger = RetrainTrigger()
    result = trigger.check_accuracy_trigger(report)
    
    if result.should_retrain:
        print(f"RETRAINING RECOMMENDED: {result.reason}")
    
    return report
```

---

## Results

### Before Implementation

| Metric | Value |
|--------|-------|
| Time to detect drift | 6 months (manual discovery) |
| False positive rate | 34% |
| Procurement team trust | Low |
| Expedited shipping costs | +40% |

### After Implementation

| Metric | Value |
|--------|-------|
| Time to detect drift | 2 weeks (automated alert) |
| False positive rate | 12% (after retraining) |
| Procurement team trust | High |
| Expedited shipping costs | -25% |

---

## Key Learnings

1. **Models degrade silently** — Operational metrics (latency, uptime) don't capture prediction quality
2. **Drift detection is essential** — PSI and KS tests catch distribution shifts before business impact
3. **Feedback loops close the gap** — Without outcome data, you can't measure real accuracy
4. **Automate everything** — Daily drift checks and weekly accuracy reports prevent surprises

---

## Grafana Dashboard

Import the provided dashboard configuration:

```bash
# Import drift monitoring dashboard
cp monitoring/dashboards/grafana_drift.json /etc/grafana/dashboards/
```

The dashboard shows:
- PSI over time by feature
- Current PSI gauges with thresholds
- KS statistic trends
- Overall drift severity indicator
- Prediction score distribution

---

## Related Modules

- `monitoring/drift/detector.py` — Core drift detection logic
- `monitoring/drift/metrics.py` — PSI, KS, JSD calculations
- `monitoring/alerts/notifier.py` — Multi-channel alerting
- `retraining/collectors/outcome_ingester.py` — Outcome collection
- `retraining/collectors/label_matcher.py` — Prediction/outcome matching

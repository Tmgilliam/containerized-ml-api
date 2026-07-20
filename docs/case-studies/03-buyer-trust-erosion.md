# Case Study 3: "Why Did You Flag This Order?" — Buyer Trust Erosion

## Company Profile

**Industry:** Industrial equipment distributor  
**Users:** 45 procurement buyers  
**Orders:** 8,000/day across 200 suppliers

---

## The Problem

### Situation

An industrial equipment distributor deployed a delay risk scoring system to help procurement buyers prioritize their workload. The model identified high-risk orders that needed intervention (supplier follow-up, alternative sourcing, etc.).

### Initial Success

The first 3 months showed promising results:
- 23% reduction in late deliveries
- Buyers focused attention on flagged orders
- Supplier relationships improved through proactive communication

### The Decline

By month 6:
- Buyers began ignoring model recommendations
- Manual review processes re-emerged
- Only 20% of flagged orders received intervention
- Late deliveries returned to baseline levels

### User Feedback

From buyer interviews:

> "The system flags orders but doesn't tell me *why*. When I call a supplier about a 'high-risk' order, I can't explain what triggered the alert. It makes me look like I don't know what I'm doing."

> "I've been doing this for 15 years. The model flagged an order from our most reliable supplier. It was wrong, but I couldn't understand why it made that decision."

> "My manager asked me to justify the extra resources we're spending on 'high-risk' orders. I couldn't explain the model's logic, so we stopped trusting it."

### Root Cause

The model was a **black box**. It produced accurate predictions but offered zero insight into:
- Which features drove each prediction
- Why specific orders were flagged
- What would change the risk assessment

---

## The Solution

### Phase 1: Add SHAP Explanations

```python
from app.explainers import SHAPExplainer, ExplanationResult
from app.model import delay_risk_model
import pandas as pd

# Initialize explainer with trained model
explainer = SHAPExplainer(
    model=delay_risk_model._classifier,
    feature_names=delay_risk_model.feature_names,
    model_version=delay_risk_model.version,
)

# Optional: Set background data for better explanations
training_data = pd.read_csv("./data/training_data.csv")
explainer = SHAPExplainer(
    model=delay_risk_model._classifier,
    feature_names=delay_risk_model.feature_names,
    background_data=training_data[delay_risk_model.feature_names],
    model_version=delay_risk_model.version,
)
```

### Phase 2: Enhanced Prediction Endpoint

```python
from fastapi import FastAPI, Request
from app.schemas import PredictRequest, PredictResponse
from pydantic import BaseModel
from typing import Optional

class ExplainedPredictResponse(BaseModel):
    """Prediction with explanation."""
    delay_risk: int
    risk_score: float
    confidence: float
    model_version: str
    explanation: Optional[dict] = None
    summary: Optional[str] = None

@app.post("/predict/explained", response_model=ExplainedPredictResponse)
def predict_with_explanation(
    payload: PredictRequest,
    request: Request,
    include_explanation: bool = True,
) -> ExplainedPredictResponse:
    """Score order with optional explanation."""
    
    # Get base prediction
    features = payload.model_dump()
    result = delay_risk_model.predict(features)
    
    response = ExplainedPredictResponse(
        delay_risk=result["delay_risk"],
        risk_score=result["risk_score"],
        confidence=result["confidence"],
        model_version=result["model_version"],
    )
    
    if include_explanation:
        # Generate SHAP explanation
        explanation = explainer.explain(features, result)
        
        response.explanation = explanation.to_dict()
        response.summary = explanation.summary(top_n=3)
    
    return response
```

### Phase 3: Human-Readable Summaries

The `ExplanationResult.summary()` method generates buyer-friendly explanations:

```python
# Example output for a high-risk order:

explanation = explainer.explain(features, prediction)
print(explanation.summary(top_n=3))
```

**Output:**
```
High delay risk (78.3% probability)

Factors increasing risk:
  - vendor_reliability_score: 0.62 (+0.234)
  - lead_time_days: 21.0 (+0.156)
  - historical_delay_rate: 0.28 (+0.089)

Factors decreasing risk:
  - inventory_buffer_days: 8.0 (-0.045)
  - days_until_due: 14.0 (-0.023)
```

### Phase 4: Counterfactual Analysis with LIME

Help buyers understand what would change the prediction:

```python
from app.explainers import LIMEExplainer

# Initialize LIME explainer
lime_explainer = LIMEExplainer(
    predict_fn=lambda x: delay_risk_model._classifier.predict_proba(x),
    feature_names=delay_risk_model.feature_names,
    training_data=training_data,
    model_version=delay_risk_model.version,
)

def get_counterfactual_advice(features: dict, prediction: dict):
    """What would flip this prediction?"""
    
    if prediction["delay_risk"] == 0:
        return {"status": "Low risk - no action needed"}
    
    # Analyze what changes would reduce risk
    analysis = lime_explainer.counterfactual_analysis(
        features=features,
        target_class=0,  # We want low risk
        num_samples=1000,
    )
    
    return analysis

# Example usage:
features = {
    "order_qty": 500,
    "lead_time_days": 21,
    "vendor_reliability_score": 0.62,
    "days_until_due": 14,
    "historical_delay_rate": 0.28,
    "inventory_buffer_days": 8,
}

prediction = delay_risk_model.predict(features)
advice = get_counterfactual_advice(features, prediction)

print("Counterfactual Analysis:")
print(f"  Current class: {advice['current_class']} "
      f"(probability: {advice['current_probability']:.2%})")
print(f"  Target class: {advice['target_class']}")
print("\nSuggested changes to reduce risk:")
for suggestion in advice.get("suggestions", []):
    print(f"  - Change {suggestion['feature']} from "
          f"{suggestion['original_value']:.2f} to "
          f"{suggestion['suggested_value']:.2f} "
          f"({suggestion['change_percent']:+.0f}%)")
    print(f"    New probability: {suggestion['new_probability']:.2%}")
```

**Example Output:**
```
Counterfactual Analysis:
  Current class: 1 (probability: 78.3%)
  Target class: 0

Suggested changes to reduce risk:
  - Change vendor_reliability_score from 0.62 to 0.78 (+25%)
    New probability: 42.1%
  - Change lead_time_days from 21.00 to 15.75 (-25%)
    New probability: 45.8%
```

### Phase 5: Buyer-Facing UI Integration

```python
# API endpoint for the buyer dashboard
@app.get("/orders/{order_id}/risk-explanation")
def get_order_explanation(order_id: str):
    """Get detailed risk explanation for buyer UI."""
    
    # Fetch order features
    order = get_order_by_id(order_id)
    features = extract_features(order)
    
    # Get prediction
    prediction = delay_risk_model.predict(features)
    
    # Get SHAP explanation
    shap_explanation = explainer.explain(features, prediction)
    
    # Get counterfactual advice if high risk
    counterfactual = None
    if prediction["delay_risk"] == 1:
        counterfactual = get_counterfactual_advice(features, prediction)
    
    return {
        "order_id": order_id,
        "risk_level": "HIGH" if prediction["delay_risk"] == 1 else "LOW",
        "risk_score": prediction["risk_score"],
        "confidence": prediction["confidence"],
        "summary": shap_explanation.summary(top_n=3),
        "top_risk_factors": [
            {
                "factor": f.feature_name,
                "value": f.feature_value,
                "impact": f.contribution,
                "direction": f.direction,
            }
            for f in shap_explanation.top_positive_features[:3]
        ],
        "protective_factors": [
            {
                "factor": f.feature_name,
                "value": f.feature_value,
                "impact": f.contribution,
                "direction": f.direction,
            }
            for f in shap_explanation.top_negative_features[:3]
        ],
        "recommendations": counterfactual.get("suggestions", []) if counterfactual else [],
    }
```

### Phase 6: Batch Explanations for Reporting

```python
def generate_daily_risk_report(orders: list[dict]):
    """Generate explained risk report for management."""
    
    high_risk_orders = []
    
    for order in orders:
        features = extract_features(order)
        prediction = delay_risk_model.predict(features)
        
        if prediction["delay_risk"] == 1:
            explanation = explainer.explain(features, prediction)
            
            high_risk_orders.append({
                "order_id": order["id"],
                "supplier": order["supplier_name"],
                "value": order["total_value"],
                "risk_score": prediction["risk_score"],
                "primary_risk_factor": explanation.top_positive_features[0].feature_name,
                "explanation": explanation.summary(top_n=2),
            })
    
    # Sort by risk score
    high_risk_orders.sort(key=lambda x: x["risk_score"], reverse=True)
    
    return {
        "date": datetime.now().isoformat(),
        "total_orders": len(orders),
        "high_risk_count": len(high_risk_orders),
        "high_risk_orders": high_risk_orders,
    }
```

---

## Results

### Buyer Adoption Metrics

| Metric | Before | After |
|--------|--------|-------|
| Orders with intervention | 20% of flagged | 78% of flagged |
| Buyer trust score | 2.1/5 | 4.3/5 |
| Time to intervention | 4.2 hours | 1.8 hours |
| Manual override rate | 45% | 12% |

### Business Impact

| Metric | Before | After |
|--------|--------|-------|
| Late deliveries | Baseline | -31% |
| Expedited shipping | Baseline | -28% |
| Supplier escalations | Baseline | +45% (proactive) |
| Customer satisfaction | 72 | 84 |

### Sample Buyer Feedback (Post-Implementation)

> "Now I can tell suppliers exactly why we're concerned. Last week I called Acme Corp and said 'Your reliability score dropped from 0.85 to 0.62 over the past quarter.' They appreciated the specific feedback."

> "The counterfactual feature is brilliant. It told me that if we could get 5 extra buffer days, the risk would drop below threshold. I negotiated an earlier ship date with the supplier."

---

## Key Learnings

1. **Accuracy alone doesn't drive adoption** — Users need to understand and trust predictions
2. **Explanations enable action** — "High risk" is useless without "because X, Y, Z"
3. **Counterfactuals empower users** — Showing what would change the outcome enables intervention
4. **Context matters** — Explanations must map to user mental models and vocabulary
5. **Audit trails satisfy stakeholders** — Management needs justification for resource allocation

---

## Integration Patterns

### Email Alerts with Explanations

```python
def send_risk_alert_email(order: dict, explanation: ExplanationResult):
    """Send explained risk alert to buyer."""
    
    subject = f"⚠️ High Risk Order: {order['id']} - {order['supplier_name']}"
    
    body = f"""
    Order {order['id']} has been flagged as HIGH RISK.
    
    Risk Score: {explanation.probability:.1%}
    
    {explanation.summary(top_n=3)}
    
    Recommended Actions:
    1. Contact supplier to confirm ship date
    2. Identify alternative sources
    3. Update customer on potential delay
    
    View full details: https://app.example.com/orders/{order['id']}
    """
    
    send_email(to=order["buyer_email"], subject=subject, body=body)
```

### Slack Integration

```python
def post_risk_to_slack(order: dict, explanation: ExplanationResult):
    """Post explained risk to Slack channel."""
    
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"⚠️ High Risk: {order['id']}"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Supplier:* {order['supplier_name']}"},
                {"type": "mrkdwn", "text": f"*Value:* ${order['total_value']:,.0f}"},
                {"type": "mrkdwn", "text": f"*Risk Score:* {explanation.probability:.1%}"},
                {"type": "mrkdwn", "text": f"*Due Date:* {order['due_date']}"},
            ]
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Why flagged:*\n{explanation.summary(top_n=2)}"}
        },
    ]
    
    post_to_slack(channel="#procurement-alerts", blocks=blocks)
```

---

## Related Modules

- `app/explainers/base.py` — Base explanation classes
- `app/explainers/shap_explainer.py` — SHAP-based explanations
- `app/explainers/lime_explainer.py` — LIME explanations with counterfactuals

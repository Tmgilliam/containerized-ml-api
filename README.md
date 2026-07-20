# Containerized ML API

**Owner:** Dr. Tatianna Gilliam | Cloud & AI Architect (AZ-305 | AI-102 | AZ-104)

> **Public repo:** [github.com/Tmgilliam/containerized-ml-api](https://github.com/Tmgilliam/containerized-ml-api)  
> **Live API:** [containerized-ml-api-787251273541.us-central1.run.app](https://containerized-ml-api-787251273541.us-central1.run.app/health)

---

## The Business Problem

ML models that work locally break in production because runtime environments differ — Python versions, compiled dependencies, and OS libraries drift between the data scientist's laptop and the server. For ERP-adjacent AI systems, where incorrect delay-risk predictions trigger bad procurement decisions and broken customer promise dates, that invisible failure is operational risk, not a notebook inconvenience.

You can achieve 0.87 AUC in training and still lose trust in production because nothing crashed — predictions just shifted.

## The Solution

A **containerized inference API** that enforces identical runtime from development through production. The same Docker image built and tested in CI is what runs on Cloud Run. If the pipeline smoke-tests `/health`, production runs the same bytes.

| Capability | Implementation |
|------------|----------------|
| Delay risk scoring | scikit-learn GradientBoostingClassifier |
| Validated API contract | FastAPI + Pydantic |
| Reproducible runtime | Docker (`python:3.11-slim`) |
| Serverless hosting | Google Cloud Run (scale 0–3) |
| Infrastructure as code | Terraform (GCP, Azure, AWS) |
| CI/CD with rollback | GitHub Actions (commit SHA tags) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (creates model/model.pkl)
python model/train.py

# Run API locally
uvicorn app.main:app --reload --port 8080

# Health check
curl http://localhost:8080/health

# Score an order
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_qty": 250,
    "lead_time_days": 12.0,
    "vendor_reliability_score": 0.82,
    "days_until_due": 8.0,
    "historical_delay_rate": 0.15,
    "inventory_buffer_days": 4.0
  }'
```

## Docker

```bash
docker build -t containerized-ml-api .
docker run -p 8080:8080 containerized-ml-api
```

## Project Structure

```
containerized-ml-api/
├── app/                          # FastAPI application
│   ├── main.py                   # API endpoints
│   ├── model.py                  # Model loading/inference
│   ├── schemas.py                # Pydantic schemas
│   ├── experiments/              # A/B testing infrastructure
│   ├── explainers/               # SHAP/LIME explainability
│   ├── models/                   # Multi-model serving
│   └── shadow/                   # Shadow mode deployment
├── batch/                        # Batch prediction pipeline
│   ├── pipeline.py               # Orchestration
│   └── io/                       # BigQuery/GCS readers/writers
├── features/                     # Feature store
│   ├── registry.py               # Feature definitions
│   ├── validation.py             # Schema enforcement
│   ├── compute/                  # Transformations
│   └── serving/                  # Online/offline retrieval
├── gateway/                      # API Gateway
│   ├── auth/                     # API keys, JWT validation
│   ├── rate_limiter.py           # Token bucket rate limiting
│   └── usage_tracker.py          # Usage analytics
├── monitoring/                   # Observability
│   ├── drift/                    # PSI, KS drift detection
│   ├── alerts/                   # Slack, PagerDuty notifications
│   └── dashboards/               # Grafana configs
├── retraining/                   # Feedback loop
│   ├── collectors/               # Outcome ingestion
│   ├── triggers/                 # Automated retrain triggers
│   └── pipelines/                # Training pipeline
├── terraform/                    # Infrastructure as Code
│   ├── gcp/                      # Google Cloud Run
│   ├── azure/                    # Azure Container Apps
│   └── aws/                      # AWS ECS Fargate
├── model/                        # Training script + artifacts
├── tests/                        # Test suite
├── docs/                         # Architecture docs
├── portfolio/                    # Interview materials
├── .github/workflows/            # CI/CD pipelines
├── Dockerfile
└── requirements.txt
```

## Expansion Modules

### 1. Model Drift Detection (`monitoring/drift/`)

Statistical tests (PSI, KS, Jensen-Shannon) to detect when production data diverges from training:

```python
from monitoring.drift import DriftDetector

detector = DriftDetector(reference_data, model_version="1.0")
result = detector.analyze(current_data)
if result.alert_triggered:
    print(f"Drift detected: {result.overall_severity}")
```

### 2. A/B Testing (`app/experiments/`)

Traffic splitting and statistical analysis for model comparison:

```python
from app.experiments import ExperimentRouter, Experiment, ModelVariant

router = ExperimentRouter(default_model=control_variant)
router.register_experiment(Experiment(
    name="new-model-test",
    variants=[control_variant, treatment_variant],
    start_time=datetime.now(timezone.utc),
))
result, variant, exp = router.route(features, user_id="user123")
```

### 3. Batch Prediction (`batch/`)

Process bulk ERP data for scheduled scoring:

```python
from batch import BatchPipeline, LocalReader, LocalWriter

pipeline = BatchPipeline(predict_fn=model.predict)
result = pipeline.run(
    reader=LocalReader("orders.csv"),
    writer=LocalWriter("predictions.jsonl"),
)
```

### 4. Feature Store (`features/`)

Centralized feature management with online/offline serving:

```python
from features import FeatureRegistry, OnlineFeatureStore

registry = create_erp_delay_risk_registry()
store = OnlineFeatureStore(registry)
store.set_features("erp_delay_risk", "order_123", features)
features = store.get_online_features("erp_delay_risk", "order_123")
```

### 5. Multi-Model Serving (`app/models/`)

Serve multiple models from a single API:

```python
from app.models import ModelRegistry, ModelRouter

registry = ModelRegistry()
registry.register("delay_risk", "v2", ...)
registry.register("demand_forecast", "v1", ...)

router = ModelRouter(registry)
result = router.predict(features, model_name="delay_risk")
```

### 6. Explainability (`app/explainers/`)

SHAP and LIME explanations for predictions:

```python
from app.explainers import SHAPExplainer

explainer = SHAPExplainer(model, feature_names)
explanation = explainer.explain(features, prediction)
print(explanation.summary())
```

### 7. Feedback Loop (`retraining/`)

Capture outcomes and trigger retraining:

```python
from retraining import OutcomeIngester, LabelMatcher, RetrainTrigger

ingester = OutcomeIngester()
ingester.ingest(entity_id="order_123", outcome=True)

matcher = LabelMatcher()
training_data = matcher.generate_training_data()

trigger = RetrainTrigger()
should_retrain, results = trigger.should_retrain(
    drift_result=drift_result,
    accuracy_report=accuracy_report,
)
```

### 8. API Gateway (`gateway/`)

Authentication, rate limiting, and usage tracking:

```python
from gateway import APIKeyAuth, RateLimiter, UsageTracker

auth = APIKeyAuth()
raw_key, api_key = auth.generate_key("client-app", expires_in_days=90)

limiter = RateLimiter()
result = limiter.check(client_id)
if not result.allowed:
    return 429, result.to_headers()

tracker = UsageTracker()
tracker.record(client_id, endpoint, method, status_code, latency_ms)
```

### 9. Multi-Cloud (`terraform/`)

Deploy to GCP, Azure, or AWS with identical patterns:

```bash
# GCP Cloud Run
cd terraform/gcp && terraform apply

# Azure Container Apps
cd terraform/azure && terraform apply

# AWS ECS Fargate
cd terraform/aws && terraform apply
```

### 10. Shadow Mode (`app/shadow/`)

Validate new models on production traffic:

```python
from app.shadow import ShadowRunner, ShadowComparator

runner = ShadowRunner(production_model, shadow_model)
result, shadow = runner.predict(features)

comparator = ShadowComparator()
is_ready, issues = comparator.is_shadow_ready_for_promotion()
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model version, uptime, last prediction, environment |
| `POST` | `/predict` | ERP features → delay risk + confidence + model version |

## Documentation

- [Architecture](docs/architecture.md) — design rationale, multi-cloud equivalents, canary deploys
- [Decision Records](docs/decisions.md) — ADRs for container, Cloud Run, secrets, logging
- [Case Study](portfolio/case-study.md) — interview-ready narrative
- [Talk Track](portfolio/interview-talk-track.md) — 60s, 3min, and deep-dive versions

## Related Work

This project is the **deployment discipline layer** from the [ERP AI Delay Risk](https://github.com/Tmgilliam/erp-ai-delay-risk) portfolio system — same procurement delay risk domain, focused on making ML trustworthy in production.

## Certifications

AZ-305 (Azure Solutions Architect) | AI-102 (Azure AI Engineer) | AZ-104 (Azure Administrator)

# Containerized ML API

**Owner:** Dr. Tatianna Gilliam | Cloud & AI Architect (AZ-305 | AI-102 | AZ-104)

> **Public repo:** [github.com/Tmgilliam/containerized-ml-api](https://github.com/Tmgilliam/containerized-ml-api)

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
| Infrastructure as code | Terraform |
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
├── app/                  # FastAPI application
├── model/                # Training script + model.pkl artifact
├── terraform/            # Cloud Run IaC
├── .github/workflows/    # CI and deploy pipelines
├── docs/                 # Architecture and ADRs
├── portfolio/            # Case study, talk tracks, resume bullets
├── Dockerfile
└── requirements.txt
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

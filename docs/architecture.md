# Architecture — Containerized ML API

**Owner:** Dr. Tatianna Gilliam | Cloud & AI Architect (AZ-305 | AI-102 | AZ-104)

---

## Business Problem

ML models that score correctly in a Jupyter notebook fail silently in production. The training environment uses Python 3.10 on Windows; the server runs Python 3.11 on Linux with different compiled dependencies. The model file loads, but predictions drift — or the service crashes on startup because `libgomp` is missing.

For ERP-adjacent AI — delay risk scoring that drives procurement and customer promise dates — that failure mode is not a data science inconvenience. It is operational risk. A buyer who trusts a broken risk score reorders from an unreliable vendor or misses a safety stock trigger.

**Docker solves this by freezing the runtime.** The same image built in CI is what runs in Cloud Run. If it passes smoke tests in the pipeline, it runs identically in production.

---

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ train.py    │────▶│ model.pkl    │────▶│ FastAPI /predict│
│ (local/CI)  │     │ + metadata   │     │ + /health       │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                     ┌──────────────┐              │
                     │ Dockerfile   │◀─────────────┘
                     │ python:3.11  │
                     └──────┬───────┘
                            │
              ┌─────────────▼─────────────┐
              │ GitHub Actions            │
              │ build → push → deploy     │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ Google Cloud Run          │
              │ scale 0–3, 512Mi, 30s     │
              └───────────────────────────┘
```

---

## Container vs Bare VM

| Dimension | Bare VM | Docker Container |
|-----------|---------|------------------|
| Dependency parity | Manual; drift over time | Locked in image layers |
| Rollback | Reinstall packages; hope | Redeploy previous image tag |
| Cold start | N/A (always on) | Cloud Run scale-to-zero |
| Cost at low traffic | Pay 24/7 | Pay per request |
| ML inference fit | Over-provisioned | Right-sized per service |

**Why Docker wins for ML inference:** The artifact is not just `model.pkl` — it is `model.pkl + Python runtime + scikit-learn version + system libraries`. Containerizing the full stack makes the deploy unit testable and immutable.

---

## FastAPI vs Flask

| Capability | Flask | FastAPI |
|------------|-------|---------|
| Request validation | Manual or extensions | Pydantic schemas, automatic 422 |
| OpenAPI docs | Optional add-on | Built-in `/docs` |
| Async I/O | Possible, not native | First-class async |
| Type safety | Minimal | Full typing + IDE support |

For an inference API consumed by ERP integrations and monitored by platform teams, **schema validation and auto-generated OpenAPI** reduce integration defects. FastAPI rejects malformed ERP payloads before they reach the model.

---

## Cloud Run vs GKE

| Factor | Cloud Run | GKE |
|--------|-----------|-----|
| Ops overhead | Minimal | Cluster management |
| Scale-to-zero | Native | Requires KEDA/HPA tuning |
| Traffic splitting | Built-in revisions | Istio/ingress complexity |
| Fit for this workload | Single-stateless API | Overkill at this scale |

**Why Cloud Run:** This service handles single-record ERP delay scoring with sub-second latency. No GPU, no persistent volume, no sidecar mesh. Cloud Run delivers production patterns (IAM, secrets, revisions) without cluster operations — appropriate for portfolio and small-to-medium production workloads.

---

## Secret Manager vs Hardcoded Keys

Hardcoded API keys and config in environment files leak through git history, screenshots, and forked repos. For any service with a public demo endpoint, **Secret Manager is the only acceptable choice** for sensitive configuration:

- `MODEL_VERSION` and future API keys reference Secret Manager in Terraform
- Cloud Run mounts secrets at runtime — not baked into the image
- Rotation updates the secret; redeploy picks up `latest` or pinned version

Images are immutable and often public in registries. Secrets never belong in layers.

---

## Commit SHA Image Tagging

Every deploy tags the Docker image with the Git commit SHA (`:${{ github.sha }}`).

| Benefit | Explanation |
|---------|-------------|
| Traceability | Production image maps 1:1 to code revision |
| Rollback | `gcloud run services update-traffic --to-revisions=PREVIOUS=100` |
| Audit | Incident response starts with exact commit, not "latest" |
| Reproducibility | Rebuild any historical deploy from git tag |

**Never use `:latest` in production.** It destroys rollback capability.

---

## Canary Deployment (Cloud Run Traffic Splitting)

Cloud Run supports revision-based traffic splitting without a service mesh:

```bash
gcloud run services update-traffic containerized-ml-api \
  --to-revisions=NEW_REVISION=10,LATEST=90
```

Monitor `/health` latency, error rate, and prediction distribution. Promote to 100% or roll back to previous revision in seconds. This is the production-grade deploy pattern for ML APIs where model version changes can shift score distributions.

---

## Multi-Cloud Equivalents

### Google Cloud (this project)

| Component | Service |
|-----------|---------|
| Container hosting | Cloud Run |
| Registry | Artifact Registry |
| Secrets | Secret Manager |
| CI/CD auth | Workload Identity Federation |
| IaC | Terraform (`google` provider) |

### Azure Equivalent

| GCP | Azure |
|-----|-------|
| Cloud Run | **Azure Container Apps** (scale-to-zero, revision management) |
| Artifact Registry | **Azure Container Registry (ACR)** |
| Secret Manager | **Azure Key Vault** |
| Workload Identity Federation | **Managed Identity** + GitHub OIDC federated credential |
| Terraform | `azurerm` provider for Container Apps + ACR |

Deploy pattern: GitHub Actions authenticates via OIDC → push to ACR → `az containerapp update` with new image digest. Secrets referenced as Key Vault secret URIs in Container Apps environment configuration.

### AWS Equivalent

| GCP | AWS |
|-----|-----|
| Cloud Run | **ECS Fargate** (serverless containers) or App Runner |
| Artifact Registry | **Amazon ECR** |
| Secret Manager | **AWS Secrets Manager** |
| Workload Identity Federation | **IAM OIDC identity provider** for GitHub Actions |
| Terraform | `aws` provider for ECS task definition + service |

Fargate task definitions pin CPU/memory like Cloud Run limits. ECS service deployments support blue/green via CodeDeploy for canary patterns equivalent to Cloud Run traffic splitting.

---

## Observability

Every request emits structured JSON logs with:

- `request_id` — correlation across load balancer and application logs
- `model_version` — which model artifact served the prediction
- `input_schema` — endpoint/schema identifier
- `prediction` — scored output (for audit, not raw PII)
- `latency_ms` — SLA monitoring input

`/health` exposes uptime, model version, last prediction timestamp, and environment — the minimum surface for deployment smoke tests and uptime checks.

---

## Related Portfolio Work

This deployment pattern is the inference layer used at enterprise scale in the **[ERP AI Delay Risk](https://github.com/Tmgilliam/erp-ai-delay-risk)** project — same business domain (procurement delay risk), extended with drift monitoring, Azure migration architecture, and executive dashboard integration.

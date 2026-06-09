# Case Study — Containerized ML API

**Dr. Tatianna Gilliam, DBA** | Cloud & AI Architect | AZ-305 | AI-102 | AZ-104

---

## The Problem

A procurement team trusts an ML model that predicts vendor delay risk. Data science validates accuracy at 0.87 AUC on a holdout set. Two weeks after "deployment," buyers report the model feels wrong — but logs show no errors.

Root cause: the production server runs a different scikit-learn patch version than training. Predictions shifted 3–8% on high-risk orders. No crash. No alert. Just wrong operational decisions.

This is the invisible failure mode in ERP-adjacent AI. Incorrect delay risk scores trigger bad reorder timing, missed safety stock alerts, and broken customer promise dates.

---

## The Solution

Build a **containerized inference API** that enforces identical runtime from laptop → CI → Cloud Run:

1. **Train** a GradientBoostingClassifier on mock ERP features with embedded metadata (`version`, `f1_score`, `auc`, `features`)
2. **Serve** via FastAPI with Pydantic validation — reject bad ERP payloads before inference
3. **Package** in Docker (`python:3.11-slim`, pinned deps, model artifact)
4. **Deploy** to Cloud Run with Terraform (512Mi, scale 0–3, Secret Manager)
5. **Pipeline** via GitHub Actions: lint → test → build → push → deploy → smoke test `/health`

If CI greenlights the image, production runs the same bytes.

---

## Architecture Highlights

| Layer | Choice | Why |
|-------|--------|-----|
| ML | scikit-learn GradientBoosting | Interpretable, no GPU, ERP tabular data |
| API | FastAPI + Pydantic | Schema validation, OpenAPI, async-ready |
| Container | Docker multi-layer | Dependency cache, reproducible builds |
| Hosting | Cloud Run | Scale-to-zero, revision rollback, no cluster ops |
| Secrets | Secret Manager | No keys in images or git |
| IaC | Terraform | Repeatable, auditable infrastructure |
| CI/CD | GitHub Actions + WIF | Keyless GCP auth, SHA-tagged images |

---

## Business Outcome

- **Deployment discipline:** Model accuracy is necessary but not sufficient — runtime parity is the production gate
- **Operational traceability:** Structured JSON logs tie every prediction to `request_id` and `model_version`
- **Rollback in seconds:** Commit SHA tags enable instant revert on score distribution drift
- **Multi-cloud portability:** Documented Azure (Container Apps + ACR + Key Vault) and AWS (Fargate + ECR + Secrets Manager) equivalents

---

## Connection to ERP AI Delay Risk

This project is the **deployment pattern** extracted from the full [ERP AI Delay Risk](https://github.com/Tmgilliam/erp-ai-delay-risk) system — same domain (procurement delay scoring), focused on the container + CI/CD layer that enterprise ML teams audit first.

---

## What I Would Say in an Interview

*"I didn't just train a model — I built the production envelope that makes the model trustworthy. Docker for runtime parity, FastAPI for contract enforcement, Cloud Run for cost-aware scaling, and CI that fails the deploy if `/health` doesn't pass. That's the difference between a notebook and an ERP system that can act on predictions."*

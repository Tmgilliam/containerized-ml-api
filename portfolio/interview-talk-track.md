# Interview Talk Track — Containerized ML API

**Dr. Tatianna Gilliam, DBA** | AZ-305 | AI-102 | AZ-104  
Three versions for different interview stages.

---

## Version 1 — 60 Seconds (Recruiter Screen)

*"I built a containerized ML inference API that solves a problem every enterprise hits: models that work locally break in production because the runtime environment differs. For ERP delay risk scoring — where a wrong prediction triggers bad procurement decisions — that invisible drift is operational risk, not a data science detail.*

*The solution packages scikit-learn inference in FastAPI, Docker, and Cloud Run with Terraform and GitHub Actions CI/CD. Same image from CI to production. Smoke-tested deploy. Commit SHA rollback. Happy to go deeper on architecture, multi-cloud equivalents, or how this connects to my full ERP AI project."*

---

## Version 2 — 3 Minutes (Hiring Manager / Solutions Architect)

### Opening — Business Problem (30 seconds)

*"Procurement teams increasingly rely on ML to flag vendor delay risk before it hits customer promise dates. The model validates fine in Jupyter. Two weeks after deployment, buyers say scores feel wrong — but nothing crashed. Root cause: different Python and scikit-learn versions between training and production shifted predictions 3–8% on high-risk orders. No alert fired because the service was 'healthy.'*

*That's the failure mode I designed against."*

### Architecture Walkthrough (90 seconds)

*"Four layers:*

*1. **Training with embedded metadata** — GradientBoostingClassifier on ERP features: order quantity, lead time, vendor reliability, days until due, historical delay rate, inventory buffer. Model artifact includes version, F1, AUC, and feature list — not just weights.*

*2. **FastAPI inference with contract enforcement** — Pydantic schemas reject malformed ERP payloads before they reach the model. Structured JSON logging on every request: request ID, model version, latency, prediction. `/health` exposes uptime and last prediction for smoke tests.*

*3. **Docker as the deploy unit** — Not model.pkl alone — the full stack: Python 3.11-slim, pinned dependencies, app code, artifact. Layer caching in Dockerfile for fast CI builds.*

*4. **Cloud Run + Terraform + GitHub Actions** — Scale 0–3, 512Mi, 30s timeout. Secrets from Secret Manager, not image layers. Deploy pipeline tags images with commit SHA, pushes to Artifact Registry, deploys, curls `/health` — fails the release if smoke test fails."*

### Trade-offs (30 seconds)

*"Cloud Run over GKE because this is single-record stateless inference — no GPU, no mesh. Scale-to-zero accepts a 2–5 second cold start in exchange for near-zero idle cost. FastAPI over Flask for native schema validation and OpenAPI — ERP integrations need contracts, not guesswork.*

*I documented Azure equivalents — Container Apps, ACR, Key Vault, Managed Identity — and AWS — Fargate, ECR, Secrets Manager — because platform choice shouldn't change the deployment discipline."*

### Close (30 seconds)

*"This is the inference layer pattern from my ERP AI Delay Risk project, extracted to prove deployment engineering independently of model training. The hard part isn't AUC — it's making sure production runs the same bytes CI tested."*

---

## Version 3 — "Why Does This Matter More Than Just Training a Model?"

*This is the deployment discipline question that separates ML practitioners from ML engineers.*

### The Question

*"You have a model with 0.87 AUC. Why do you need Docker, Terraform, and CI/CD? Can't you just deploy the pickle file?"*

### The Answer (2–3 minutes)

*"AUC measures offline performance on a fixed dataset. Production failure modes are different:*

*1. **Runtime drift** — scikit-learn 1.4 vs 1.5 changes tree traversal. Same pickle, different probabilities. Docker pins the entire runtime, not just the model file.*

*2. **Integration drift** — ERP sends `lead_time` as a string, or omits `vendor_reliability_score`. Without Pydantic validation, you get NaN predictions or silent defaults. FastAPI returns 422 with field-level errors — the integration team fixes it before production traffic.*

*3. **Operational blindness** — When a buyer disputes a score six weeks later, you need `request_id`, `model_version`, and `latency_ms` in structured logs. Training metrics don't help you answer 'what model served this order on March 3rd?'*

*4. **Rollback capability** — If a new model version shifts score distribution, `:latest` tags leave you guessing which image to revert. Commit SHA tags map production to exact git revision. Cloud Run traffic splitting lets you canary 10% before full promotion.*

*5. **Secret hygiene** — API keys in Dockerfile layers end up in public registries. Secret Manager injects config at runtime — the image stays shareable, the secrets stay out of git.*

*Training proves the model can learn. Containerization, validation, observability, and CI/CD prove the system can be operated — audited, rolled back, and trusted by people who make procurement decisions based on the output.*

*That's why I lead with deployment discipline, not AUC. In ERP operations, I managed 98% inventory accuracy because the process was trustworthy, not because we had good spreadsheets. ML in production needs the same standard."*

---

## Objection Handling

**"Why not SageMaker / Vertex AI endpoints?"**  
*"Managed endpoints are right at scale with autoscaling model fleets and A/B testing infrastructure. For a focused inference API with one model and bursty ERP traffic, Cloud Run + container gives 80% of the operational pattern at 20% of the platform complexity — and the Dockerfile ports to SageMaker or Vertex if the workload grows."*

**"Why GradientBoosting over deep learning?"**  
*"Tabular ERP features, interpretable risk drivers, no GPU requirement, sub-10ms inference. The portfolio point is deployment, not architecture search."*

**"Public endpoint with allUsers invoker?"**  
*"Portfolio demo only. Enterprise deployment restricts to authenticated service accounts or API gateway — same container, different IAM binding."*

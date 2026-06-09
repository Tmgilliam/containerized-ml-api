# Resume Bullets — Containerized ML API

**Dr. Tatianna Gilliam, DBA** | Use as written or adapt to job description keywords.

---

## Solutions Architect / Cloud Architect (2)

- Architected containerized ML inference platform on Google Cloud Run with Terraform IaC, Artifact Registry, and Secret Manager — enforcing runtime parity from CI to production for ERP delay risk scoring where prediction drift creates procurement operational risk.

- Designed multi-cloud deployment reference architecture documenting Azure (Container Apps, ACR, Key Vault, Managed Identity) and AWS (ECS Fargate, ECR, Secrets Manager) equivalents — enabling platform-agnostic ML serving without re-architecting the inference contract.

---

## ML Engineer / DevOps Engineer (2)

- Built production FastAPI inference service with Pydantic schema validation, structured JSON logging (request ID, model version, latency), and startup model loading — rejecting malformed ERP payloads before inference and enabling end-to-end prediction traceability.

- Implemented GitHub Actions CI/CD pipeline with Workload Identity Federation: lint, unit test, Docker build verification on PR; commit SHA-tagged image push, Cloud Run deploy, and automated `/health` smoke test on main — failing releases that don't pass operational gates.

---

## ERP AI Connection (1)

- Extracted and hardened the containerized deployment pattern from enterprise ERP AI Delay Risk system — same procurement delay scoring domain, focused on the production envelope (Docker + Cloud Run + Terraform + CI/CD) that makes ML predictions auditable and rollback-capable at operational scale.

# Architecture Decision Records

**Project:** Containerized ML API  
**Owner:** Dr. Tatianna Gilliam

---

## ADR-001: Containerize the Full Inference Stack, Not Just the Model

**Context:** Saving `model.pkl` to blob storage and running `pip install scikit-learn` on a VM reproduces the classic "works on my machine" failure. ERP operations cannot tolerate silent prediction drift.

**Decision:** Docker image bundles Python 3.11-slim, pinned dependencies, application code, and model artifact.

**Consequences:** Slightly larger deploy unit; dramatically higher confidence in runtime parity. CI validates the image builds before merge.

---

## ADR-002: Load Model at Startup, Not Per Request

**Context:** Loading `model.pkl` on every `/predict` adds 100–500ms latency and disk I/O under load.

**Decision:** Load once in FastAPI lifespan; serve from in-memory classifier.

**Consequences:** Container memory includes model footprint (~512Mi limit sufficient). Startup probe must wait for load completion.

---

## ADR-003: Pydantic Strict Validation on ERP Features

**Context:** ERP integrations send malformed payloads — missing fields, wrong types, negative quantities.

**Decision:** `PredictRequest` with `extra="forbid"`; explicit 422 responses with field-level errors.

**Consequences:** Integration teams get immediate feedback; model never sees garbage input.

---

## ADR-004: Cloud Run Scale-to-Zero for Portfolio + Production

**Context:** Inference traffic is bursty (batch scoring windows, dashboard refreshes). Always-on VMs waste budget.

**Decision:** Cloud Run min=0, max=3, 512Mi, 30s timeout.

**Consequences:** Cold start on first request after idle (~2–5s). Acceptable for delay risk scoring; not for sub-100ms HFT.

---

## ADR-005: Commit SHA Tags, Never `:latest`

**Context:** Production incidents require instant rollback to known-good artifact.

**Decision:** GitHub Actions tags images with `${{ github.sha }}`; Terraform/gcloud deploy references explicit tag.

**Consequences:** Registry grows with each deploy; retention policy required. Rollback is one command.

---

## ADR-006: Secret Manager for Runtime Configuration

**Context:** Model version and future API keys must not appear in Dockerfile layers or git.

**Decision:** Terraform references Secret Manager for `MODEL_VERSION`; `.env.example` documents pattern only.

**Consequences:** Requires GCP Secret Manager setup before first Terraform apply. Enterprise-appropriate security posture.

---

## ADR-007: Structured JSON Logging with Request ID

**Context:** Debugging production prediction disputes requires tracing a single ERP order's score through logs.

**Decision:** Every request logs JSON with `request_id`, `model_version`, `latency_ms`, `prediction`.

**Consequences:** Log volume increases; Cloud Logging ingestion cost negligible at this scale. Traceability is non-negotiable for ERP-adjacent AI.

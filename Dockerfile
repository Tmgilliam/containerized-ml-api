# Containerized ML API — ERP Delay Risk Inference
#
# Business problem: ML models that work on a laptop break in production because
# Python versions, dependencies, and OS libraries differ between environments.
# For ERP-adjacent AI — where a wrong delay-risk score triggers bad procurement
# decisions — that invisible drift is operational risk, not a data science detail.
#
# Solution: identical runtime from train → build → deploy via Docker, with
# FastAPI for validated inference and Cloud Run for scale-to-zero serving.

# Pin base image for reproducible builds across dev laptops and CI runners.
FROM python:3.11-slim

# Application root inside the container.
WORKDIR /app

# Install dependencies first — this layer caches until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-trained model artifact.
COPY app/ ./app/
COPY model/model.pkl ./model/

# Cloud Run expects the service to listen on PORT (default 8080).
EXPOSE 8080

# Start ASGI server bound to all interfaces for container orchestration.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

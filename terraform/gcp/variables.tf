variable "project_id" {
  description = "GCP project ID for Cloud Run and Artifact Registry."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run deployment."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "containerized-ml-api"
}

variable "image_uri" {
  description = "Full Artifact Registry image URI including tag (commit SHA recommended)."
  type        = string
}

variable "environment" {
  description = "Runtime environment label exposed via /health."
  type        = string
  default     = "production"
}

variable "model_version_secret" {
  description = "Secret Manager resource name for MODEL_VERSION."
  type        = string
}

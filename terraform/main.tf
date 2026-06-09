# Cloud Run v2 service — serverless container hosting for ML inference API.
# Scale-to-zero keeps portfolio/demo cost near zero while preserving production patterns.
resource "google_cloud_run_v2_service" "ml_api" {
  name     = var.service_name
  location = var.region
  project  = var.project_id

  template {
    # Request timeout — inference should complete well under 30s for single-record scoring.
    timeout = "30s"

    containers {
      image = var.image_uri

      # Right-sized for scikit-learn single-record inference; revisit if batch scoring added.
      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }

      ports {
        container_port = 8080
      }

      # Environment variables — non-secret config inline, secrets from Secret Manager.
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name = "MODEL_VERSION"
        value_source {
          secret_key_ref {
            secret  = var.model_version_secret
            version = "latest"
          }
        }
      }
    }

    # Scale to zero when idle; cap at 3 instances for demo/portfolio traffic patterns.
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  # Disable default IAM invoker binding; explicit binding below for clarity.
  ingress = "INGRESS_TRAFFIC_ALL"
}

# Public invoker for portfolio demo — restrict to authenticated callers in enterprise deployments.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.ml_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

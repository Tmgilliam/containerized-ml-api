output "service_url" {
  description = "HTTPS URL of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.ml_api.uri
}

output "service_name" {
  description = "Cloud Run service name for gcloud deploy commands."
  value       = google_cloud_run_v2_service.ml_api.name
}

output "latest_revision" {
  description = "Latest deployed revision — use for rollback targeting."
  value       = google_cloud_run_v2_service.ml_api.latest_ready_revision
}

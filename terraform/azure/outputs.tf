output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "container_app_url" {
  description = "URL of the Container App"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "container_registry_login_server" {
  description = "Container Registry login server"
  value       = azurerm_container_registry.main.login_server
}

output "container_registry_admin_username" {
  description = "Container Registry admin username"
  value       = azurerm_container_registry.main.admin_username
  sensitive   = true
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}

output "log_analytics_workspace_id" {
  description = "Log Analytics Workspace ID"
  value       = azurerm_log_analytics_workspace.main.id
}

output "health_endpoint" {
  description = "Health check endpoint"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}/health"
}

output "predict_endpoint" {
  description = "Prediction endpoint"
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}/predict"
}

# Azure Container Apps deployment for Containerized ML API

locals {
  resource_prefix = "${var.project_name}-${var.environment}"
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${local.resource_prefix}-rg"
  location = var.location
  tags     = var.tags
}

# Log Analytics Workspace for Container Apps
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${local.resource_prefix}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = replace("${local.resource_prefix}acr", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = var.tags
}

# Key Vault for secrets
resource "azurerm_key_vault" "main" {
  name                       = "${local.resource_prefix}-kv"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false
  tags                       = var.tags

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge"
    ]
  }
}

# Store model version in Key Vault
resource "azurerm_key_vault_secret" "model_version" {
  name         = "model-version"
  value        = var.model_version
  key_vault_id = azurerm_key_vault.main.id
}

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "${local.resource_prefix}-env"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  tags                       = var.tags
}

# Container App
resource "azurerm_container_app" "api" {
  name                         = "${local.resource_prefix}-app"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Multiple"
  tags                         = var.tags

  template {
    container {
      name   = "ml-api"
      image  = var.container_image
      cpu    = var.container_cpu
      memory = var.container_memory

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }

      env {
        name        = "MODEL_VERSION"
        secret_name = "model-version"
      }

      liveness_probe {
        path             = "/health"
        port             = 8080
        transport        = "HTTP"
        initial_delay    = 10
        interval_seconds = 30
      }

      readiness_probe {
        path             = "/health"
        port             = 8080
        transport        = "HTTP"
        interval_seconds = 10
      }
    }

    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = 50
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "http"

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  secret {
    name  = "model-version"
    value = var.model_version
  }

  registry {
    server               = azurerm_container_registry.main.login_server
    username             = azurerm_container_registry.main.admin_username
    password_secret_name = "registry-password"
  }

  secret {
    name  = "registry-password"
    value = azurerm_container_registry.main.admin_password
  }
}

# Current client config for Key Vault access
data "azurerm_client_config" "current" {}

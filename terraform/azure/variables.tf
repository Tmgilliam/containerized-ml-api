variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "containerized-ml-api"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "container_image" {
  description = "Docker image for the API"
  type        = string
}

variable "container_cpu" {
  description = "CPU cores for container"
  type        = number
  default     = 0.5
}

variable "container_memory" {
  description = "Memory in GB for container"
  type        = string
  default     = "1Gi"
}

variable "min_replicas" {
  description = "Minimum number of replicas"
  type        = number
  default     = 0
}

variable "max_replicas" {
  description = "Maximum number of replicas"
  type        = number
  default     = 3
}

variable "model_version" {
  description = "Model version identifier"
  type        = string
  default     = "1.0.0"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    Project     = "containerized-ml-api"
    ManagedBy   = "terraform"
  }
}

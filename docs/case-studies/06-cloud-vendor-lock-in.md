# Case Study 6: Cloud Vendor Lock-In Threatens Continuity

## Company Profile

**Industry:** Medical device manufacturer  
**Regions:** North America, Europe, Asia-Pacific  
**Constraint:** HIPAA, GDPR, and regional data residency requirements

---

## The Problem

### Situation

A medical device manufacturer deployed their delay risk scoring API exclusively on Google Cloud Platform. The system processed sensitive supply chain data with strict compliance requirements.

### The Incident

In Q3, a major GCP region experienced a 4-hour outage during a critical production planning cycle. The impact:

- **847 orders** couldn't be risk-scored during the window
- **$2.3M** in inventory commitments made without risk visibility
- **3 high-risk orders** slipped through, causing production delays
- **Compliance audit flagged** single-cloud dependency as a risk

### Strategic Concerns

The CTO identified additional risks:

1. **Negotiating leverage** — 100% GCP dependency weakened contract negotiations
2. **Regional requirements** — EU operations required data processing within EU borders
3. **Disaster recovery** — 4-hour RTO wasn't acceptable for critical operations
4. **Cost optimization** — No ability to leverage competitive cloud pricing

---

## The Solution

### Phase 1: Multi-Cloud Infrastructure with Terraform

Refactor from single-cloud to multi-cloud Terraform modules:

#### Directory Structure

```
terraform/
├── modules/
│   └── ml-api/
│       ├── main.tf          # Cloud-agnostic resources
│       ├── variables.tf     # Common variables
│       └── outputs.tf       # Common outputs
├── gcp/
│   ├── main.tf              # GCP-specific configuration
│   ├── provider.tf
│   ├── variables.tf
│   └── outputs.tf
├── azure/
│   ├── main.tf              # Azure-specific configuration
│   ├── provider.tf
│   ├── variables.tf
│   └── outputs.tf
└── aws/
    ├── main.tf              # AWS-specific configuration
    ├── provider.tf
    ├── variables.tf
    └── outputs.tf
```

#### GCP Configuration (Primary)

```hcl
# terraform/gcp/main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "api" {
  name     = var.service_name
  location = var.region

  template {
    spec {
      containers {
        image = var.image_url
        
        ports {
          container_port = 8080
        }
        
        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }
        
        env {
          name  = "MODEL_PATH"
          value = var.model_path
        }
        
        env {
          name = "DB_CONNECTION_STRING"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.db_connection.secret_id
              key  = "latest"
            }
          }
        }
      }
      
      service_account_name = google_service_account.api.email
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = var.min_instances
        "autoscaling.knative.dev/maxScale" = var.max_instances
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_secret_manager_secret" "db_connection" {
  secret_id = "${var.service_name}-db-connection"
  
  replication {
    auto {}
  }
}
```

#### Azure Configuration (EU Region)

```hcl
# terraform/azure/main.tf
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "api" {
  name     = "${var.service_name}-rg"
  location = var.location  # "westeurope" for EU data residency
}

resource "azurerm_container_app_environment" "api" {
  name                = "${var.service_name}-env"
  location            = azurerm_resource_group.api.location
  resource_group_name = azurerm_resource_group.api.name
  
  log_analytics_workspace_id = azurerm_log_analytics_workspace.api.id
}

resource "azurerm_container_app" "api" {
  name                         = var.service_name
  container_app_environment_id = azurerm_container_app_environment.api.id
  resource_group_name          = azurerm_resource_group.api.name
  revision_mode                = "Single"

  template {
    container {
      name   = "api"
      image  = var.image_url
      cpu    = var.cpu_limit
      memory = var.memory_limit

      env {
        name  = "MODEL_PATH"
        value = var.model_path
      }

      env {
        name        = "DB_CONNECTION_STRING"
        secret_name = "db-connection"
      }
    }
    
    min_replicas = var.min_instances
    max_replicas = var.max_instances
  }

  secret {
    name  = "db-connection"
    value = var.db_connection_string
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

resource "azurerm_key_vault" "api" {
  name                = "${var.service_name}-kv"
  location            = azurerm_resource_group.api.location
  resource_group_name = azurerm_resource_group.api.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  purge_protection_enabled = true  # Required for GDPR compliance
}
```

#### AWS Configuration (Disaster Recovery)

```hcl
# terraform/aws/main.tf
provider "aws" {
  region = var.region
}

resource "aws_ecs_cluster" "api" {
  name = "${var.service_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_service" "api" {
  name            = var.service_name
  cluster         = aws_ecs_cluster.api.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.min_instances
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8080
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = var.service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu_limit
  memory                   = var.memory_limit
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "api"
      image = var.image_url
      
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "MODEL_PATH"
          value = var.model_path
        }
      ]
      
      secrets = [
        {
          name      = "DB_CONNECTION_STRING"
          valueFrom = aws_secretsmanager_secret.db_connection.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

resource "aws_appautoscaling_target" "api" {
  max_capacity       = var.max_instances
  min_capacity       = var.min_instances
  resource_id        = "service/${aws_ecs_cluster.api.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
```

### Phase 2: CI/CD for Multi-Cloud

#### GCP Deployment Workflow

```yaml
# .github/workflows/deploy-gcp.yml
name: Deploy to GCP

on:
  push:
    branches: [main]
    paths:
      - 'app/**'
      - 'model/**'
      - 'Dockerfile'
      - '.github/workflows/deploy-gcp.yml'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: delay-risk-api
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - uses: google-github-actions/setup-gcloud@v2
      
      - name: Configure Docker
        run: gcloud auth configure-docker
      
      - name: Build and Push
        run: |
          docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }} .
          docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy $SERVICE_NAME \
            --image gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }} \
            --region $REGION \
            --platform managed \
            --allow-unauthenticated
      
      - name: Health Check
        run: |
          URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
          curl -f "$URL/health" || exit 1
```

#### Azure Deployment Workflow

```yaml
# .github/workflows/deploy-azure.yml
name: Deploy to Azure

on:
  push:
    branches: [main]
    paths:
      - 'app/**'
      - 'model/**'
      - 'Dockerfile'
      - '.github/workflows/deploy-azure.yml'

env:
  AZURE_CONTAINER_REGISTRY: delayriskapiacr
  RESOURCE_GROUP: delay-risk-api-rg
  CONTAINER_APP_NAME: delay-risk-api

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build and Push to ACR
        run: |
          az acr login --name $AZURE_CONTAINER_REGISTRY
          docker build -t $AZURE_CONTAINER_REGISTRY.azurecr.io/delay-risk-api:${{ github.sha }} .
          docker push $AZURE_CONTAINER_REGISTRY.azurecr.io/delay-risk-api:${{ github.sha }}
      
      - name: Deploy to Container Apps
        run: |
          az containerapp update \
            --name $CONTAINER_APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --image $AZURE_CONTAINER_REGISTRY.azurecr.io/delay-risk-api:${{ github.sha }}
      
      - name: Health Check
        run: |
          URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
          curl -f "https://$URL/health" || exit 1
```

#### AWS Deployment Workflow

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS

on:
  push:
    branches: [main]
    paths:
      - 'app/**'
      - 'model/**'
      - 'Dockerfile'
      - '.github/workflows/deploy-aws.yml'

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: delay-risk-api
  ECS_CLUSTER: delay-risk-api-cluster
  ECS_SERVICE: delay-risk-api

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - uses: aws-actions/amazon-ecr-login@v2
        id: login-ecr
      
      - name: Build and Push
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }} .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
      
      - name: Update Task Definition
        id: task-def
        run: |
          # Get current task definition
          TASK_DEF=$(aws ecs describe-task-definition --task-definition $ECS_SERVICE)
          
          # Update image
          NEW_TASK_DEF=$(echo $TASK_DEF | jq --arg IMAGE "${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:${{ github.sha }}" \
            '.taskDefinition | .containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn) | del(.revision) | del(.status) | del(.requiresAttributes) | del(.compatibilities) | del(.registeredAt) | del(.registeredBy)')
          
          # Register new task definition
          aws ecs register-task-definition --cli-input-json "$NEW_TASK_DEF"
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service $ECS_SERVICE \
            --force-new-deployment
      
      - name: Wait for Deployment
        run: |
          aws ecs wait services-stable \
            --cluster $ECS_CLUSTER \
            --services $ECS_SERVICE
```

### Phase 3: Traffic Routing and Failover

```python
# app/routing/multi_cloud.py
from typing import Optional
import httpx
import asyncio
from dataclasses import dataclass

@dataclass
class CloudEndpoint:
    name: str
    url: str
    region: str
    priority: int
    is_healthy: bool = True
    latency_ms: float = 0.0

class MultiCloudRouter:
    """Route requests across multiple cloud endpoints with failover."""
    
    def __init__(self, endpoints: list[CloudEndpoint]):
        self.endpoints = sorted(endpoints, key=lambda e: e.priority)
        self._health_check_interval = 30  # seconds
    
    async def route_request(
        self,
        path: str,
        method: str = "POST",
        json: Optional[dict] = None,
        timeout: float = 5.0,
    ) -> dict:
        """Route request to healthiest endpoint with failover."""
        
        # Try endpoints in priority order
        for endpoint in self.endpoints:
            if not endpoint.is_healthy:
                continue
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method=method,
                        url=f"{endpoint.url}{path}",
                        json=json,
                        timeout=timeout,
                    )
                    
                    if response.status_code == 200:
                        return {
                            "data": response.json(),
                            "endpoint": endpoint.name,
                            "region": endpoint.region,
                        }
                    
            except (httpx.RequestError, httpx.TimeoutException):
                # Mark endpoint as unhealthy
                endpoint.is_healthy = False
                continue
        
        raise Exception("All cloud endpoints are unavailable")
    
    async def health_check_loop(self):
        """Continuously check endpoint health."""
        
        while True:
            for endpoint in self.endpoints:
                try:
                    async with httpx.AsyncClient() as client:
                        start = asyncio.get_event_loop().time()
                        response = await client.get(
                            f"{endpoint.url}/health",
                            timeout=5.0,
                        )
                        latency = (asyncio.get_event_loop().time() - start) * 1000
                        
                        endpoint.is_healthy = response.status_code == 200
                        endpoint.latency_ms = latency
                        
                except Exception:
                    endpoint.is_healthy = False
            
            await asyncio.sleep(self._health_check_interval)


# Configuration
router = MultiCloudRouter([
    CloudEndpoint(
        name="gcp-us",
        url="https://delay-risk-api-xxx.run.app",
        region="us-central1",
        priority=1,  # Primary
    ),
    CloudEndpoint(
        name="azure-eu",
        url="https://delay-risk-api.azurecontainerapps.io",
        region="westeurope",
        priority=2,  # Secondary / EU data residency
    ),
    CloudEndpoint(
        name="aws-us",
        url="https://delay-risk-api.us-east-1.elb.amazonaws.com",
        region="us-east-1",
        priority=3,  # Disaster recovery
    ),
])
```

### Phase 4: Region-Aware Routing

```python
# app/routing/geo_router.py
from typing import Optional

class GeoAwareRouter:
    """Route requests based on geographic requirements."""
    
    REGION_MAPPING = {
        # EU data residency - must stay in EU
        "DE": "azure-eu",
        "FR": "azure-eu",
        "IT": "azure-eu",
        "ES": "azure-eu",
        "NL": "azure-eu",
        "BE": "azure-eu",
        "AT": "azure-eu",
        "PL": "azure-eu",
        
        # US - prefer GCP
        "US": "gcp-us",
        "CA": "gcp-us",
        "MX": "gcp-us",
        
        # APAC - prefer AWS (future)
        "JP": "aws-apac",
        "AU": "aws-apac",
        "SG": "aws-apac",
    }
    
    def __init__(self, multi_cloud_router: MultiCloudRouter):
        self.router = multi_cloud_router
    
    def get_preferred_endpoint(
        self,
        country_code: Optional[str] = None,
        data_residency_required: bool = False,
    ) -> str:
        """Get preferred endpoint based on geography and compliance."""
        
        if country_code and country_code in self.REGION_MAPPING:
            preferred = self.REGION_MAPPING[country_code]
            
            # For EU data residency, MUST use EU endpoint
            if data_residency_required and country_code in ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL"]:
                return preferred
            
            # Otherwise, use preferred but allow failover
            return preferred
        
        # Default to primary
        return "gcp-us"
    
    async def route_with_geo(
        self,
        path: str,
        json: dict,
        country_code: Optional[str] = None,
        data_residency_required: bool = False,
    ) -> dict:
        """Route request with geographic awareness."""
        
        preferred = self.get_preferred_endpoint(country_code, data_residency_required)
        
        # Reorder endpoints to prefer geographic match
        endpoints = sorted(
            self.router.endpoints,
            key=lambda e: (0 if e.name == preferred else 1, e.priority)
        )
        
        # If data residency is required, filter to compliant endpoints only
        if data_residency_required:
            endpoints = [e for e in endpoints if e.name == preferred]
        
        # Route with failover
        for endpoint in endpoints:
            if not endpoint.is_healthy:
                continue
            
            try:
                result = await self.router.route_request(
                    path=path,
                    json=json,
                    timeout=5.0,
                )
                
                return {
                    **result,
                    "geo_routing": {
                        "country_code": country_code,
                        "data_residency_required": data_residency_required,
                        "preferred_endpoint": preferred,
                        "actual_endpoint": result["endpoint"],
                    }
                }
                
            except Exception:
                continue
        
        raise Exception(f"No compliant endpoints available for country={country_code}")
```

---

## Results

### Before Implementation

| Metric | Value |
|--------|-------|
| Cloud providers | 1 (GCP) |
| Disaster recovery time | 4+ hours (manual) |
| EU data residency | Non-compliant |
| Vendor negotiating leverage | Weak |
| Annual cloud spend | $480K (GCP only) |

### After Implementation

| Metric | Value |
|--------|-------|
| Cloud providers | 3 (GCP, Azure, AWS) |
| Disaster recovery time | < 30 seconds (auto-failover) |
| EU data residency | Compliant (Azure EU) |
| Vendor negotiating leverage | Strong |
| Annual cloud spend | $420K (-12%, multi-cloud pricing) |

### Availability Improvement

| Scenario | Before | After |
|----------|--------|-------|
| Single region outage | 100% downtime | 0% downtime |
| Provider-wide outage | 100% downtime | 0% downtime |
| Network partition | 100% downtime | Degraded (regional) |

---

## Key Learnings

1. **Cloud-agnostic infrastructure** — Terraform modules enable consistent deployments across providers
2. **Automated failover** — Health checks and routing eliminate manual intervention
3. **Data residency compliance** — Region-aware routing satisfies GDPR and local regulations
4. **Cost optimization** — Multi-cloud enables competitive pricing and reserved capacity negotiation
5. **Shared Dockerfiles** — Container-based deployment ensures identical behavior across clouds

---

## Deployment Architecture

```
                    ┌─────────────────┐
                    │  Global Load    │
                    │   Balancer      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  GCP Cloud    │   │ Azure         │   │ AWS ECS       │
│  Run          │   │ Container     │   │ Fargate       │
│  (Primary)    │   │ Apps (EU)     │   │ (DR)          │
│               │   │               │   │               │
│  us-central1  │   │  westeurope   │   │  us-east-1    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  GCP Secret   │   │ Azure Key     │   │ AWS Secrets   │
│  Manager      │   │ Vault         │   │ Manager       │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## Related Modules

- `terraform/gcp/` — GCP Cloud Run configuration
- `terraform/azure/` — Azure Container Apps configuration
- `terraform/aws/` — AWS ECS Fargate configuration
- `.github/workflows/deploy-gcp.yml` — GCP CI/CD
- `.github/workflows/deploy-azure.yml` — Azure CI/CD
- `.github/workflows/deploy-aws.yml` — AWS CI/CD

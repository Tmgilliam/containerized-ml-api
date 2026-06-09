# Deploy containerized-ml-api to Google Cloud Run (portfolio demo).
param(
    [string]$ProjectId = "black-practice-478018-u4",
    [string]$Region = "us-central1",
    [string]$ServiceName = "containerized-ml-api",
    [string]$ArtifactRegistry = "erp-repo",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$Image = "${Region}-docker.pkg.dev/${ProjectId}/${ArtifactRegistry}/${ServiceName}:${Tag}"

Write-Host "Building Docker image: $Image"
docker build -t $Image .

Write-Host "Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${Region}-docker.pkg.dev" --quiet

Write-Host "Pushing image..."
docker push $Image

Write-Host "Deploying to Cloud Run..."
gcloud run deploy $ServiceName `
    --image $Image `
    --region $Region `
    --project $ProjectId `
    --platform managed `
    --allow-unauthenticated `
    --memory 512Mi `
    --cpu 1 `
    --min-instances 0 `
    --max-instances 3 `
    --timeout 30 `
    --set-env-vars "ENVIRONMENT=production,MODEL_VERSION=1.0.0" `
    --quiet

$Url = gcloud run services describe $ServiceName `
    --region $Region `
    --project $ProjectId `
    --format "value(status.url)"

Write-Host ""
Write-Host "Deployed: $Url"
Write-Host "Health:   $Url/health"

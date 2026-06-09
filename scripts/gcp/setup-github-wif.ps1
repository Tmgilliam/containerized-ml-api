# One-time GitHub Actions → GCP auth setup (Workload Identity Federation).
param(
    [string]$ProjectId = "black-practice-478018-u4",
    [string]$ProjectNumber = "787251273541",
    [string]$GitHubRepo = "Tmgilliam/containerized-ml-api",
    [string]$ServiceAccountName = "github-ml-api-deploy",
    [string]$PoolId = "github-pool",
    [string]$ProviderId = "github-provider"
)

$ErrorActionPreference = "Stop"
$SaEmail = "${ServiceAccountName}@${ProjectId}.iam.gserviceaccount.com"

Write-Host "Enabling required APIs..."
gcloud services enable iamcredentials.googleapis.com run.googleapis.com artifactregistry.googleapis.com --project=$ProjectId

Write-Host "Creating service account: $SaEmail"
gcloud iam service-accounts create $ServiceAccountName `
    --display-name="GitHub Actions deploy for containerized-ml-api" `
    --project=$ProjectId 2>$null

Write-Host "Granting IAM roles..."
foreach ($role in @("roles/run.admin", "roles/artifactregistry.writer", "roles/iam.serviceAccountUser")) {
    gcloud projects add-iam-policy-binding $ProjectId `
        --member="serviceAccount:$SaEmail" `
        --role=$role `
        --quiet | Out-Null
}

Write-Host "Creating workload identity pool..."
gcloud iam workload-identity-pools create $PoolId `
    --location=global `
    --display-name="GitHub Actions Pool" `
    --project=$ProjectId 2>$null

Write-Host "Creating OIDC provider..."
gcloud iam workload-identity-pools providers create-oidc $ProviderId `
    --location=global `
    --workload-identity-pool=$PoolId `
    --display-name="GitHub Provider" `
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" `
    --issuer-uri="https://token.actions.githubusercontent.com" `
    --project=$ProjectId 2>$null

Write-Host "Binding service account to repository..."
gcloud iam service-accounts add-iam-policy-binding $SaEmail `
    --project=$ProjectId `
    --role="roles/iam.workloadIdentityUser" `
    --member="principalSet://iam.googleapis.com/projects/$ProjectNumber/locations/global/workloadIdentityPools/$PoolId/attribute.repository/$GitHubRepo" `
    --quiet | Out-Null

$Provider = "projects/$ProjectNumber/locations/global/workloadIdentityPools/$PoolId/providers/$ProviderId"

Write-Host ""
Write-Host "=== GitHub repository secrets ==="
Write-Host "GCP_PROJECT_ID=$ProjectId"
Write-Host "GCP_SERVICE_ACCOUNT=$SaEmail"
Write-Host "GCP_WORKLOAD_IDENTITY_PROVIDER=$Provider"
Write-Host ""
Write-Host "Set with:"
Write-Host "  gh secret set GCP_PROJECT_ID -b `"$ProjectId`""
Write-Host "  gh secret set GCP_SERVICE_ACCOUNT -b `"$SaEmail`""
Write-Host "  gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER -b `"$Provider`""

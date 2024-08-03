#####################
# Import requirements
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.38.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "5.38.0"
    }
  }
}

################
# Init providers
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "google-beta" {
  project = var.gcp_project
  region  = var.gcp_region
}

data "google_project" "gcp_project" {}

##########################
# Create Services Accounts (SA)
resource "google_service_account" "gcp_ml_sa" {
  account_id                   = var.gcp_ml_sa_account_id
  display_name                 = "Core ML Service Account to manage necessary services for AI/ML workloads."
  create_ignore_already_exists = true
}

###################
# Bind roles to SA
resource "google_project_iam_member" "gcp_cloudbuild_builds_editor" {
  project = var.gcp_project
  role    = "roles/cloudbuild.builds.editor"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_cloudbuild_integrations_editor" {
  project = var.gcp_project
  role    = "roles/cloudbuild.integrations.editor"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_secret_accessor" {
  project = var.gcp_project
  role    = "roles/secretmanager.secretAccessor"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_cloudstorage_user" {
  project = var.gcp_project
  role    = "roles/storage.objectUser"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_aiplatform_user" {
  project = var.gcp_project
  role    = "roles/aiplatform.user"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_aiplatform_developer" {
  project = var.gcp_project
  role    = "roles/ml.developer"
  member  = google_service_account.gcp_ml_sa.member
}

resource "google_project_iam_member" "gcp_artifactregistry_createonpushwriter" {
  project = var.gcp_project
  role    = "roles/artifactregistry.createOnPushWriter"
  member  = google_service_account.gcp_ml_sa.member
}

#####################################
# Create Workload identity (WI) Pools
resource "google_iam_workload_identity_pool" "gcp_wi_mlops_pool" {
  workload_identity_pool_id = "mlops-pool-v4"
  display_name              = "MLOps pool"
  description               = "Group all externals applications that need communication with GCP to perform CI/CD/CT."
  disabled                  = false
}

############################
# Define Identity Providers
resource "google_iam_workload_identity_pool_provider" "gcp_gha_oidc_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.gcp_wi_mlops_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions-oidc-provider"
  display_name                       = "GitHub Actions OIDC Provider"
  description                        = "Used to authenticate to Google Cloud from GHA"
  disabled                           = false
  attribute_mapping = {
    "google.subject"  = "assertion.sub"
    "attribute.aud"   = "assertion.aud"
    "attribute.actor" = "assertion.actor"
  }
  attribute_condition = "assertion.sub=='${var.gha_assertion_sub}' && assertion.aud=='${var.gha_assertion_aud}' && assertion.actor=='${var.gha_assertion_actor}'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

################################
# SA impersonations by providers
resource "google_service_account_iam_member" "gcp_ml_sa_impersonate_by_gha_oidc_provider" {
  service_account_id = google_service_account.gcp_ml_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gcp_wi_mlops_pool.name}/*"
}

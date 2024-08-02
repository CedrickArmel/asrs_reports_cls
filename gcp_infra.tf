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

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

provider "google-beta" {
  project = var.gcp_project
  region  = var.gcp_region
}

data "google_project" "gcp_project" {}

resource "google_service_account" "gcp_project_sa" {
  account_id                   = var.gcp_project_sa_account_id
  display_name                 = "Core Service Account"
  create_ignore_already_exists = true
}

resource "google_project_iam_member" "gcp_cloudbuild_builds_editor" {
  project = var.gcp_project
  role    = "roles/cloudbuild.builds.editor"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_cloudbuild_integrations_editor" {
  project = var.gcp_project
  role    = "roles/cloudbuild.integrations.editor"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_secret_accessor" {
  project = var.gcp_project
  role    = "roles/secretmanager.secretAccessor"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_cloudstorage_user" {
  project = var.gcp_project
  role    = "roles/storage.objectUser"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_aiplatform_user" {
  project = var.gcp_project
  role    = "roles/aiplatform.user"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_aiplatform_developer" {
  project = var.gcp_project
  role    = "roles/ml.developer"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_project_iam_member" "gcp_artifactregistry_createonpushwriter" {
  project = var.gcp_project
  role    = "roles/artifactregistry.createOnPushWriter"
  member  = google_service_account.gcp_project_sa.member
}

resource "google_iam_workload_identity_pool" "gcp_workload_identity_mlops_pool" {
  workload_identity_pool_id = "mlops-pool"
  display_name              = "MLOps pool"
  description               = "Group all externals applications that need communication with GCP to perform CI/CD/CT."
  disabled                  = false
}

resource "google_iam_workload_identity_pool_provider" "gcp_github_identity_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.gcp_workload_identity_mlops_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions-oidc"
  display_name                       = "GitHub Actions OIDC Provider"
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

resource "google_service_account_iam_member" "gcp_project_sa_impersonate" {
  service_account_id = google_service_account.gcp_project_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gcp_workload_identity_mlops_pool.name}/*"
}

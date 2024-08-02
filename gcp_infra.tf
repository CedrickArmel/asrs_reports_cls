##################
# Import providers
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

resource "google_service_account" "gcp_infra_sa" {
  account_id                   = var.gcp_infra_sa_account_id
  display_name                 = "Core Service Account to manage infrastructure and IAM"
  create_ignore_already_exists = true
}

###################
# Bind roles to SA

resource "google_project_iam_member" "gcp_platform_editor" {
  project = var.gcp_project
  role    = "roles/editor" # TODO: To permissive ! Analyse later in GCP console to fit better the need.
  member  = google_service_account.gcp_infra_sa.member
}

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
  workload_identity_pool_id = "mlops-pool-v2"
  display_name              = "MLOps pool"
  description               = "Group all externals applications that need communication with GCP to perform CI/CD/CT."
  disabled                  = false
}

resource "google_iam_workload_identity_pool" "gcp_wi_infra_pool" {
  workload_identity_pool_id = "infra-pool"
  display_name              = "Infrastructure pool"
  description               = "Group all externals applications that need communication with GCP to perform infrastructure lifecyle management."
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

resource "google_iam_workload_identity_pool_provider" "gcp_hcp_tf_oidc_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.gcp_wi_infra_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "hcp-tf-oidc-provider"
  display_name                       = "HCP Terraform OIDC Provider"
  description                        = "Used to authenticate to Google Cloud from HCP Terraform"
  attribute_condition                = "assertion.terraform_organization_name=='${var.hcp_terraform_org_name}'"
  attribute_mapping = {
    "google.subject"                     = "assertion.sub"
    "attribute.terraform_workspace_id"   = "assertion.terraform_workspace_id"
    "attribute.terraform_full_workspace" = "assertion.terraform_full_workspace"
  }
  oidc {
    issuer_uri = "https://app.terraform.io"
  }
}


################################
# SA impersonations by providers

resource "google_service_account_iam_member" "gcp_ml_sa_impersonate_by_gha_oidc_provider" {
  service_account_id = google_service_account.gcp_ml_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gcp_wi_mlops_pool.name}/*"
}

resource "google_service_account_iam_member" "gcp_infra_sa_impersonate_by_hcp_tf_oidc_provider" {
  service_account_id = google_service_account.gcp_ml_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.gcp_wi_infra_pool.name}/*"
}

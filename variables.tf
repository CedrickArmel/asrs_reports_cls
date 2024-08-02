###########
# Variables
variable "gha_assertion_aud" {
  description = "GHA workload identity JWk token aud attribute"
  type        = string
  default     = "https://github.com/CedrickArmel"
}

variable "gha_assertion_sub" {
  description = "GHA workload identity JWk token sub attribute"
  type        = string
  default     = "CedrickArmel/asrs_reports_cls:ref:refs/heads/main"
}

variable "gha_assertion_actor" {
  description = "GHA workload identity JWk token actor attribute"
  type        = string
  default     = "CedrickArmel"
}


#########
# Secrets
variable "gcp_infra_sa_account_id" {
  description = "Core Infra LC management SA"
  type        = string
  sensitive   = true
}

variable "gcp_ml_sa_account_id" {
  description = "Core ML tasks SA"
  type        = string
  sensitive   = true
}

variable "gcp_project" {
  description = "Google Cloud project to deploy on"
  type        = string
  sensitive   = true
}

variable "gcp_region" {
  description = "Google Cloud region to deploy on"
  type        = string
  sensitive   = true
}

variable "hcp_terraform_org_name" {
  description = "Organization name in HCP Terraform Cloud"
  type        = string
  sensitive   = true
}

variable "hcp_terraform_ws_id" {
  description = "Project's worksapce in HCP Terraform Cloud"
  type        = string
  sensitive   = true
}

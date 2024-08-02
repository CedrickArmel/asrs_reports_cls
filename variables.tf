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

variable "gcp_project_sa_account_id" {
  description = "Core project SA"
  type        = string
  sensitive   = true
}

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

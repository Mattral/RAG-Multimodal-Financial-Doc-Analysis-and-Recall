# Terraform — RAG Financial Multimodal Infrastructure

> **Status**: Reference stub. Adapt for your cloud provider (AWS/GCP/Azure).

## What this provisions
- EKS/GKE cluster with auto-scaling node groups
- RDS PostgreSQL (for pgvector adapter)
- ElastiCache Redis (for caching + rate limiting)
- S3 bucket (for document storage + audit logs)
- IAM roles + IRSA for pod-level AWS access
- KMS key for at-rest encryption
- WAF rules for the API ingress

## Usage
```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars   # fill in your values
terraform init
terraform plan
terraform apply
```

## Files
- `main.tf`      — Root module
- `variables.tf` — Input variables
- `outputs.tf`   — Output values (cluster endpoint, Redis URL, etc.)
- `modules/`     — Reusable modules (eks, rds, redis, s3, iam)

## Prerequisites
- Terraform >= 1.6
- AWS CLI configured, or GCP/Azure equivalent
- kubectl + helm installed

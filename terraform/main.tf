# ─────────────────────────────────────────────────────────────────────────────
# RAG Financial Multimodal — Terraform Root Module
# Provisions: EKS, RDS (pgvector), ElastiCache Redis, S3, IAM IRSA, KMS
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars  # fill in values
#   terraform init && terraform plan && terraform apply
#   aws eks update-kubeconfig --name <cluster> --region <region>
#   helm install rag-financial helm/rag-financial/ -n rag-prod --create-namespace
# ─────────────────────────────────────────────────────────────────────────────
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws        = { source = "hashicorp/aws",        version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes",  version = "~> 2.30" }
    helm       = { source = "hashicorp/helm",        version = "~> 2.13" }
    random     = { source = "hashicorp/random",      version = "~> 3.6" }
  }
  # backend "s3" {
  #   bucket         = "your-tf-state-bucket"
  #   key            = "rag-financial/terraform.tfstate"
  #   region         = var.region
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project     = "rag-financial-multimodal"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_availability_zones" "available" { state = "available" }
data "aws_caller_identity" "current" {}

resource "random_id" "suffix" { byte_length = 4 }

locals {
  name_prefix = "${var.cluster_name}-${var.environment}"
  account_id  = data.aws_caller_identity.current.account_id
  azs         = slice(data.aws_availability_zones.available.names, 0, 3)
}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.8"
  name    = "${local.name_prefix}-vpc"
  cidr    = "10.0.0.0/16"
  azs             = local.azs
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "prod"
  enable_dns_hostnames = true
  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

# ── KMS ───────────────────────────────────────────────────────────────────────
resource "aws_kms_key" "rag" {
  description             = "RAG Financial — envelope encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "rag" {
  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.rag.key_id
}

# ── S3: documents + audit logs ────────────────────────────────────────────────
resource "aws_s3_bucket" "documents" {
  bucket = "${local.name_prefix}-docs-${random_id.suffix.hex}"
}
resource "aws_s3_bucket" "audit" {
  bucket = "${local.name_prefix}-audit-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_versioning" "audit" {
  bucket = aws_s3_bucket.audit.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule { apply_server_side_encryption_by_default {
    sse_algorithm     = "aws:kms"
    kms_master_key_id = aws_kms_key.rag.arn
  }}
}
resource "aws_s3_bucket_server_side_encryption_configuration" "audit" {
  bucket = aws_s3_bucket.audit.id
  rule { apply_server_side_encryption_by_default {
    sse_algorithm     = "aws:kms"
    kms_master_key_id = aws_kms_key.rag.arn
  }}
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true; block_public_policy     = true
  ignore_public_acls      = true; restrict_public_buckets = true
}
resource "aws_s3_bucket_public_access_block" "audit" {
  bucket                  = aws_s3_bucket.audit.id
  block_public_acls       = true; block_public_policy     = true
  ignore_public_acls      = true; restrict_public_buckets = true
}

# Retain audit logs for 7 years (financial compliance)
resource "aws_s3_bucket_lifecycle_configuration" "audit" {
  bucket = aws_s3_bucket.audit.id
  rule {
    id     = "audit-retention"
    status = "Enabled"
    transition { days = 90;  storage_class = "STANDARD_IA" }
    transition { days = 365; storage_class = "GLACIER" }
    expiration { days = 2555 }
  }
}

# ── EKS cluster ───────────────────────────────────────────────────────────────
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.11"
  cluster_name    = var.cluster_name
  cluster_version = "1.30"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_encryption_config = {
    resources        = ["secrets"]
    provider_key_arn = aws_kms_key.rag.arn
  }
  eks_managed_node_groups = {
    general = {
      instance_types = [var.node_instance_type]
      min_size       = var.min_nodes
      max_size       = var.max_nodes
      desired_size   = var.min_nodes
    }
  }
  enable_irsa = true
}

# ── IAM IRSA role for the RAG API pod ────────────────────────────────────────
resource "aws_iam_role" "rag_api" {
  name = "${local.name_prefix}-api"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = module.eks.oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider}:sub" = "system:serviceaccount:rag-prod:rag-financial"
          "${module.eks.oidc_provider}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "rag_api_s3" {
  name = "s3-access"
  role = aws_iam_role.rag_api.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn, "${aws_s3_bucket.documents.arn}/*",
          aws_s3_bucket.audit.arn,     "${aws_s3_bucket.audit.arn}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["kms:GenerateDataKey","kms:Decrypt"]
        Resource = aws_kms_key.rag.arn
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:${local.name_prefix}/*"
      }
    ]
  })
}

# ── RDS PostgreSQL + pgvector ─────────────────────────────────────────────────
resource "aws_db_subnet_group" "rag" {
  name       = "${local.name_prefix}-db"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "${local.name_prefix}-rds"
  vpc_id = module.vpc.vpc_id
  ingress { from_port=5432; to_port=5432; protocol="tcp"; cidr_blocks=[module.vpc.vpc_cidr_block] }
  egress  { from_port=0; to_port=0; protocol="-1"; cidr_blocks=["0.0.0.0/0"] }
}

resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}:?"
}

resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${local.name_prefix}/rds-password"
  recovery_window_in_days = 7
  kms_key_id              = aws_kms_key.rag.arn
}
resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db.result
}

resource "aws_db_parameter_group" "rag" {
  name   = "${local.name_prefix}-pg16"
  family = "postgres16"
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
}

resource "aws_db_instance" "rag" {
  identifier             = "${local.name_prefix}-postgres"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.db_instance_class
  allocated_storage      = 100
  max_allocated_storage  = 1000
  storage_encrypted      = true
  kms_key_id             = aws_kms_key.rag.arn
  db_name                = "ragfinancial"
  username               = "ragadmin"
  password               = random_password.db.result
  db_subnet_group_name   = aws_db_subnet_group.rag.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.rag.name
  backup_retention_period = 7
  deletion_protection     = var.environment == "prod"
  skip_final_snapshot     = var.environment != "prod"
  performance_insights_enabled = true
  monitoring_interval          = 60
  tags = { Name = "${local.name_prefix}-postgres" }
}

# ── ElastiCache Redis ─────────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "rag" {
  name       = "${local.name_prefix}-cache"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name   = "${local.name_prefix}-redis"
  vpc_id = module.vpc.vpc_id
  ingress { from_port=6379; to_port=6379; protocol="tcp"; cidr_blocks=[module.vpc.vpc_cidr_block] }
  egress  { from_port=0; to_port=0; protocol="-1"; cidr_blocks=["0.0.0.0/0"] }
}

resource "aws_elasticache_replication_group" "rag" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "RAG Financial — cache + rate limiter"
  node_type                  = var.redis_node_type
  port                       = 6379
  num_cache_clusters         = var.environment == "prod" ? 2 : 1
  subnet_group_name          = aws_elasticache_subnet_group.rag.name
  security_group_ids         = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                 = aws_kms_key.rag.arn
  automatic_failover_enabled = var.environment == "prod"
  multi_az_enabled           = var.environment == "prod"
  snapshot_retention_limit   = 1
  tags = { Name = "${local.name_prefix}-redis" }
}

# ── Secrets Manager — API keys ────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "openai_api_key" {
  name                    = "${local.name_prefix}/openai-api-key"
  recovery_window_in_days = 7
  kms_key_id              = aws_kms_key.rag.arn
}
resource "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id     = aws_secretsmanager_secret.openai_api_key.id
  secret_string = var.openai_api_key
  lifecycle { ignore_changes = [secret_string] }
}

# ── K8s providers (post-cluster) ──────────────────────────────────────────────
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
    }
  }
}

# ── Namespaces + IRSA annotation ──────────────────────────────────────────────
resource "kubernetes_namespace" "rag_prod" {
  metadata { name = "rag-prod" }
}

resource "kubernetes_annotations" "rag_sa_irsa" {
  api_version = "v1"
  kind        = "ServiceAccount"
  metadata {
    name      = "rag-financial"
    namespace = kubernetes_namespace.rag_prod.metadata[0].name
  }
  annotations = {
    "eks.amazonaws.com/role-arn" = aws_iam_role.rag_api.arn
  }
  depends_on = [module.eks]
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = "# aws eks describe-cluster --name ${var.cluster_name} --query cluster.endpoint"
  sensitive   = false
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = "# See AWS console → ElastiCache → ${var.cluster_name}-redis"
}

output "s3_bucket_name" {
  description = "S3 bucket for documents and audit logs"
  value       = "${var.cluster_name}-${var.environment}-documents"
}

output "deploy_command" {
  description = "Command to deploy the Helm chart after provisioning"
  value       = "helm install rag-financial helm/rag-financial/ --namespace rag-prod --create-namespace -f helm/rag-financial/values-${var.environment}.yaml"
}

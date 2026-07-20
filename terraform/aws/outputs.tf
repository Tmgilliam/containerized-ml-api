output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.api.repository_url
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.api.dns_name
}

output "api_url" {
  description = "API base URL"
  value       = "http://${aws_lb.api.dns_name}"
}

output "health_endpoint" {
  description = "Health check endpoint"
  value       = "http://${aws_lb.api.dns_name}/health"
}

output "predict_endpoint" {
  description = "Prediction endpoint"
  value       = "http://${aws_lb.api.dns_name}/predict"
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.api.name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.api.name
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

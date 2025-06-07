variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "spygate"
}

variable "environment" {
  description = "Environment (e.g., dev, staging, prod)"
  type        = string
  default     = "staging"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance (Windows Server 2019)"
  type        = string
  default     = "ami-0d80714a054d3360c"  # Update with latest Windows Server 2019 AMI
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.large"
}

variable "admin_cidr_block" {
  description = "CIDR block for admin access"
  type        = string
  default     = "0.0.0.0/0"  # Update with your IP range
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

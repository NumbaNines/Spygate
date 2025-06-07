terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket = "spygate-terraform-state"
    key    = "state/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for video storage
resource "aws_s3_bucket" "video_storage" {
  bucket = "${var.project_name}-video-storage-${var.environment}"

  tags = {
    Name        = "Video Storage"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "video_storage_versioning" {
  bucket = aws_s3_bucket.video_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "video_storage_encryption" {
  bucket = aws_s3_bucket.video_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# EC2 instance for application
resource "aws_instance" "app_server" {
  ami           = var.ami_id
  instance_type = var.instance_type

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  vpc_security_group_ids = [aws_security_group.app_security_group.id]

  tags = {
    Name        = "${var.project_name}-app-server-${var.environment}"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Security group for EC2 instance
resource "aws_security_group" "app_security_group" {
  name        = "${var.project_name}-security-group-${var.environment}"
  description = "Security group for Spygate application"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-security-group-${var.environment}"
    Environment = var.environment
    Project     = var.project_name
  }
}

# RDS instance for PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier           = "${var.project_name}-db-${var.environment}"
  allocated_storage    = 20
  storage_type        = "gp2"
  engine              = "postgres"
  engine_version      = "13.7"
  instance_class      = "db.t3.micro"
  db_name             = "spygate"
  username            = var.db_username
  password            = var.db_password
  skip_final_snapshot = true

  tags = {
    Name        = "${var.project_name}-db-${var.environment}"
    Environment = var.environment
    Project     = var.project_name
  }
}

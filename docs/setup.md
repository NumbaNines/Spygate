# Spygate Setup Guide

This guide provides step-by-step instructions for setting up the Spygate development environment.

## Prerequisites

- Windows 10 Pro/Enterprise/Education
- Python 3.9+
- Docker Desktop for Windows
- Git
- GitHub account with appropriate permissions
- AWS account with appropriate permissions
- Terraform installed

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spygate.git
   cd spygate
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Database Configuration

1. Create `.env.database` file in the project root:

   ```env
   DB_TYPE=postgresql
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=spygate
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```

2. Initialize the database:

   ```bash
   python scripts/init_db.py
   ```

   Note: The script will automatically fall back to SQLite if PostgreSQL is not available.

## Docker Setup

1. Build and start containers:

   ```bash
   docker-compose up --build
   ```

2. Verify the application is running:
   - Web interface: http://localhost:8000
   - PostgreSQL: localhost:5432

## AWS Infrastructure

1. Configure AWS credentials:

   ```bash
   aws configure
   ```

2. Initialize Terraform:

   ```bash
   cd terraform
   terraform init
   ```

3. Create a `terraform.tfvars` file:

   ```hcl
   environment    = "staging"
   db_username    = "your_username"
   db_password    = "your_password"
   admin_cidr_block = "your_ip_range"
   ```

4. Apply Terraform configuration:
   ```bash
   terraform plan
   terraform apply
   ```

## Additional Repositories

1. Set GitHub token:

   ```bash
   export GITHUB_TOKEN='your-github-token'
   ```

2. Create additional repositories:
   ```bash
   python scripts/create_repos.py
   ```

## Logging and Monitoring

1. Configure Sentry:

   ```bash
   export SENTRY_DSN='your-sentry-dsn'
   ```

2. Access monitoring endpoints:
   - Health check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

## CI/CD Setup

1. Add required secrets to GitHub repository:

   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

2. Push changes to trigger CI/CD:
   ```bash
   git add .
   git commit -m "Initial setup"
   git push origin main
   ```

## Development Workflow

1. Create a new branch for features:

   ```bash
   git checkout -b feature/your-feature
   ```

2. Run tests:

   ```bash
   pytest
   ```

3. Check code quality:

   ```bash
   black .
   flake8 .
   mypy src tests
   ```

4. Create pull request when ready

## Troubleshooting

### Common Issues

1. Database Connection:

   - Verify PostgreSQL is running
   - Check credentials in `.env.database`
   - Try SQLite fallback

2. Docker:

   - Ensure Docker Desktop is running
   - Check port conflicts
   - Verify Windows containers mode

3. AWS:
   - Verify credentials
   - Check VPC settings
   - Validate security group rules

### Getting Help

- Check the [documentation](docs/)
- Open an issue on GitHub
- Contact the development team

## Security Notes

- Keep `.env` files secure and never commit them
- Regularly rotate API keys and credentials
- Follow least privilege principle for AWS IAM
- Keep Windows and Docker up to date

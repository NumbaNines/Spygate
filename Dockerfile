# Use Python 3.9 Windows base image
FROM python:3.9-windowsservercore

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="C:\Program Files\PostgreSQL\13\bin;${PATH}"

# Set working directory
WORKDIR /app

# Install system dependencies
SHELL ["powershell", "-Command"]
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1')); \
    choco install -y postgresql13 vcredist140 git

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs data models

# Expose ports for the application
EXPOSE 8000

# Set the default command
CMD ["python", "src/main.py"]

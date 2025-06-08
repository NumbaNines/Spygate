# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=spygate_site.settings.production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/media /app/static

# Collect static files
RUN python spygate_site/manage.py collectstatic --noinput

# Create a non-root user
RUN useradd -m spygate
RUN chown -R spygate:spygate /app
USER spygate

# Run gunicorn
CMD ["gunicorn", "--chdir", "spygate_site", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "spygate_site.wsgi:application"]

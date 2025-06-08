"""Health check utilities."""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import requests

from src.utils.logging import logger


class HealthChecker:
    """Health check implementation."""

    def __init__(self):
        """Initialize health checker."""
        self.last_check = None
        self.metrics = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_usage_percent": 0,
            "api_latency": 0,
        }

    def check_system_health(self) -> dict[str, Any]:
        """Check system health metrics."""
        try:
            # CPU usage
            self.metrics["cpu_percent"] = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_percent"] = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            self.metrics["disk_usage_percent"] = disk.percent

            # Update last check timestamp
            self.last_check = datetime.now()

            logger.info(
                "System health check completed",
                extra={
                    "metrics": self.metrics,
                    "timestamp": self.last_check.isoformat(),
                },
            )

            return self.metrics

        except Exception as e:
            logger.error("Error during system health check", extra={"error": str(e)})
            raise

    def check_api_health(self, endpoint: str) -> bool:
        """Check API endpoint health."""
        try:
            start_time = datetime.now()
            response = requests.get(endpoint)
            end_time = datetime.now()

            # Calculate latency
            latency = (end_time - start_time).total_seconds() * 1000  # ms
            self.metrics["api_latency"] = latency

            is_healthy = response.status_code == 200

            logger.info(
                "API health check completed",
                extra={
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "latency_ms": latency,
                    "is_healthy": is_healthy,
                },
            )

            return is_healthy

        except Exception as e:
            logger.error(
                "Error during API health check",
                extra={"endpoint": endpoint, "error": str(e)},
            )
            return False

    def get_metrics(self) -> dict[str, Any]:
        """Get current health metrics."""
        return {
            "metrics": self.metrics,
            "last_check": self.last_check.isoformat() if self.last_check else None,
        }


# Create singleton instance
health_checker = HealthChecker()

# Example usage:
# status = health_checker.check_system_health()
# print(status)

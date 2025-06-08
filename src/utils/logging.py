"""Logging configuration for Spygate."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger("spygate")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = RotatingFileHandler(
    logs_dir / "spygate.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set propagate to False to avoid duplicate logs
logger.propagate = False

# Sentry configuration
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

# Prometheus metrics
from prometheus_client import Counter, Gauge, start_http_server

REQUESTS = Counter("spygate_requests_total", "Total requests processed")
PROCESSING_TIME = Gauge("spygate_processing_seconds", "Time spent processing request")
ERROR_COUNT = Counter("spygate_errors_total", "Total number of errors")
DISK_USAGE = Gauge("spygate_disk_usage_bytes", "Disk space usage in bytes")

# Start Prometheus metrics server if port is configured
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
try:
    start_http_server(METRICS_PORT)
except Exception as e:
    logger.warning(f"Failed to start metrics server: {e}")


class CustomLogger:
    """Custom logger with context support."""

    def __init__(self, context: Optional[dict[str, Any]] = None):
        """Initialize the logger with custom configuration."""
        self.context = context or {}

    def log(self, level: str, message: str, extra: Optional[dict[str, Any]] = None) -> None:
        """Log a message with context."""
        log_data = {**self.context}
        if extra:
            log_data.update(extra)

        if level == "debug":
            logger.debug(message, extra=log_data)
        elif level == "info":
            logger.info(message, extra=log_data)
        elif level == "warning":
            logger.warning(message, extra=log_data)
        elif level == "error":
            logger.error(message, extra=log_data)

            # Send error to Sentry if configured
            if SENTRY_DSN:
                with sentry_sdk.push_scope() as scope:
                    for key, value in log_data.items():
                        scope.set_extra(key, value)
                    sentry_sdk.capture_message(message, level="error")

    def monitor_disk_space(self, path: str = ".") -> None:
        """Monitor disk space usage."""
        try:
            total, used, free = self._get_disk_usage(path)
            DISK_USAGE.set(used)

            # Log warning if disk usage is above 90%
            usage_percent = (used / total) * 100
            if usage_percent > 90:
                self.log("warning", f"High disk usage: {usage_percent:.1f}%")

        except Exception as e:
            self.log("error", f"Error monitoring disk space: {e}")

    def _get_disk_usage(self, path: str) -> tuple[int, int, int]:
        """Get disk usage statistics."""
        if sys.platform == "win32":
            import ctypes

            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path),
                None,
                ctypes.pointer(total_bytes),
                ctypes.pointer(free_bytes),
            )
            total = total_bytes.value
            free = free_bytes.value
            used = total - free
            return total, used, free
        else:
            import shutil

            total, used, free = shutil.disk_usage(path)
            return total, used, free


# Create global logger instance
log = CustomLogger()

# Example usage:
# log.info("Database connection established", db_type="postgresql", host="localhost")
# log.error("Failed to process video", video_id="123", error_type="FileNotFound")

import logging
import os
import sys
import time
from pathlib import Path

import pytest
from prometheus_client.parser import text_string_to_metric_families

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from src.utils.health_check import health_checker
from src.utils.logging import logger


@pytest.fixture(scope="function")
def test_log_file(tmp_path):
    """Create a temporary log file."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "spygate.log"

    # Store original handlers
    original_handlers = logger.handlers.copy()

    # Create a new file handler for testing
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    yield log_file

    # Clean up
    logger.handlers = original_handlers
    if log_file.exists():
        log_file.unlink()


def test_logger_configuration():
    """Test logger configuration."""
    assert logger.name == "spygate"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 2  # Console and file handlers
    assert not logger.propagate


def test_log_levels(test_log_file):
    """Test different log levels."""
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_content = test_log_file.read_text()

    # Debug messages should not be logged (default level is INFO)
    assert "Debug message" not in log_content
    assert "Info message" in log_content
    assert "Warning message" in log_content
    assert "Error message" in log_content


def test_log_formatting(test_log_file):
    """Test log message formatting."""
    logger.info("Test message")

    log_content = test_log_file.read_text()
    log_line = log_content.strip().split("\n")[-1]

    # Check basic formatting
    assert "INFO" in log_line
    assert "spygate" in log_line
    assert "Test message" in log_line

    # Check timestamp format
    timestamp_part = log_line.split("|")[0].strip()
    assert len(timestamp_part) == 19  # YYYY-MM-DD HH:mm:ss


def test_health_check_initialization():
    """Test that the health checker is properly initialized."""
    # Test database health check
    db_status = health_checker.check_database()
    assert isinstance(db_status, dict), "Database health check should return a dictionary"
    assert "status" in db_status, "Database health check should include status"
    assert "timestamp" in db_status, "Database health check should include timestamp"

    # Test disk space check
    disk_status = health_checker.check_disk_space()
    assert isinstance(disk_status, dict), "Disk space check should return a dictionary"
    assert "status" in disk_status, "Disk space check should include status"
    assert "free_space_gb" in disk_status, "Disk space check should include free space"

    # Test overall health check
    overall_status = health_checker.check_all()
    assert isinstance(overall_status, dict), "Overall health check should return a dictionary"
    assert "database" in overall_status, "Overall health check should include database status"
    assert "disk" in overall_status, "Overall health check should include disk status"
    assert "overall_status" in overall_status, "Overall health check should include overall status"


def test_health_check_database():
    """Test database health check functionality."""
    result = health_checker.check_database()
    assert isinstance(result, dict), "Health check should return a dictionary"
    assert "status" in result, "Health check should include status"
    assert "db_type" in result, "Health check should include database type"
    assert "timestamp" in result, "Health check should include timestamp"


def test_health_check_disk_space():
    """Test disk space health check functionality."""
    result = health_checker.check_disk_space()
    assert isinstance(result, dict), "Health check should return a dictionary"
    assert "status" in result, "Health check should include status"
    assert (
        "free_space_gb" in result or "error" in result
    ), "Health check should include space info or error"
    assert "timestamp" in result, "Health check should include timestamp"


def test_prometheus_metrics():
    """Test that Prometheus metrics are being recorded."""
    # Wait a bit for metrics to be registered
    time.sleep(1)

    # Get metrics from Prometheus
    import requests

    try:
        response = requests.get("http://localhost:9090/metrics", timeout=10)
        metrics_text = response.text
    except:
        # If Prometheus server is not accessible, skip test
        pytest.skip("Prometheus server not accessible")

    # Parse metrics
    metrics = list(text_string_to_metric_families(metrics_text))
    metric_names = [metric.name for metric in metrics]

    # Check if our custom metrics are present
    assert "spygate_system_health" in metric_names, "System health metric should be present"
    assert "spygate_db_connection_status" in metric_names, "DB connection metric should be present"


def test_health_check_all():
    """Test the comprehensive health check functionality."""
    result = health_checker.check_all()
    assert isinstance(result, dict), "Health check should return a dictionary"
    assert "database" in result, "Health check should include database status"
    assert "disk" in result, "Health check should include disk status"
    assert "overall_status" in result, "Health check should include overall status"
    assert "timestamp" in result, "Health check should include timestamp"


def test_logger_context():
    """Test logger context functionality."""
    # Create logger with context
    context_logger = logger.__class__({"app_version": "1.0.0", "environment": "test"})

    # Log with additional context
    context_logger.info("Test message with context", user_id="123", action="test")

    # No assertion needed as we're mainly testing that it doesn't raise exceptions
    # In a real environment, you'd verify the log output contains the context


def test_sentry_integration():
    """Test Sentry integration (if configured)."""
    if os.getenv("SENTRY_DSN"):
        # Test error reporting
        try:
            raise ValueError("Test error for Sentry")
        except Exception as e:
            logger.error("Test error occurred", error=str(e), error_type=type(e).__name__)
    else:
        pytest.skip("Sentry DSN not configured")

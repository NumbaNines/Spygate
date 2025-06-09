"""
Tests for Prometheus metrics exporter.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from ..core.performance_monitor import PerformanceMonitor
from ..utils.prometheus_exporter import PrometheusExporter


@pytest.fixture
def mock_monitor():
    """Create a mock performance monitor."""
    monitor = MagicMock(spec=PerformanceMonitor)

    # Mock stats
    monitor.get_performance_stats.return_value = {
        "fps": 30.0,
        "memory_usage_mb": 1024.0,
        "gpu_memory_mb": 512.0,
        "quality_level": 0.8,
        "dropped_frames": 0,
        "optimization_events": 0,
        "frames_processed": 1000,
    }

    # Mock metrics report
    monitor.get_metrics_report.return_value = {
        "processing_times": [0.033] * 10,
        "batch_times": [0.15] * 5,
    }

    return monitor


def test_prometheus_exporter(mock_monitor):
    """Test Prometheus metrics export."""
    # Start exporter on random port
    exporter = PrometheusExporter(monitor=mock_monitor, port=8765, interval=0.1)

    try:
        # Wait for metrics collection
        time.sleep(0.5)

        # Get metrics
        response = requests.get("http://localhost:8765/metrics", timeout=10)
        assert response.status_code == 200

        metrics = response.text

        # Check basic metrics presence
        assert "spygate_fps" in metrics
        assert "spygate_memory_usage_mb" in metrics
        assert "spygate_gpu_memory_mb" in metrics
        assert "spygate_quality_level" in metrics
        assert "spygate_dropped_frames_total" in metrics
        assert "spygate_optimization_events_total" in metrics
        assert "spygate_frames_processed_total" in metrics
        assert "spygate_processing_time_seconds" in metrics

        # Check metric values
        assert "30.0" in metrics  # FPS
        assert "1024.0" in metrics  # Memory
        assert "512.0" in metrics  # GPU Memory
        assert "0.8" in metrics  # Quality

    finally:
        exporter.stop()


def test_prometheus_error_handling(mock_monitor):
    """Test error handling in metrics collection."""
    # Make monitor raise an exception
    mock_monitor.get_performance_stats.side_effect = Exception("Test error")

    with patch("logging.error") as mock_log:
        exporter = PrometheusExporter(monitor=mock_monitor, port=8766, interval=0.1)

        try:
            # Wait for error to occur
            time.sleep(0.5)

            # Check error was logged
            mock_log.assert_called_with("Error updating Prometheus metrics: Test error")

        finally:
            exporter.stop()


def test_prometheus_cleanup(mock_monitor):
    """Test proper cleanup of exporter resources."""
    exporter = PrometheusExporter(monitor=mock_monitor, port=8767, interval=0.1)

    # Get initial metrics
    response = requests.get("http://localhost:8767/metrics", timeout=10)
    assert response.status_code == 200

    # Stop exporter
    exporter.stop()

    # Thread should be stopped
    assert not exporter._thread.is_alive()


def test_prometheus_custom_prefix(mock_monitor):
    """Test custom metric prefix."""
    exporter = PrometheusExporter(monitor=mock_monitor, port=8768, prefix="custom_")

    try:
        time.sleep(0.5)

        response = requests.get("http://localhost:8768/metrics", timeout=10)
        metrics = response.text

        # Check custom prefix
        assert "custom_fps" in metrics
        assert "custom_memory_usage_mb" in metrics
        assert "custom_gpu_memory_mb" in metrics
        assert "custom_quality_level" in metrics
        assert "custom_dropped_frames_total" in metrics
        assert "custom_optimization_events_total" in metrics
        assert "custom_frames_processed_total" in metrics
        assert "custom_processing_time_seconds" in metrics

    finally:
        exporter.stop()


def test_prometheus_metric_types(mock_monitor):
    """Test different metric types are exported correctly."""
    exporter = PrometheusExporter(monitor=mock_monitor, port=8769)

    try:
        time.sleep(0.5)

        response = requests.get("http://localhost:8769/metrics", timeout=10)
        metrics = response.text

        # Check gauge format
        assert "spygate_fps{" not in metrics  # No labels
        assert "# TYPE spygate_fps gauge" in metrics

        # Check counter format
        assert "# TYPE spygate_frames_processed_total counter" in metrics

        # Check histogram format
        assert "# TYPE spygate_processing_time_seconds histogram" in metrics
        assert "spygate_processing_time_seconds_bucket{le=" in metrics
        assert "spygate_processing_time_seconds_sum" in metrics
        assert "spygate_processing_time_seconds_count" in metrics

    finally:
        exporter.stop()

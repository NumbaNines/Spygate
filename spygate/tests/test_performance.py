"""
Performance optimization test suite.

This module contains tests to validate performance targets and optimization behavior.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ..core.hardware import HardwareDetector
from ..core.performance_monitor import PerformanceMonitor, PerformanceThresholds
from ..utils.metrics import MetricsCollector


def test_performance_thresholds():
    """Test performance thresholds are enforced."""
    monitor = PerformanceMonitor()

    # Simulate high memory usage
    with patch.object(monitor, "_update_system_metrics") as mock_metrics:
        mock_metrics.side_effect = lambda: monitor.memory_usage.append(
            monitor.thresholds.max_memory_mb * 0.95
        )

        # Let monitor run
        time.sleep(2)

        # Check quality was reduced
        assert monitor.get_quality_level() < monitor.thresholds.max_quality
        assert monitor.stats["optimization_events"] > 0


def test_adaptive_quality():
    """Test quality adapts based on performance."""
    monitor = PerformanceMonitor()

    # Simulate good performance
    start_time = time.perf_counter()
    for _ in range(100):
        monitor.end_frame(start_time)
        start_time = time.perf_counter()
        time.sleep(0.01)  # 100 FPS

    # Quality should stay at max
    assert monitor.get_quality_level() == monitor.thresholds.max_quality

    # Simulate poor performance
    start_time = time.perf_counter()
    for _ in range(100):
        monitor.end_frame(start_time)
        start_time = time.perf_counter()
        time.sleep(0.1)  # 10 FPS

    # Quality should be reduced
    assert monitor.get_quality_level() < monitor.thresholds.max_quality


def test_metrics_collection():
    """Test comprehensive metrics collection."""
    metrics = MetricsCollector(storage_path=".taskmaster/.temp/metrics")

    # Record various metric types
    metrics.record_gauge("fps", 30.0)
    metrics.record_counter("frames", 1)
    metrics.record_event("quality_reduction", {"reason": "memory"})
    metrics.record_histogram("processing_time", 0.033)

    # Check gauge stats
    stats = metrics.get_gauge_stats("fps")
    assert stats["count"] == 1
    assert stats["mean"] == 30.0

    # Check counter
    assert metrics.get_counter_value("frames") == 1

    # Check events
    events = metrics.get_events("quality_reduction")
    assert len(events) == 1
    assert events[0]["attributes"]["reason"] == "memory"

    # Check histogram
    stats = metrics.get_histogram_stats("processing_time", percentiles=[50, 95, 99])
    assert stats["count"] == 1
    assert stats["mean"] == 0.033
    assert "p95" in stats


def test_performance_targets():
    """Test system meets performance targets under load."""
    monitor = PerformanceMonitor()

    # Process 1000 frames
    times = []
    for _ in range(1000):
        start = monitor.start_frame()
        # Simulate processing
        time.sleep(0.01)
        monitor.end_frame(start)
        times.append(time.perf_counter() - start)

    stats = monitor.get_performance_stats()

    # Verify performance targets
    assert stats["fps"] >= monitor.thresholds.min_fps
    assert stats["avg_processing_time"] <= monitor.thresholds.max_processing_time
    assert stats["memory_usage_mb"] <= monitor.thresholds.max_memory_mb

    # Check timing stability
    std_time = np.std(times)
    mean_time = np.mean(times)
    cv = std_time / mean_time  # Coefficient of variation
    assert cv < 0.2  # Less than 20% variation


def test_gpu_optimization():
    """Test GPU memory optimization if available."""
    hardware = HardwareDetector()
    if not hardware.has_cuda:
        pytest.skip("No GPU available")

    monitor = PerformanceMonitor(hardware=hardware)

    # Simulate high GPU usage
    with patch.object(hardware, "get_gpu_memory_usage") as mock_gpu:
        mock_gpu.return_value = monitor.thresholds.max_gpu_memory_mb * 0.95

        # Let monitor run
        time.sleep(2)

        # Check quality was reduced
        assert monitor.get_quality_level() < monitor.thresholds.max_quality
        assert monitor.stats["optimization_events"] > 0


def test_metrics_persistence():
    """Test metrics are properly saved and loaded."""
    metrics = MetricsCollector(storage_path=".taskmaster/.temp/metrics")

    # Record some metrics
    metrics.record_gauge("test_gauge", 42.0)
    metrics.record_counter("test_counter", 7)
    metrics.record_histogram("test_hist", 0.123)

    # Save metrics
    metrics.save_metrics()

    # Create new collector and load
    new_metrics = MetricsCollector(storage_path=".taskmaster/.temp/metrics")
    new_metrics.load_metrics()

    # Verify metrics were restored
    gauge_stats = new_metrics.get_gauge_stats("test_gauge")
    assert gauge_stats["mean"] == 42.0

    hist_stats = new_metrics.get_histogram_stats("test_hist")
    assert hist_stats["mean"] == 0.123


def test_cleanup_behavior():
    """Test automatic cleanup and memory recovery."""
    monitor = PerformanceMonitor()
    metrics = monitor.metrics

    # Add test data
    for _ in range(monitor.thresholds.cleanup_interval + 10):
        start = monitor.start_frame()
        monitor.end_frame(start)

    # Force cleanup
    initial_gauges = len(metrics._gauges)
    metrics.cleanup_old_data()

    # Verify cleanup occurred
    assert len(metrics._gauges) <= initial_gauges

    # Check quality recovery
    if monitor.stats["fps"] > monitor.thresholds.target_fps * 1.1:
        assert monitor.get_quality_level() >= monitor.thresholds.min_quality


def test_batch_processing():
    """Test batch processing performance."""
    monitor = PerformanceMonitor()

    # Process several batches
    for _ in range(10):
        start = monitor.start_batch()
        # Simulate batch processing
        time.sleep(0.1)
        monitor.end_batch(start)

    # Get metrics report
    report = monitor.get_metrics_report()

    # Verify batch times
    assert len(report["batch_times"]) == 10
    assert all(t <= monitor.thresholds.max_batch_time for t in report["batch_times"])


def test_concurrent_access():
    """Test thread-safety of monitoring system."""
    monitor = PerformanceMonitor()
    metrics = monitor.metrics

    def worker():
        for _ in range(100):
            start = monitor.start_frame()
            time.sleep(0.01)
            monitor.end_frame(start)
            metrics.record_gauge("worker_fps", 30.0)

    # Start multiple threads
    threads = []
    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    # Verify no data corruption
    stats = monitor.get_performance_stats()
    assert isinstance(stats["fps"], float)
    assert isinstance(stats["avg_processing_time"], float)

    gauge_stats = metrics.get_gauge_stats("worker_fps")
    assert gauge_stats["count"] > 0

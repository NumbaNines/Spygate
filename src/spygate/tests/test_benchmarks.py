"""
Performance benchmark tests.

This module contains benchmark tests to validate performance targets.
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from ..core.hardware import HardwareDetector
from ..core.performance_monitor import PerformanceMonitor
from ..video.frame_preprocessor import FramePreprocessor
from ..video.object_tracker import ObjectTracker


@pytest.fixture
def mock_frame():
    """Create a mock video frame."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def mock_detections():
    """Create mock object detections."""
    return [
        {"bbox": [100, 100, 200, 200], "confidence": 0.95, "class_id": 0, "class_name": "person"},
        {"bbox": [300, 300, 400, 400], "confidence": 0.85, "class_id": 0, "class_name": "person"},
    ]


def test_frame_preprocessing_speed(mock_frame, mock_config, mock_hardware, benchmark):
    """Benchmark frame preprocessing performance."""
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)

    def preprocess():
        frame = preprocessor.preprocess(mock_frame.copy())
        assert frame is not None
        assert frame.shape[0] > 0

    # Run benchmark
    result = benchmark(preprocess)

    # Verify performance targets
    assert result.stats["mean"] < 0.02  # 20ms max
    assert result.stats["stddev"] / result.stats["mean"] < 0.2  # Stable timing


def test_object_tracking_speed(mock_frame, mock_detections, mock_config, mock_hardware, benchmark):
    """Benchmark object tracking performance."""
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)

    def track():
        tracks = tracker.update(mock_frame.copy(), mock_detections)
        assert len(tracks) > 0

    # Run benchmark
    result = benchmark(track)

    # Verify performance targets
    assert result.stats["mean"] < 0.05  # 50ms max
    assert result.stats["stddev"] / result.stats["mean"] < 0.2  # Stable timing


def test_memory_usage(mock_frame, mock_detections, mock_config, mock_hardware):
    """Test memory usage under load."""
    import psutil

    process = psutil.Process()

    # Get initial memory
    initial_memory = process.memory_info().rss

    # Create components
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)
    monitor = PerformanceMonitor()

    # Process many frames
    for _ in range(1000):
        frame = preprocessor.preprocess(mock_frame.copy())
        tracks = tracker.update(frame, mock_detections)

        # Record metrics
        monitor.end_frame(monitor.start_frame())

    # Get final memory
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

    # Verify memory targets
    assert memory_increase < 500  # Max 500MB increase
    assert monitor.stats["memory_usage_mb"] < monitor.thresholds.max_memory_mb


def test_gpu_memory(mock_frame, mock_detections, mock_config, mock_hardware):
    """Test GPU memory usage if available."""
    if not mock_hardware.has_cuda:
        pytest.skip("No GPU available")

    # Get initial GPU memory
    initial_gpu = mock_hardware.get_gpu_memory_usage()

    # Create components with GPU support
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)
    monitor = PerformanceMonitor(hardware=mock_hardware)

    # Process many frames
    for _ in range(1000):
        frame = preprocessor.preprocess(mock_frame.copy())
        tracks = tracker.update(frame, mock_detections)
        monitor.end_frame(monitor.start_frame())

    # Get final GPU memory
    final_gpu = mock_hardware.get_gpu_memory_usage()
    gpu_increase = final_gpu - initial_gpu

    # Verify GPU memory targets
    assert gpu_increase < 500  # Max 500MB increase
    assert monitor.stats["gpu_memory_mb"] < monitor.thresholds.max_gpu_memory_mb


def test_batch_processing_speed(mock_frame, mock_detections, mock_config, mock_hardware, benchmark):
    """Benchmark batch processing performance."""
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)
    monitor = PerformanceMonitor()

    # Create batch
    frames = [mock_frame.copy() for _ in range(10)]
    detections = [mock_detections.copy() for _ in range(10)]

    def process_batch():
        start = monitor.start_batch()

        # Process batch
        processed_frames = []
        for frame, dets in zip(frames, detections):
            frame = preprocessor.preprocess(frame)
            tracks = tracker.update(frame, dets)
            processed_frames.append((frame, tracks))

        monitor.end_batch(start)
        assert len(processed_frames) == 10

    # Run benchmark
    result = benchmark(process_batch)

    # Verify batch performance targets
    assert result.stats["mean"] < 0.2  # 200ms max for batch
    assert result.stats["stddev"] / result.stats["mean"] < 0.2  # Stable timing


def test_quality_scaling(mock_frame, mock_detections, mock_config, mock_hardware):
    """Test quality scaling under load."""
    monitor = PerformanceMonitor()
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)

    # Process frames with artificial delay
    initial_quality = monitor.get_quality_level()

    for i in range(100):
        start = monitor.start_frame()

        # Add increasing delay
        time.sleep(0.01 * (i / 20))  # Gradually increase delay

        # Process frame
        frame = preprocessor.preprocess(mock_frame.copy())
        tracks = tracker.update(frame, mock_detections)

        monitor.end_frame(start)

    # Quality should have adapted
    final_quality = monitor.get_quality_level()
    assert final_quality < initial_quality

    # FPS should still be acceptable
    assert monitor.stats["fps"] >= monitor.thresholds.min_fps


def test_parallel_processing(mock_frame, mock_detections, mock_config, mock_hardware):
    """Test parallel processing performance."""
    import threading

    monitor = PerformanceMonitor()
    tracker = ObjectTracker(config=mock_config, hardware=mock_hardware)
    preprocessor = FramePreprocessor(config=mock_config, hardware=mock_hardware)

    def worker():
        for _ in range(100):
            start = monitor.start_frame()

            # Process frame
            frame = preprocessor.preprocess(mock_frame.copy())
            tracks = tracker.update(frame, mock_detections)

            monitor.end_frame(start)

    # Run multiple workers
    threads = []
    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    # Verify performance
    stats = monitor.get_performance_stats()
    assert stats["fps"] >= monitor.thresholds.min_fps * 2  # Should be faster with parallel
    assert stats["dropped_frames"] == 0  # No drops

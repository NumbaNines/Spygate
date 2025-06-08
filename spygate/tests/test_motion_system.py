"""
Tests for the motion system module.
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from sqlalchemy.orm import Session

from spygate.ml.situation_detector import SituationDetector
from spygate.services.motion_service import MotionService
from spygate.utils.hardware_monitor import HardwareMonitor
from spygate.video.motion_detector import MotionDetectionMethod
from spygate.video.motion_system import MotionSystem
from spygate.video.video_source import VideoSource
from spygate.visualization.motion_visualizer import MotionVisualizer


class MockVideoSource(VideoSource):
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.max_frames = 100

    def read(self):
        if self.frame_count >= self.max_frames:
            return None

        # Create a test frame with some motion
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self.frame_count % 2 == 0:  # Add motion every other frame
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        self.frame_count += 1
        return frame


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_hardware_monitor():
    """Create a mock hardware monitor."""
    monitor = MagicMock(spec=HardwareMonitor)
    monitor.get_performance_tier.return_value = "high"
    monitor.get_cpu_utilization.return_value = 50.0
    monitor.get_memory_usage.return_value = 60.0
    monitor.get_gpu_utilization.return_value = 40.0
    monitor.get_system_info.return_value = {
        "cpu_count": 8,
        "total_memory": 16000000000,
        "gpu_available": True,
    }
    return monitor


@pytest.fixture
def test_frame():
    """Create a test frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def motion_system(mock_db_session, mock_hardware_monitor):
    """Create a motion system instance with mocked dependencies."""
    with patch(
        "spygate.video.motion_system.HardwareMonitor",
        return_value=mock_hardware_monitor,
    ):
        system = MotionSystem(
            db_session=mock_db_session,
            frame_width=640,
            frame_height=480,
            detection_method=MotionDetectionMethod.FRAME_DIFFERENCING,
            hardware_aware=True,
            store_heatmaps=True,
            store_patterns=True,
            enable_situation_detection=True,
            enable_visualization=True,
        )
        return system


def test_motion_system_initialization(motion_system, mock_hardware_monitor):
    """Test motion system initialization."""
    assert motion_system.performance_tier == "high"
    assert motion_system.frame_count == 0
    assert motion_system.current_video_id is None
    assert motion_system.motion_detector is not None
    assert motion_system.motion_service is not None
    assert motion_system.situation_detector is not None
    assert motion_system.visualizer is not None


def test_get_detector_params(motion_system):
    """Test detector parameter selection based on hardware tier."""
    # High tier
    params = motion_system._get_detector_params()
    assert params["use_gpu"] is True
    assert params["num_threads"] == 4
    assert params["frame_skip"] == 0

    # Medium tier
    motion_system.performance_tier = "medium"
    params = motion_system._get_detector_params()
    assert params["use_gpu"] is True
    assert params["num_threads"] == 2
    assert params["frame_skip"] == 1

    # Low tier
    motion_system.performance_tier = "low"
    params = motion_system._get_detector_params()
    assert params["use_gpu"] is False
    assert params["num_threads"] == 1
    assert params["frame_skip"] == 2


def test_process_frame(motion_system, test_frame):
    """Test frame processing pipeline."""
    # Mock component responses
    motion_result = {
        "motion_detected": True,
        "motion_mask": np.ones((480, 640), dtype=np.uint8),
        "bounding_boxes": [(100, 100, 50, 50)],
        "score": 0.8,
    }
    motion_system.motion_detector.detect_motion.return_value = motion_result

    situations = {
        "frame_number": 1,
        "timestamp": 0.033,
        "situations": [
            {"type": "motion", "confidence": 0.9, "details": {"region": "center"}}
        ],
    }
    motion_system.situation_detector.detect_situations.return_value = situations

    vis_frame = test_frame.copy()
    cv2.rectangle(vis_frame, (100, 100), (150, 150), (0, 255, 0), 2)
    motion_system.visualizer.update.return_value = vis_frame

    # Process frame
    frame_out, results = motion_system.process_frame(
        frame=test_frame, video_id=1, frame_number=1, fps=30.0
    )

    # Verify pipeline execution
    assert motion_system.motion_detector.detect_motion.called
    assert motion_system.situation_detector.detect_situations.called
    assert motion_system.motion_service.process_frame.called
    assert motion_system.visualizer.update.called

    # Verify results
    assert frame_out is not None
    assert results["hardware"]["tier"] == "high"
    assert "motion" in results
    assert "situations" in results


def test_handle_video_change(motion_system):
    """Test handling of video changes."""
    motion_system._handle_video_change(video_id=1)
    assert motion_system.current_video_id == 1
    assert motion_system.frame_count == 0

    # Verify component resets
    assert motion_system.motion_detector.reset.called
    assert motion_system.situation_detector.initialize.called
    assert motion_system.visualizer.reset.called


def test_check_resources(motion_system, mock_hardware_monitor):
    """Test resource monitoring and adaptation."""
    # Normal resource usage
    motion_system._check_resources()
    assert not motion_system.motion_detector.frame_skip

    # High resource usage
    mock_hardware_monitor.get_cpu_utilization.return_value = 95.0
    motion_system._check_resources()
    assert motion_system.motion_detector.frame_skip > 0


def test_get_motion_events(motion_system):
    """Test retrieval of motion events."""
    events = [
        {
            "id": 1,
            "frame_number": 30,
            "timestamp": 1.0,
            "motion_score": 0.8,
            "situations": [{"type": "motion", "confidence": 0.9}],
        }
    ]
    motion_system.motion_service.get_motion_events.return_value = events

    result = motion_system.get_motion_events(
        video_id=1, start_time=0.0, end_time=2.0, min_confidence=0.6
    )

    assert motion_system.motion_service.get_motion_events.called
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["motion_score"] == 0.8


def test_get_motion_patterns(motion_system):
    """Test retrieval of motion patterns."""
    patterns = [
        {
            "id": 1,
            "pattern_type": "rapid_movement",
            "confidence": 0.85,
            "start_frame": 30,
            "end_frame": 60,
        }
    ]
    motion_system.motion_service.get_motion_patterns.return_value = patterns

    result = motion_system.get_motion_patterns(
        video_id=1, pattern_type="rapid_movement", min_confidence=0.8
    )

    assert motion_system.motion_service.get_motion_patterns.called
    assert len(result) == 1
    assert result[0]["pattern_type"] == "rapid_movement"
    assert result[0]["confidence"] == 0.85


def test_get_motion_heatmap(motion_system):
    """Test retrieval of motion heatmaps."""
    heatmap = np.random.rand(480, 640).astype(np.float32)
    motion_system.motion_service.get_motion_heatmap.return_value = heatmap

    result = motion_system.get_motion_heatmap(video_id=1, start_time=0.0, end_time=2.0)

    assert motion_system.motion_service.get_motion_heatmap.called
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640)


def test_get_system_info(motion_system, mock_hardware_monitor):
    """Test retrieval of system information."""
    info = motion_system.get_system_info()

    assert info["hardware_tier"] == "high"
    assert info["current_video"] is None
    assert info["frame_count"] == 0
    assert "cpu" in info["resource_usage"]
    assert "memory" in info["resource_usage"]
    assert "gpu" in info["resource_usage"]


def test_invalid_frame(motion_system):
    """Test handling of invalid frames."""
    with pytest.raises(ValueError):
        motion_system.process_frame(frame=None, video_id=1, frame_number=1, fps=30.0)

    with pytest.raises(ValueError):
        motion_system.process_frame(
            frame=np.array([]), video_id=1, frame_number=1, fps=30.0
        )


def test_motion_system_initialization():
    """Test motion system initialization with various configurations."""
    video_source = MockVideoSource()

    # Test default initialization
    system = MotionSystem(video_source)
    assert system.video_source == video_source
    assert system.frame_width == 640
    assert system.frame_height == 480
    assert not system.is_running
    assert system.processing_thread is None

    # Test with GPU disabled
    system = MotionSystem(video_source, use_gpu=False)
    assert not system.use_gpu

    # Test with visualization disabled
    system = MotionSystem(video_source, visualization_enabled=False)
    assert not system.visualization_enabled
    assert system.visualizer is None


def test_motion_system_start_stop():
    """Test starting and stopping the motion system."""
    video_source = MockVideoSource()
    system = MotionSystem(video_source)

    # Test start
    system.start()
    assert system.is_running
    assert system.processing_thread is not None
    assert system.processing_thread.is_alive()

    # Test double start
    system.start()  # Should not create new thread
    assert len([t for t in threading.enumerate() if t.name.startswith("Thread")]) == 1

    # Test stop
    system.stop()
    assert not system.is_running
    assert not system.processing_thread.is_alive()

    # Wait for thread to fully stop
    time.sleep(0.1)
    assert len([t for t in threading.enumerate() if t.name.startswith("Thread")]) == 0


def test_motion_detection_processing():
    """Test motion detection processing pipeline."""
    video_source = MockVideoSource()
    system = MotionSystem(
        video_source, detection_method=MotionDetectionMethod.FRAME_DIFF
    )

    # Add mock alert callback
    alert_called = False

    def alert_callback(result):
        nonlocal alert_called
        alert_called = True

    system.add_alert_callback(alert_callback)

    # Start system and process some frames
    system.start()
    time.sleep(0.5)  # Allow time for processing

    # Check that frames were processed
    frame, result = system.get_next_frame()
    assert frame is not None
    assert result is not None
    assert result.motion_detected  # Should detect motion in test frames
    assert alert_called  # Alert should have been triggered

    # Stop system
    system.stop()


def test_database_integration():
    """Test motion detection results storage in database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = str(Path(temp_dir) / "test.db")
        video_source = MockVideoSource()

        system = MotionSystem(
            video_source,
            db_path=db_path,
            detection_method=MotionDetectionMethod.FRAME_DIFF,
        )

        # Process some frames
        system.start()
        time.sleep(0.5)
        system.stop()

        # Check database contents
        assert Path(db_path).exists()
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verify motion_data table exists and has entries
        cursor.execute("SELECT COUNT(*) FROM motion_data")
        count = cursor.fetchone()[0]
        assert count > 0

        # Verify data format
        cursor.execute("SELECT * FROM motion_data LIMIT 1")
        row = cursor.fetchone()
        assert len(row) >= 4  # timestamp, motion_detected, bounding_boxes, metadata

        # Verify JSON fields are valid
        bounding_boxes = json.loads(row[2])
        metadata = json.loads(row[3])
        assert isinstance(bounding_boxes, list)
        assert isinstance(metadata, dict)

        conn.close()


def test_performance_metrics():
    """Test performance metrics collection and reporting."""
    video_source = MockVideoSource()
    system = MotionSystem(video_source)

    # Process some frames
    system.start()
    time.sleep(0.5)

    # Get metrics
    metrics = system.get_performance_metrics()
    assert "fps" in metrics
    assert "queue_size" in metrics
    assert "gpu_enabled" in metrics
    assert "hardware_tier" in metrics
    assert "cpu_utilization" in metrics

    # Verify FPS calculation
    assert metrics["fps"] > 0
    assert isinstance(metrics["fps"], float)

    # Stop system
    system.stop()


def test_system_reset():
    """Test resetting the motion detection system."""
    video_source = MockVideoSource()
    system = MotionSystem(video_source)

    # Process some frames
    system.start()
    time.sleep(0.5)

    # Add some items to processing queue
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        system.processing_queue.put((frame, None))

    # Reset system
    system.reset()

    # Verify reset state
    assert system.processing_queue.empty()
    assert len(system.fps_buffer) == 0

    # Stop system
    system.stop()


def test_error_handling():
    """Test error handling in the motion system."""

    # Create a video source that raises an exception
    class ErrorVideoSource(VideoSource):
        def read(self):
            raise Exception("Test error")

    video_source = ErrorVideoSource()
    system = MotionSystem(video_source)

    # Start system and verify it handles errors gracefully
    system.start()
    time.sleep(0.5)

    # System should still be running despite errors
    assert system.is_running
    assert system.processing_thread.is_alive()

    # Stop system
    system.stop()

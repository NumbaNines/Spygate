"""Tests for the single object tracking functionality."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from spygate.video.object_tracker import ObjectTracker, TrackingConfig
from spygate.utils.tracking_hardware import TrackingMode
from spygate.core.hardware import HardwareDetector

# Constants for test configuration
TEST_FRAME_SIZE = (640, 480)
TEST_OBJECT_SIZE = (50, 50)
TEST_OBJECT_COLOR = (255, 255, 255)

class TestObjectTracker:
    """Test suite for the ObjectTracker class."""

    @pytest.mark.parametrize("tracker_type", ObjectTracker.SUPPORTED_TYPES)
    def test_tracker_initialization(self, tracker_type, base_config):
        """Test tracker initialization with different algorithms."""
        tracker = ObjectTracker(tracker_type=tracker_type, config=base_config)
        assert tracker.tracker_type == tracker_type
        assert tracker.is_initialized is False
        assert tracker.frame_count == 0

    def test_frame_tracking(self, object_tracker, moving_object_sequence):
        """Test basic object tracking across frames."""
        # Initialize tracker with first frame
        initial_frame = moving_object_sequence[0]
        bbox = (50, 50, 50, 50)  # Initial object position
        success = object_tracker.init(initial_frame, bbox)
        assert success, "Failed to initialize tracker"
        
        # Track object through sequence
        for i, frame in enumerate(moving_object_sequence[1:], 1):
            success, bbox = object_tracker.update(frame)
            assert success, f"Lost tracking at frame {i}"
            # Verify object position (with tolerance for tracking variations)
            expected_x = 50 + i*10  # Based on moving_object_sequence fixture
            assert abs(bbox[0] - expected_x) < 15, f"Tracking deviation too large at frame {i}"

    @pytest.mark.parametrize("quality", [0.5, 0.75, 1.0])
    def test_quality_adjustment(self, object_tracker, moving_object_sequence, quality):
        """Test tracker's quality adjustment capabilities."""
        object_tracker.set_quality(quality)
        
        # Initialize and track
        bbox = (50, 50, 50, 50)
        object_tracker.init(moving_object_sequence[0], bbox)
        
        # Track and measure performance
        start_time = cv2.getTickCount()
        for frame in moving_object_sequence[1:]:
            object_tracker.update(frame)
        end_time = cv2.getTickCount()
        
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        assert processing_time > 0, "Invalid processing time measurement"

    @pytest.mark.benchmark
    def test_tracking_performance(self, object_tracker, moving_object_sequence, benchmark):
        """Benchmark tracking performance."""
        bbox = (50, 50, 50, 50)
        object_tracker.init(moving_object_sequence[0], bbox)
        
        def track_sequence():
            for frame in moving_object_sequence[1:]:
                object_tracker.update(frame)
        
        # Run benchmark
        benchmark(track_sequence)

    def test_hardware_optimization(self, base_config):
        """Test hardware-aware optimization features."""
        # Mock hardware detection
        with patch('spygate.core.hardware.HardwareDetector') as mock_detector:
            mock_detector.has_gpu.return_value = True
            mock_detector.get_available_memory.return_value = 8192.0  # 8GB
            
            config = base_config
            config.enable_gpu = True
            config.max_memory_usage = 4096.0  # 4GB
            
            tracker = ObjectTracker(config=config)
            assert tracker.is_gpu_enabled
            assert tracker.max_memory_usage == 4096.0

    def test_memory_management(self, object_tracker, moving_object_sequence):
        """Test memory management and cleanup."""
        bbox = (50, 50, 50, 50)
        object_tracker.init(moving_object_sequence[0], bbox)
        
        # Track objects and monitor memory
        for frame in moving_object_sequence:
            object_tracker.update(frame)
            assert object_tracker.get_memory_usage() <= object_tracker.max_memory_usage
        
        # Force cleanup
        object_tracker.cleanup()
        assert object_tracker.get_memory_usage() < object_tracker.max_memory_usage / 2

    @pytest.mark.parametrize("frame_size", [(640, 480), (1280, 720), (1920, 1080)])
    def test_resolution_handling(self, base_config, frame_factory, frame_size):
        """Test tracker's ability to handle different resolutions."""
        tracker = ObjectTracker(config=base_config)
        
        # Create frame with specified resolution
        frame = frame_factory(size=frame_size)
        bbox = (frame_size[0]//4, frame_size[1]//4, 50, 50)
        
        success = tracker.init(frame, bbox)
        assert success, f"Failed to initialize tracker with resolution {frame_size}"

    def test_error_handling(self, object_tracker):
        """Test tracker's error handling capabilities."""
        # Test invalid frame
        with pytest.raises(ValueError):
            object_tracker.update(None)
        
        # Test invalid bbox
        with pytest.raises(ValueError):
            object_tracker.init(np.zeros(TEST_FRAME_SIZE + (3,), dtype=np.uint8), (-1, -1, 50, 50))
        
        # Test update without initialization
        with pytest.raises(RuntimeError):
            object_tracker.update(np.zeros(TEST_FRAME_SIZE + (3,), dtype=np.uint8))

    def test_reset_functionality(self, object_tracker, moving_object_sequence):
        """Test tracker reset capabilities."""
        # Initialize and track
        bbox = (50, 50, 50, 50)
        object_tracker.init(moving_object_sequence[0], bbox)
        object_tracker.update(moving_object_sequence[1])
        
        # Reset tracker
        object_tracker.reset()
        assert object_tracker.is_initialized is False
        assert object_tracker.frame_count == 0
        
        # Verify can initialize again
        success = object_tracker.init(moving_object_sequence[0], bbox)
        assert success, "Failed to initialize tracker after reset"

    @pytest.mark.parametrize("prefetch_size", [2, 4, 8])
    def test_frame_prefetching(self, base_config, moving_object_sequence, prefetch_size):
        """Test frame prefetching functionality."""
        config = base_config
        config.enable_prefetch = True
        config.prefetch_size = prefetch_size
        
        tracker = ObjectTracker(config=config)
        bbox = (50, 50, 50, 50)
        
        # Initialize and track with prefetching
        tracker.init(moving_object_sequence[0], bbox)
        for frame in moving_object_sequence[1:]:
            success, _ = tracker.update(frame)
            assert success, "Tracking failed with prefetching enabled"

    def test_prediction_generation(self, object_tracker, moving_object_sequence):
        """Test object position prediction capabilities."""
        bbox = (50, 50, 50, 50)
        object_tracker.init(moving_object_sequence[0], bbox)
        
        # Track for a few frames to establish motion pattern
        positions = []
        for frame in moving_object_sequence[1:4]:
            success, bbox = object_tracker.update(frame)
            assert success
            positions.append(bbox)
        
        # Generate prediction
        predicted_bbox = object_tracker.predict_next_position()
        assert predicted_bbox is not None
        assert len(predicted_bbox) == 4
        
        # Verify prediction is reasonable (within expected range)
        last_x = positions[-1][0]
        predicted_x = predicted_bbox[0]
        movement_direction = predicted_x - last_x
        assert movement_direction > 0, "Prediction should follow object movement direction"

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 
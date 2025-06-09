"""Tests for the multi-object tracking functionality."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from spygate.utils.tracking_hardware import TrackingMode
from spygate.video.object_tracker import MultiObjectTracker

# Constants for test configuration
TEST_FRAME_SIZE = (640, 480)
TEST_OBJECT_SIZE = (50, 50)
TEST_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class TestMultiObjectTracker:
    """Test suite for the MultiObjectTracker class."""

    def test_initialization(self, multi_tracker):
        """Test multi-object tracker initialization."""
        assert multi_tracker.tracker_type == "KCF"
        assert multi_tracker.max_lost_frames == 30
        assert multi_tracker.iou_threshold == 0.3
        assert multi_tracker.reidentify_threshold == 0.7
        assert len(multi_tracker.tracked_objects) == 0

    def test_basic_tracking(self, multi_tracker, complex_scene_sequence):
        """Test basic multi-object tracking functionality."""
        # Initialize with first frame objects
        initial_objects = [
            (50, 50, 50, 50),  # Object 1
            (200, 200, 50, 50),  # Object 2
            (350, 350, 50, 50),  # Object 3
        ]

        for bbox in initial_objects:
            success = multi_tracker.add_object(complex_scene_sequence[0], bbox)
            assert success, "Failed to add object for tracking"

        # Track objects through sequence
        for i, frame in enumerate(complex_scene_sequence[1:], 1):
            tracked_objects = multi_tracker.update(frame)
            assert len(tracked_objects) == 3, f"Lost objects at frame {i}"

            # Verify object positions (with tolerance)
            expected_positions = [
                (50 + i * 10, 50),  # Object 1: Moving right
                (200, 200 + i * 10),  # Object 2: Moving down
                (350 + i * 7, 350 + i * 7),  # Object 3: Moving diagonal
            ]

            for obj, expected in zip(tracked_objects, expected_positions):
                assert (
                    abs(obj.bbox[0] - expected[0]) < 15
                ), f"Object tracking deviation too large at frame {i}"
                assert (
                    abs(obj.bbox[1] - expected[1]) < 15
                ), f"Object tracking deviation too large at frame {i}"

    def test_occlusion_handling(self, multi_tracker, occlusion_sequence):
        """Test handling of object occlusions."""
        # Initialize objects before occlusion
        initial_objects = [(100, 100, 50, 50), (200, 100, 50, 50)]  # Object 1  # Object 2

        for bbox in initial_objects:
            success = multi_tracker.add_object(occlusion_sequence[0], bbox)
            assert success

        object_states = []
        # Track through occlusion sequence
        for frame in occlusion_sequence[1:]:
            tracked_objects = multi_tracker.update(frame)
            object_states.append(
                {
                    "count": len(tracked_objects),
                    "positions": [(obj.bbox[0], obj.bbox[1]) for obj in tracked_objects],
                }
            )

        # Verify tracking through occlusion
        assert object_states[2]["count"] >= 1, "All objects lost during occlusion"
        assert object_states[-1]["count"] == 2, "Objects not recovered after occlusion"

    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9])
    def test_reidentification(self, multi_tracker, moving_object_sequence, threshold):
        """Test object reidentification with different thresholds."""
        multi_tracker.reidentify_threshold = threshold

        # Initialize with single object
        bbox = (50, 50, 50, 50)
        multi_tracker.add_object(moving_object_sequence[0], bbox)

        # Track, simulate loss, and attempt reidentification
        tracked_objects = multi_tracker.update(moving_object_sequence[1])
        original_id = tracked_objects[0].id

        # Simulate temporary occlusion
        multi_tracker.update(np.zeros_like(moving_object_sequence[0]))  # Empty frame

        # Attempt reidentification
        tracked_objects = multi_tracker.update(moving_object_sequence[3])
        if threshold < 0.8:  # Expected successful reidentification
            assert len(tracked_objects) == 1
            assert tracked_objects[0].id == original_id
        else:  # Expected new object creation
            assert len(tracked_objects) <= 1

    def test_velocity_prediction(self, multi_tracker, complex_scene_sequence):
        """Test velocity-based position prediction."""
        # Initialize objects
        initial_objects = [(50, 50, 50, 50), (200, 200, 50, 50)]

        for bbox in initial_objects:
            multi_tracker.add_object(complex_scene_sequence[0], bbox)

        # Track for several frames to establish velocity
        for frame in complex_scene_sequence[1:4]:
            tracked_objects = multi_tracker.update(frame)

        # Get predictions
        predictions = multi_tracker.predict_object_positions()
        assert len(predictions) == 2

        # Verify predictions against actual next positions
        next_objects = multi_tracker.update(complex_scene_sequence[4])
        for pred, actual in zip(predictions, next_objects):
            assert abs(pred[0] - actual.bbox[0]) < 20
            assert abs(pred[1] - actual.bbox[1]) < 20

    @pytest.mark.benchmark
    def test_tracking_performance(self, multi_tracker, complex_scene_sequence, benchmark):
        """Benchmark multi-object tracking performance."""
        # Initialize with multiple objects
        initial_objects = [(50, 50, 50, 50), (200, 200, 50, 50), (350, 350, 50, 50)]

        for bbox in initial_objects:
            multi_tracker.add_object(complex_scene_sequence[0], bbox)

        def track_sequence():
            for frame in complex_scene_sequence[1:]:
                multi_tracker.update(frame)

        # Run benchmark
        benchmark(track_sequence)

    def test_lost_object_handling(self, multi_tracker, complex_scene_sequence):
        """Test handling of lost objects."""
        # Initialize tracker
        bbox = (50, 50, 50, 50)
        multi_tracker.add_object(complex_scene_sequence[0], bbox)

        # Track normally
        multi_tracker.update(complex_scene_sequence[1])

        # Simulate object loss with empty frames
        for _ in range(multi_tracker.max_lost_frames - 1):
            tracked_objects = multi_tracker.update(np.zeros_like(complex_scene_sequence[0]))
            assert len(tracked_objects) == 1, "Object removed too early"

        # Object should be removed after max_lost_frames
        tracked_objects = multi_tracker.update(np.zeros_like(complex_scene_sequence[0]))
        assert len(tracked_objects) == 0, "Object not removed after maximum lost frames"

    def test_multiple_similar_objects(self, multi_tracker, frame_factory):
        """Test tracking of multiple similar objects."""
        # Create sequence with similar objects
        frames = []
        for i in range(5):
            objects = [
                (100 + i * 10, 100, 50, 50, (255, 255, 255)),
                (100 + i * 10, 200, 50, 50, (255, 255, 255)),
            ]
            frames.append(frame_factory(objects=objects))

        # Initialize tracking
        initial_objects = [(100, 100, 50, 50), (100, 200, 50, 50)]
        for bbox in initial_objects:
            multi_tracker.add_object(frames[0], bbox)

        # Track and verify distinct IDs maintained
        tracked_objects = multi_tracker.update(frames[1])
        initial_ids = {obj.id for obj in tracked_objects}

        for frame in frames[2:]:
            tracked_objects = multi_tracker.update(frame)
            current_ids = {obj.id for obj in tracked_objects}
            assert current_ids == initial_ids, "Object IDs not maintained for similar objects"

    def test_error_handling(self, multi_tracker):
        """Test error handling in multi-object tracker."""
        # Test invalid frame
        with pytest.raises(ValueError):
            multi_tracker.update(None)

        # Test invalid bbox
        with pytest.raises(ValueError):
            multi_tracker.add_object(
                np.zeros(TEST_FRAME_SIZE + (3,), dtype=np.uint8), (-1, -1, 50, 50)
            )

        # Test invalid threshold values
        with pytest.raises(ValueError):
            multi_tracker.iou_threshold = -0.1
        with pytest.raises(ValueError):
            multi_tracker.reidentify_threshold = 1.5

    def test_reset_functionality(self, multi_tracker, complex_scene_sequence):
        """Test tracker reset capabilities."""
        # Initialize with multiple objects
        initial_objects = [(50, 50, 50, 50), (200, 200, 50, 50)]
        for bbox in initial_objects:
            multi_tracker.add_object(complex_scene_sequence[0], bbox)

        # Track for a few frames
        multi_tracker.update(complex_scene_sequence[1])

        # Reset tracker
        multi_tracker.reset()
        assert len(multi_tracker.tracked_objects) == 0
        assert len(multi_tracker.lost_objects) == 0

        # Verify can initialize again
        success = multi_tracker.add_object(complex_scene_sequence[0], initial_objects[0])
        assert success, "Failed to add object after reset"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

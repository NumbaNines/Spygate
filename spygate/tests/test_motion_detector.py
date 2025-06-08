"""
Tests for the motion detection module.
"""

import threading

import cv2
import numpy as np
import pytest

from spygate.video.motion_detector import (
    MotionDetectionMethod,
    MotionDetectionResult,
    MotionDetector,
)


def create_test_frame(
    width: int = 640, height: int = 480, color: tuple = (0, 0, 0)
) -> np.ndarray:
    """Create a test frame with specified dimensions and color."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_moving_object_frame(
    width: int = 640,
    height: int = 480,
    rect_pos: tuple = (100, 100),
    rect_size: tuple = (50, 50),
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Create a test frame with a rectangle at specified position."""
    frame = create_test_frame(width, height)
    x, y = rect_pos
    w, h = rect_size
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    return frame


class TestMotionDetector:
    """Test suite for MotionDetector class."""

    def test_initialization(self):
        """Test MotionDetector initialization with default parameters."""
        detector = MotionDetector()
        assert detector.method == MotionDetectionMethod.FRAME_DIFFERENCING
        assert detector.threshold == 30
        assert detector.min_area == 500
        assert detector.blur_size % 2 == 1  # Should be odd
        assert detector.prev_frame is None
        assert detector.prev_gray is None
        assert detector.learning_rate == 0.01

    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        detector = MotionDetector()
        frame = create_test_frame()
        processed = detector.preprocess_frame(frame)

        assert processed.shape == (480, 640)  # Should be grayscale
        assert processed.dtype == np.uint8

    def test_no_motion_detection(self):
        """Test motion detection with identical frames."""
        detector = MotionDetector()
        frame = create_test_frame()

        # First frame should return no motion
        result1 = detector.detect_motion(frame)
        assert result1.motion_score == 0.0
        assert len(result1.bounding_boxes) == 0

        # Second identical frame should also return no motion
        result2 = detector.detect_motion(frame)
        assert result2.motion_score == 0.0
        assert len(result2.bounding_boxes) == 0

    def test_motion_detection(self):
        """Test motion detection with moving object."""
        detector = MotionDetector(min_area=100)  # Lower min_area for test

        # First frame with object at position 1
        frame1 = create_moving_object_frame(rect_pos=(100, 100))
        result1 = detector.detect_motion(frame1)

        # Second frame with object at position 2
        frame2 = create_moving_object_frame(rect_pos=(150, 150))
        result2 = detector.detect_motion(frame2)

        # Should detect motion
        assert result2.motion_score > 0.0
        assert len(result2.bounding_boxes) > 0

        # Motion mask should be binary
        assert np.array_equal(
            result2.motion_mask, result2.motion_mask.astype(bool) * 255
        )

    def test_reset(self):
        """Test detector reset."""
        detector = MotionDetector()
        frame = create_test_frame()

        # Process a frame
        detector.detect_motion(frame)
        assert detector.prev_gray is not None

        # Reset detector
        detector.reset()
        assert detector.prev_frame is None
        assert detector.prev_gray is None

    def test_invalid_method(self):
        """Test handling of unimplemented detection methods."""
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        frame = create_test_frame()

        with pytest.raises(NotImplementedError):
            detector.detect_motion(frame)

    def test_motion_result_metadata(self):
        """Test motion detection result metadata."""
        detector = MotionDetector()
        frame = create_test_frame()
        result = detector.detect_motion(frame)

        assert "method" in result.metadata
        assert (
            result.metadata["method"] == MotionDetectionMethod.FRAME_DIFFERENCING.value
        )
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == detector.threshold

    def test_large_motion_detection(self):
        """Test detection of large motion areas."""
        detector = MotionDetector(min_area=100)

        # Create frames with large moving object
        frame1 = create_moving_object_frame(rect_size=(200, 200))
        frame2 = create_moving_object_frame(rect_pos=(50, 50), rect_size=(200, 200))

        # First frame
        detector.detect_motion(frame1)

        # Second frame should detect large motion
        result = detector.detect_motion(frame2)
        assert result.motion_score > 0.1  # Significant motion
        assert len(result.bounding_boxes) > 0

        # Check bounding box size
        x, y, w, h = result.bounding_boxes[0]
        assert w >= 150 and h >= 150  # Should detect large area

    def test_background_subtraction_initialization(self):
        """Test initialization with background subtraction method."""
        detector = MotionDetector(method=MotionDetectionMethod.BACKGROUND_SUBTRACTION)
        assert detector.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION
        assert detector.bg_subtractor is not None
        assert detector.learning_rate == 0.01

    def test_background_subtraction_no_motion(self):
        """Test background subtraction with static scene."""
        detector = MotionDetector(method=MotionDetectionMethod.BACKGROUND_SUBTRACTION)
        frame = create_test_frame()

        # Process multiple frames to build background model
        for _ in range(10):
            result = detector.detect_motion(frame)

        # Static scene should have minimal motion
        assert result.motion_score < 0.1
        assert len(result.bounding_boxes) == 0

    def test_background_subtraction_motion(self):
        """Test background subtraction with moving object."""
        detector = MotionDetector(
            method=MotionDetectionMethod.BACKGROUND_SUBTRACTION, min_area=100
        )

        # Process multiple frames with static background
        static_frame = create_test_frame()
        for _ in range(10):
            detector.detect_motion(static_frame)

        # Introduce moving object
        moving_frame = create_moving_object_frame()
        result = detector.detect_motion(moving_frame)

        # Should detect motion
        assert result.motion_score > 0.0
        assert len(result.bounding_boxes) > 0

        # Check metadata
        assert (
            result.metadata["method"]
            == MotionDetectionMethod.BACKGROUND_SUBTRACTION.value
        )
        assert result.metadata["learning_rate"] == detector.learning_rate

    def test_background_subtraction_learning_rate(self):
        """Test background subtraction with different learning rates."""
        # Fast learning rate
        fast_detector = MotionDetector(
            method=MotionDetectionMethod.BACKGROUND_SUBTRACTION, learning_rate=0.1
        )

        # Slow learning rate
        slow_detector = MotionDetector(
            method=MotionDetectionMethod.BACKGROUND_SUBTRACTION, learning_rate=0.001
        )

        # Process frames with moving object
        for _ in range(5):
            frame = create_moving_object_frame(rect_pos=(100 + _ * 10, 100))
            fast_result = fast_detector.detect_motion(frame)
            slow_result = slow_detector.detect_motion(frame)

            # Fast learner should adapt quicker (lower motion scores)
            assert fast_result.motion_score <= slow_result.motion_score

    def test_background_subtraction_reset(self):
        """Test resetting background subtraction state."""
        detector = MotionDetector(method=MotionDetectionMethod.BACKGROUND_SUBTRACTION)

        # Process some frames
        for _ in range(5):
            frame = create_moving_object_frame(rect_pos=(100 + _ * 10, 100))
            detector.detect_motion(frame)

        # Reset detector
        detector.reset()

        # Process new frame
        frame = create_test_frame()
        result = detector.detect_motion(frame)

        # Should behave like first frame
        assert result.motion_score < 0.1  # Minimal motion detected
        assert len(result.bounding_boxes) == 0

    def test_optical_flow_initialization(self):
        """Test optical flow detector initialization."""
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        assert detector.method == MotionDetectionMethod.OPTICAL_FLOW
        assert detector.max_corners == 100
        assert detector.quality_level == 0.3
        assert detector.min_distance == 7
        assert detector.block_size == 7
        assert detector.prev_points is None
        assert detector.prev_gray is None

    def test_optical_flow_no_motion(self):
        """Test optical flow with static frame."""
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        frame = create_test_frame(motion=False)

        # First frame should initialize tracking points
        result1 = detector.detect_motion(frame)
        assert result1.motion_score == 0.0
        assert len(result1.bounding_boxes) == 0
        assert detector.prev_points is not None

        # Second identical frame should detect no motion
        result2 = detector.detect_motion(frame)
        assert result2.motion_score == 0.0
        assert len(result2.bounding_boxes) == 0
        assert "num_tracked_points" in result2.metadata

    def test_optical_flow_with_motion(self):
        """Test optical flow with moving object."""
        detector = MotionDetector(
            method=MotionDetectionMethod.OPTICAL_FLOW,
            min_area=100,  # Lower min_area for test
            threshold=5,  # Lower threshold for test
        )

        # First frame with object at position 1
        frame1 = create_moving_object_frame(rect_pos=(100, 100))
        result1 = detector.detect_motion(frame1)
        assert detector.prev_points is not None

        # Second frame with object at position 2
        frame2 = create_moving_object_frame(rect_pos=(150, 150))
        result2 = detector.detect_motion(frame2)

        # Should detect motion
        assert result2.motion_score > 0.0
        assert len(result2.bounding_boxes) > 0
        assert result2.metadata["num_tracked_points"] > 0

        # Motion mask should be binary
        assert np.array_equal(
            result2.motion_mask, result2.motion_mask.astype(bool) * 255
        )

    def test_optical_flow_reset(self):
        """Test optical flow detector reset."""
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        frame = create_test_frame(motion=True)

        # Process a frame to initialize state
        detector.detect_motion(frame)
        assert detector.prev_points is not None
        assert detector.prev_gray is not None

        # Reset detector
        detector.reset()
        assert detector.prev_points is None
        assert detector.prev_gray is None

    def test_optical_flow_low_quality_frame(self):
        """Test optical flow with low quality frame (few features)."""
        detector = MotionDetector(
            method=MotionDetectionMethod.OPTICAL_FLOW,
            quality_level=0.5,  # Higher quality threshold
        )

        # Create a mostly blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(
            frame, (10, 10), (30, 30), (255, 255, 255), -1
        )  # Small white rectangle

        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)
        assert result.motion_score == 0.0
        assert len(result.bounding_boxes) == 0

    def test_optical_flow_rapid_motion(self):
        """Test optical flow with rapid motion (large displacement)."""
        detector = MotionDetector(
            method=MotionDetectionMethod.OPTICAL_FLOW, min_area=100, threshold=5
        )

        # First frame with object at position 1
        frame1 = create_moving_object_frame(rect_pos=(100, 100))
        detector.detect_motion(frame1)

        # Second frame with object at far position (rapid motion)
        frame2 = create_moving_object_frame(rect_pos=(400, 400))
        result = detector.detect_motion(frame2)

        # Should still track some points despite large motion
        assert isinstance(result, MotionDetectionResult)
        assert "num_tracked_points" in result.metadata

    def test_detect_motion_integration(self):
        """Test that detect_motion correctly routes to each detection method."""
        frame = create_test_frame(motion=True)

        # Test frame differencing
        detector = MotionDetector(method=MotionDetectionMethod.FRAME_DIFF)
        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)

        # Test background subtraction
        detector = MotionDetector(method=MotionDetectionMethod.BACKGROUND_SUB)
        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)

        # Test optical flow
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)

    def test_detect_motion_invalid_frame(self):
        """Test that detect_motion handles invalid frames properly."""
        detector = MotionDetector()

        # Test None frame
        with pytest.raises(ValueError, match="Invalid frame provided"):
            detector.detect_motion(None)

        # Test empty frame
        with pytest.raises(ValueError, match="Invalid frame provided"):
            detector.detect_motion(np.array([]))

    def test_reset_all_methods(self):
        """Test that reset properly clears state for all detection methods."""
        detector = MotionDetector(method=MotionDetectionMethod.OPTICAL_FLOW)
        frame = create_test_frame(motion=True)

        # Create some state
        detector.detect_motion(frame)
        assert detector.prev_frame is not None
        assert detector.prev_gray is not None
        assert detector.bg_subtractor is not None

        # Reset should clear all state
        detector.reset()
        assert detector.prev_frame is None
        assert detector.prev_gray is None
        assert detector.prev_points is None
        assert detector.bg_subtractor is not None  # Should be recreated, not None

    def test_gpu_acceleration(self):
        """Test GPU acceleration when available."""
        # Test with GPU enabled
        detector = MotionDetector(use_gpu=True)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            assert detector.use_gpu is True
            assert hasattr(detector, "gpu_stream")
            assert hasattr(detector, "gpu_detector")
        else:
            assert detector.use_gpu is False
            assert not hasattr(detector, "gpu_stream")

        # Test with GPU disabled
        detector = MotionDetector(use_gpu=False)
        assert detector.use_gpu is False
        assert not hasattr(detector, "gpu_stream")

    def test_multi_threading(self):
        """Test multi-threaded processing."""
        # Test with multiple threads
        num_threads = 4
        detector = MotionDetector(num_threads=num_threads)
        assert detector.num_threads == num_threads
        assert detector.thread_pool._max_workers == num_threads

        # Process a frame to test thread-local storage
        frame = create_test_frame(motion=True)
        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)

        # Test with single thread
        detector = MotionDetector(num_threads=1)
        assert detector.num_threads == 1
        assert detector.thread_pool._max_workers == 1

    def test_thread_local_subtractor(self):
        """Test thread-local background subtractor creation."""
        detector = MotionDetector(method=MotionDetectionMethod.BACKGROUND_SUB)

        # Get subtractor in main thread
        subtractor1 = detector._get_thread_local_subtractor()
        assert subtractor1 is not None

        # Get subtractor again in main thread (should be same instance)
        subtractor2 = detector._get_thread_local_subtractor()
        assert subtractor2 is subtractor1

        # Test in a different thread
        def get_subtractor_in_thread():
            subtractor = detector._get_thread_local_subtractor()
            assert subtractor is not None
            assert subtractor is not subtractor1

        thread = threading.Thread(target=get_subtractor_in_thread)
        thread.start()
        thread.join()

    def test_reset_with_gpu(self):
        """Test reset functionality with GPU acceleration."""
        detector = MotionDetector(use_gpu=True)
        frame = create_test_frame(motion=True)

        # Process a frame to initialize state
        detector.detect_motion(frame)

        # Reset detector
        detector.reset()

        # Verify all state is properly reset
        assert detector.prev_frame is None
        assert detector.prev_gray is None
        assert detector.prev_points is None

        if detector.use_gpu:
            assert hasattr(detector, "gpu_detector")
        else:
            assert hasattr(detector, "bg_subtractor")

        # Process another frame after reset
        result = detector.detect_motion(frame)
        assert isinstance(result, MotionDetectionResult)

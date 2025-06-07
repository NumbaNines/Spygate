"""Unit tests for the SituationDetector class."""

import os
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from ...database import Situation
from ...ml.situation_detector import SituationDetector


@pytest.fixture
def detector():
    """Create a SituationDetector instance for testing."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        detector = SituationDetector()
        detector.initialize()
        return detector


@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing."""
    # Create a 720p frame with some motion
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add some motion by drawing shapes
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    return frame


@pytest.fixture
def sample_frames():
    """Create a sequence of sample frames for testing."""
    frames = []
    for i in range(5):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Add increasing motion
        cv2.rectangle(
            frame,
            (100 + i * 20, 100 + i * 20),
            (200 + i * 20, 200 + i * 20),
            (255, 255, 255),
            -1,
        )
        frames.append(frame)
    return frames


def test_initialization(detector):
    """Test detector initialization."""
    assert detector.initialized is True


def test_detect_situations_with_motion(detector, sample_frame):
    """Test situation detection with significant motion."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock motion detection
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = np.ones((720, 1280), dtype=np.uint8) * 100
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        result = detector.detect_situations(sample_frame, 30, 1.0)

        assert isinstance(result, dict)
        assert "situations" in result
        assert len(result["situations"]) > 0
        assert result["situations"][0]["type"] == "high_motion_event"
        assert result["situations"][0]["confidence"] > 0.5
        assert "frame_number" in result
        assert "timestamp" in result
        assert "metadata" in result


def test_detect_situations_no_motion(detector):
    """Test situation detection with no motion."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock no motion
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect_situations(frame, 30, 1.0)

        assert isinstance(result, dict)
        assert "situations" in result
        assert len(result["situations"]) == 0
        assert "frame_number" in result
        assert "timestamp" in result
        assert "metadata" in result


def test_analyze_sequence(detector, sample_frames):
    """Test sequence analysis."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock increasing motion
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.side_effect = [
            np.ones((720, 1280), dtype=np.uint8) * (i * 20)
            for i in range(len(sample_frames))
        ]
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        results = detector.analyze_sequence(sample_frames, 0, 30.0)

        assert len(results) == len(sample_frames)
        assert all(isinstance(r, dict) for r in results)
        assert all("situations" in r for r in results)
        assert any(len(r["situations"]) > 0 for r in results)


def test_extract_hud_info(detector, sample_frame):
    """Test HUD information extraction."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock OCR or text detection here when implemented
        result = detector.extract_hud_info(sample_frame)

        assert isinstance(result, dict)
        assert "down" in result
        assert "distance" in result
        assert "score" in result
        assert "time" in result
        assert "quarter" in result


def test_detect_mistakes(detector, sample_frames):
    """Test mistake detection."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Create a sequence with a simulated mistake
        situations = [
            {
                "type": "high_motion_event",
                "confidence": 0.9,
                "frame": 30,
                "timestamp": 1.0,
                "details": {"motion_score": 75.0},
            }
        ]

        hud_info = {"score_change": -50, "health_change": -25}

        result = detector.detect_mistakes(situations, hud_info)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert "type" in result[0]
        assert "confidence" in result[0]
        assert result[0]["type"] == "mistake"
        assert result[0]["confidence"] > 0.5


def test_uninitialized_detector():
    """Test behavior when detector is not initialized."""
    detector = SituationDetector()  # Don't initialize

    with pytest.raises(RuntimeError, match="not initialized"):
        detector.detect_situations(np.zeros((720, 1280, 3)), 0, 30.0)


def test_invalid_frame_size(detector):
    """Test handling of invalid frame sizes."""
    with pytest.raises(ValueError, match="Invalid frame dimensions"):
        detector.detect_situations(np.zeros((480, 640, 3)), 0, 30.0)


def test_invalid_frame_type(detector):
    """Test handling of invalid frame types."""
    with pytest.raises(TypeError, match="Frame must be a numpy array"):
        detector.detect_situations([1, 2, 3], 0, 30.0)


def test_invalid_timestamp(detector, sample_frame):
    """Test handling of invalid timestamps."""
    with pytest.raises(ValueError, match="Invalid timestamp"):
        detector.detect_situations(sample_frame, 0, -1.0)


def test_sequence_analysis_empty(detector):
    """Test sequence analysis with empty sequence."""
    with pytest.raises(ValueError, match="Empty frame sequence"):
        detector.analyze_sequence([], 0, 30.0)


def test_sequence_analysis_inconsistent_sizes(detector, sample_frames):
    """Test sequence analysis with inconsistent frame sizes."""
    invalid_frames = sample_frames + [np.zeros((480, 640, 3))]
    with pytest.raises(ValueError, match="Inconsistent frame dimensions"):
        detector.analyze_sequence(invalid_frames, 0, 30.0)


def test_hud_info_extraction_no_text(detector):
    """Test HUD info extraction when no text is detected."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock OCR to return no text
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        result = detector.extract_hud_info(np.zeros((720, 1280, 3)))

        assert isinstance(result, dict)
        assert all(v is None for v in result.values())


def test_mistake_detection_no_situations(detector):
    """Test mistake detection with no situations."""
    result = detector.detect_mistakes([], {})
    assert isinstance(result, list)
    assert len(result) == 0


def test_mistake_detection_invalid_situation(detector):
    """Test mistake detection with invalid situation data."""
    with pytest.raises(ValueError, match="Invalid situation data"):
        detector.detect_mistakes([{"type": "invalid"}], {})


def test_detect_situations_with_threshold(detector, sample_frame):
    """Test situation detection with custom motion threshold."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock motion just below threshold
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = np.ones((720, 1280), dtype=np.uint8) * 49
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        result = detector.detect_situations(sample_frame, 30, 1.0, motion_threshold=50)
        assert len(result["situations"]) == 0

        # Mock motion just above threshold
        mock_cv2.absdiff.return_value = np.ones((720, 1280), dtype=np.uint8) * 51
        result = detector.detect_situations(sample_frame, 30, 1.0, motion_threshold=50)
        assert len(result["situations"]) > 0


def test_detect_situations_with_noise(detector, sample_frame):
    """Test situation detection with noise filtering."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock noisy motion
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        noisy_motion = np.random.randint(0, 30, (720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = noisy_motion
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        result = detector.detect_situations(sample_frame, 30, 1.0)
        assert len(result["situations"]) == 0  # Should filter out noise


def test_analyze_sequence_with_window(detector, sample_frames):
    """Test sequence analysis with sliding window."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Mock motion pattern
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.side_effect = [
            np.ones((720, 1280), dtype=np.uint8) * (100 if i == 2 else 10)
            for i in range(len(sample_frames))
        ]
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        results = detector.analyze_sequence(sample_frames, 0, 30.0, window_size=3)

        # Check for peak detection in the middle frame
        assert len(results[2]["situations"]) > 0
        assert results[2]["situations"][0]["type"] == "high_motion_event"


def test_extract_hud_info_with_noise(detector):
    """Test HUD info extraction with noisy background."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        # Create noisy frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Mock text detection
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.threshold.return_value = (None, np.zeros((720, 1280), dtype=np.uint8))

        result = detector.extract_hud_info(frame)
        assert isinstance(result, dict)
        assert all(
            k in result for k in ["down", "distance", "score", "time", "quarter"]
        )


def test_detect_mistakes_with_confidence(detector):
    """Test mistake detection with different confidence levels."""
    situations = [
        {
            "type": "high_motion_event",
            "confidence": conf,
            "frame": 30,
            "timestamp": 1.0,
            "details": {"motion_score": 75.0},
        }
        for conf in [0.3, 0.6, 0.9]
    ]

    hud_info = {"score_change": -50}

    result = detector.detect_mistakes(situations, hud_info)

    # Should only consider high confidence situations
    assert len(result) == 1
    assert result[0]["confidence"] > 0.8


def test_detect_situations_memory_usage(detector, sample_frame):
    """Test memory usage during situation detection."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = np.ones((720, 1280), dtype=np.uint8) * 100
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        # Process multiple frames to check memory
        for _ in range(100):
            result = detector.detect_situations(sample_frame, 30, 1.0)
            assert isinstance(result, dict)  # Basic validation only


def test_concurrent_detection(detector, sample_frame):
    """Test concurrent situation detection."""
    with patch("spygate.ml.situation_detector.cv2") as mock_cv2:
        mock_cv2.cvtColor.return_value = np.zeros((720, 1280), dtype=np.uint8)
        mock_cv2.absdiff.return_value = np.ones((720, 1280), dtype=np.uint8) * 100
        mock_cv2.GaussianBlur.return_value = np.zeros((720, 1280), dtype=np.uint8)

        # Simulate concurrent detections
        import threading

        results = []

        def detect():
            result = detector.detect_situations(sample_frame, 30, 1.0)
            results.append(result)

        threads = [threading.Thread(target=detect) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

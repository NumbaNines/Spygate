"""
Tests for the player detection module.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from ..utils.tracking_hardware import TrackingMode
from ..video.player_detector import PlayerDetector


@pytest.fixture
def mock_frame():
    """Create a mock video frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_hog_detector():
    """Create a mock HOG detector."""
    detector = MagicMock()
    detector.detectMultiScale.return_value = (
        np.array([[100, 200, 50, 100]]),  # boxes
        np.array([0.8]),  # weights
    )
    return detector


@pytest.fixture
def mock_frcnn():
    """Create a mock Faster R-CNN model."""
    model = MagicMock()
    model.return_value = [
        {
            "boxes": torch.tensor([[100, 200, 150, 300]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),  # person class
        }
    ]
    return model


@pytest.fixture
def mock_yolo():
    """Create a mock YOLOv5 model."""
    model = MagicMock()
    results = MagicMock()
    results.xyxy = [torch.tensor([[100, 200, 150, 300, 0.95, 0]])]  # box, conf, class
    model.return_value = results
    return model


def test_player_detector_initialization():
    """Test PlayerDetector initialization."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_hog.return_value = MagicMock()
        detector = PlayerDetector()
        assert detector.confidence_threshold == 0.7
        assert "hog" in detector.models


def test_detect_players_hog(mock_frame, mock_hog_detector):
    """Test player detection using HOG."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_hog.return_value = mock_hog_detector

        detector = PlayerDetector()
        detections = detector.detect_players(mock_frame, method="hog")

        assert len(detections) == 1
        assert "bbox" in detections[0]
        assert "confidence" in detections[0]
        assert "class" in detections[0]
        assert detections[0]["class"] == "person"


def test_detect_players_frcnn(mock_frame, mock_frcnn):
    """Test player detection using Faster R-CNN."""
    with patch("torchvision.models.detection.fasterrcnn_resnet50_fpn") as mock_model:
        mock_model.return_value = mock_frcnn

        detector = PlayerDetector()
        detector.models["frcnn"] = mock_frcnn
        detections = detector.detect_players(mock_frame, method="frcnn")

        assert len(detections) == 1
        assert "bbox" in detections[0]
        assert "confidence" in detections[0]
        assert "class" in detections[0]
        assert detections[0]["class"] == "person"


def test_detect_players_yolo(mock_frame, mock_yolo):
    """Test player detection using YOLOv5."""
    detector = PlayerDetector()
    detector.models["yolo"] = mock_yolo
    detections = detector.detect_players(mock_frame, method="yolo")

    assert len(detections) == 1
    assert "bbox" in detections[0]
    assert "confidence" in detections[0]
    assert "class" in detections[0]
    assert detections[0]["class"] == "person"


def test_auto_method_selection():
    """Test automatic detection method selection."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_hog.return_value = MagicMock()

        detector = PlayerDetector()

        # Test BASIC mode
        detector.tracking_mode = TrackingMode.BASIC
        assert detector._select_detection_method() == "hog"

        # Test STANDARD mode
        detector.tracking_mode = TrackingMode.STANDARD
        detector.models["frcnn"] = MagicMock()
        assert detector._select_detection_method() == "frcnn"

        # Test ADVANCED mode with YOLO
        detector.tracking_mode = TrackingMode.ADVANCED
        detector.models["yolo"] = MagicMock()
        assert detector._select_detection_method() == "yolo"

        # Test ADVANCED mode without YOLO
        del detector.models["yolo"]
        assert detector._select_detection_method() == "frcnn"


def test_fallback_to_hog():
    """Test fallback to HOG when requested method is unavailable."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_hog.return_value = MagicMock()
        mock_hog.return_value.detectMultiScale.return_value = (
            np.array([]),
            np.array([]),
        )

        detector = PlayerDetector()
        detections = detector.detect_players(
            np.zeros((480, 640, 3), dtype=np.uint8), method="frcnn"
        )

        assert len(detections) == 0  # No detections in empty frame
        assert "frcnn" not in detector.models  # FRCNN not available
        assert "hog" in detector.models  # HOG is available as fallback


def test_confidence_threshold():
    """Test confidence threshold filtering."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_detector = MagicMock()
        mock_detector.detectMultiScale.return_value = (
            np.array([[100, 200, 50, 100], [300, 400, 50, 100]]),  # boxes
            np.array([0.8, 0.6]),  # weights
        )
        mock_hog.return_value = mock_detector

        detector = PlayerDetector(confidence_threshold=0.7)
        detections = detector.detect_players(
            np.zeros((480, 640, 3), dtype=np.uint8), method="hog"
        )

        assert len(detections) == 1  # Only one detection above threshold
        assert detections[0]["confidence"] >= 0.7


def test_get_detection_info():
    """Test detection info retrieval."""
    with patch("cv2.HOGDescriptor") as mock_hog:
        mock_hog.return_value = MagicMock()

        detector = PlayerDetector()
        info = detector.get_detection_info()

        assert "tracking_mode" in info
        assert "available_methods" in info
        assert "gpu_available" in info
        assert "confidence_threshold" in info
        assert info["confidence_threshold"] == 0.7
        assert "hog" in info["available_methods"]

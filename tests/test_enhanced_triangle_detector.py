"""
Unit tests for enhanced triangle detector.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.training.enhanced_triangle_detector import (
    EnhancedTriangleDetector,
    TriangleDetection,
    TriangleOrientation,
    TriangleType,
)


@pytest.fixture
def detector():
    """Create triangle detector instance."""
    return EnhancedTriangleDetector()


@pytest.fixture
def sample_image():
    """Create sample test image."""
    # Create 100x100 black image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Draw possession triangle (left side)
    pts = np.array([[20, 50], [30, 40], [30, 60]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 255))

    # Draw territory triangle pointing up (right side)
    pts = np.array([[70, 40], [80, 60], [90, 60]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 255))

    return image


def test_validate_image(detector):
    """Test image validation."""
    # Valid image
    valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
    detector._validate_image(valid_image)  # Should not raise

    # Invalid images
    with pytest.raises(ValueError):
        detector._validate_image(None)

    with pytest.raises(ValueError):
        invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2 channels
        detector._validate_image(invalid_image)

    with pytest.raises(ValueError):
        invalid_image = np.zeros((100, 100, 3), dtype=np.float32)  # Wrong dtype
        detector._validate_image(invalid_image)


def test_detect_orientation(detector, sample_image):
    """Test triangle orientation detection."""
    # Find contours in sample image
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Test territory triangle (should point up)
    territory_contour = contours[1]  # Right side triangle
    orientation = detector._detect_orientation(territory_contour, sample_image.shape[0])
    assert orientation == TriangleOrientation.UP


def test_detect_triangles(detector, sample_image):
    """Test triangle detection."""
    detections = detector.detect_triangles(sample_image)

    # Should find 2 triangles
    assert len(detections) == 2

    # Check possession triangle
    possession = next(d for d in detections if d.type == TriangleType.POSSESSION)
    assert possession.orientation is None
    assert possession.bbox[0] < sample_image.shape[1] // 2  # Left side

    # Check territory triangle
    territory = next(d for d in detections if d.type == TriangleType.TERRITORY)
    assert territory.orientation == TriangleOrientation.UP
    assert territory.bbox[0] > sample_image.shape[1] // 2  # Right side


def test_determine_field_position(detector):
    """Test field position determination."""
    # Create sample detections
    possession = TriangleDetection(
        type=TriangleType.POSSESSION,
        orientation=None,
        bbox=(20, 40, 30, 60),  # Left side = away team
        confidence=0.95,
    )

    territory = TriangleDetection(
        type=TriangleType.TERRITORY,
        orientation=TriangleOrientation.UP,  # Pointing up = opponent's territory
        bbox=(70, 40, 90, 60),
        confidence=0.95,
    )

    # Test field position determination
    position = detector.determine_field_position(possession, territory)
    assert position["possession_team"] == "away"
    assert position["in_territory"] == "opponent"
    assert position["on_offense"] is True  # Away team in opponent's territory = offense

    # Test with missing detections
    with pytest.raises(ValueError):
        detector.determine_field_position(None, territory)
    with pytest.raises(ValueError):
        detector.determine_field_position(possession, None)


def test_roi_detection(detector, sample_image):
    """Test detection with region of interest."""
    roi = (10, 30, 90, 70)  # Central region
    detections = detector.detect_triangles(sample_image, roi)

    # Should still find both triangles
    assert len(detections) == 2

    # All detections should be within ROI
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert x1 >= 0 and x2 <= 80  # Relative to ROI
        assert y1 >= 0 and y2 <= 40  # Relative to ROI

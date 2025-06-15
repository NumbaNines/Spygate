"""
Tests for the enhanced OCR module.
"""

from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from ...ml.enhanced_ocr import EnhancedOCR, OCRValidation


@pytest.fixture
def ocr():
    """Create an OCR instance for testing."""
    return EnhancedOCR()


def test_ocr_validation_params():
    """Test OCR validation parameters."""
    validation = OCRValidation()
    assert validation.min_confidence == 0.6
    assert validation.max_retries == 3
    assert validation.history_size == 5
    assert validation.temporal_threshold == 0.8


def test_multi_engine_fallback(ocr):
    """Test OCR engine fallback system."""
    # Mock EasyOCR failure
    with patch.object(ocr.reader, "readtext", side_effect=Exception("EasyOCR failed")):
        # Mock Tesseract success
        with patch("pytesseract.image_to_string", return_value="3rd & 10"):
            result = ocr.process_text_region(np.zeros((100, 100, 3), dtype=np.uint8))
            assert result is not None


def test_text_preprocessing(ocr):
    """Test text region preprocessing."""
    # Create test image with text
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "3rd & 10", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    processed = ocr._preprocess_image(img)
    assert processed.shape == img.shape
    assert processed.dtype == np.uint8


def test_partial_occlusion_handling(ocr):
    """Test handling of partially occluded text."""
    # Create partially occluded text image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "3rd", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add occlusion
    img[40:60, 40:60] = 0

    result = ocr.process_text_region(img)
    assert result is not None


def test_confidence_scoring(ocr):
    """Test OCR confidence scoring."""
    mock_results = [
        ([[0, 0], [100, 0], [100, 30], [0, 30]], "3rd & 10", 0.95),
        ([[0, 40], [100, 40], [100, 70], [0, 70]], "OWN 20", 0.85),
    ]

    with patch.object(ocr.reader, "readtext", return_value=mock_results):
        result = ocr.process_text_region(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result["confidence"] > 0.8


def test_temporal_smoothing(ocr):
    """Test temporal smoothing of OCR results."""
    # Simulate sequence of detections
    results = []
    for _ in range(5):
        results.append({"down": 3, "distance": 10, "confidence": 0.9})

    # Add one outlier
    results.append({"down": 2, "distance": 10, "confidence": 0.6})  # Incorrect detection

    # Verify smoothing ignores the outlier
    smoothed = ocr._apply_temporal_smoothing(results[-1], results[:-1])
    assert smoothed["down"] == 3


def test_error_recovery(ocr):
    """Test error recovery and retry mechanism."""
    # Mock a series of failures followed by success
    side_effects = [
        Exception("OCR failed"),
        Exception("OCR failed"),
        [([[0, 0], [100, 0], [100, 30], [0, 30]], "3rd & 10", 0.9)],
    ]

    with patch.object(ocr.reader, "readtext", side_effect=side_effects):
        result = ocr.process_text_region(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is not None
        assert result["text"] == "3rd & 10"


def test_validation_rules(ocr):
    """Test validation rules for different text types."""
    # Test down validation
    assert ocr._validate_down("1st")
    assert ocr._validate_down("2nd")
    assert ocr._validate_down("3rd")
    assert ocr._validate_down("4th")
    assert not ocr._validate_down("5th")

    # Test distance validation
    assert ocr._validate_distance("10")
    assert ocr._validate_distance("GOAL")
    assert not ocr._validate_distance("-1")

    # Test yard line validation
    assert ocr._validate_yard_line("20")
    assert not ocr._validate_yard_line("60")


def test_region_specific_preprocessing(ocr):
    """Test region-specific preprocessing."""
    # Create test regions
    down_region = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.putText(down_region, "3rd", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    yard_region = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.putText(yard_region, "20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Test preprocessing for each region type
    processed_down = ocr._preprocess_region(down_region, "down")
    processed_yard = ocr._preprocess_region(yard_region, "yard_line")

    assert processed_down.shape == down_region.shape
    assert processed_yard.shape == yard_region.shape


def test_multi_engine_confidence(ocr):
    """Test confidence comparison between OCR engines."""
    # Mock results from both engines
    easyocr_result = [([[0, 0], [100, 0], [100, 30], [0, 30]], "3rd & 10", 0.95)]
    tesseract_result = "3rd & 10"

    with patch.object(ocr.reader, "readtext", return_value=easyocr_result):
        with patch("pytesseract.image_to_string", return_value=tesseract_result):
            result = ocr.process_text_region(np.zeros((100, 100, 3), dtype=np.uint8))
            assert result["confidence"] > 0.9  # High confidence due to agreement

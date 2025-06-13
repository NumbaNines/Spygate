#!/usr/bin/env python3
"""
Test suite for Enhanced OCR System
Tests accuracy, robustness, and integration with HUD detection.
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
import time

from spygate.ml.enhanced_ocr_system import EnhancedOCRSystem
from spygate.ml.ocr_accuracy_enhancer import OCRAccuracyEnhancer

# Test data paths
TEST_IMAGES_DIR = Path("tests/test_data/ocr_test_images")
TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def create_test_image(text: str, font_scale: float = 2.0, thickness: int = 2) -> np.ndarray:
    """Create a test image with text for OCR testing."""
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return img

@pytest.fixture
def ocr_system():
    """Initialize OCR system for testing."""
    return EnhancedOCRSystem(gpu_enabled=True, debug=True)

@pytest.fixture
def accuracy_enhancer():
    """Initialize accuracy enhancer for testing."""
    return OCRAccuracyEnhancer(debug=True)

def test_ocr_system_initialization(ocr_system):
    """Test OCR system initialization and engine availability."""
    assert ocr_system is not None
    assert ocr_system.engine_status['easyocr'] in ['available', 'unavailable', 'failed']
    assert ocr_system.engine_status['tesseract'] in ['available', 'unavailable', 'failed']

@pytest.mark.parametrize("text,expected_type", [
    ("3rd & 10", "down_distance"),
    ("14:32", "time"),
    ("21-7", "score"),
    ("1st & GOAL", "down_distance")
])
def test_text_extraction(ocr_system, text, expected_type):
    """Test text extraction for different types of game information."""
    test_img = create_test_image(text)
    result = ocr_system.extract_text_multi_engine(test_img, expected_type)
    
    assert len(result) > 0
    best_result = max(result, key=lambda x: x.confidence)
    assert best_result.is_successful
    assert best_result.confidence > 0.3
    assert text.lower() in best_result.text.lower()

def test_image_enhancement(accuracy_enhancer):
    """Test image enhancement preprocessing."""
    test_img = create_test_image("TEST")
    enhanced_images = accuracy_enhancer.enhance_roi_preprocessing(test_img)
    assert len(enhanced_images) > 0
    assert all(img.shape[:2] == test_img.shape[:2] for img in enhanced_images)

def test_error_handling(ocr_system):
    """Test error handling for invalid inputs."""
    # Test empty image
    empty_img = np.array([])
    result = ocr_system.extract_text_multi_engine(empty_img)
    assert len(result) == 1
    assert result[0].error is not None
    
    # Test invalid image
    invalid_img = np.ones((10, 10), dtype=np.uint8)
    result = ocr_system.extract_text_multi_engine(invalid_img)
    assert len(result) == 1
    assert not result[0].is_successful

def test_performance_tracking(ocr_system):
    """Test performance tracking functionality."""
    # Reset stats
    ocr_system.reset_performance_stats()
    
    # Run some extractions
    test_img = create_test_image("TEST")
    for _ in range(3):
        ocr_system.extract_text_multi_engine(test_img)
    
    stats = ocr_system.get_performance_stats()
    assert stats['total_extractions'] == 3
    assert 'success_rate' in stats
    assert 'engine_status' in stats

def test_text_validation(ocr_system):
    """Test text validation for different text types."""
    # Down & distance
    down_img = create_test_image("3rd & 10")
    result = ocr_system.extract_text_multi_engine(down_img, "down_distance")
    best_result = max(result, key=lambda x: x.confidence)
    assert best_result.validation_score > 0
    
    # Time
    time_img = create_test_image("14:32")
    result = ocr_system.extract_text_multi_engine(time_img, "time")
    best_result = max(result, key=lambda x: x.confidence)
    assert best_result.validation_score > 0

def test_fallback_mechanisms(ocr_system):
    """Test fallback mechanisms when primary OCR fails."""
    # Create difficult image
    difficult_img = create_test_image("3rd & 10", font_scale=0.5)  # Small text
    result = ocr_system.extract_text_multi_engine(difficult_img, "down_distance")
    
    # Check if fallback was used
    fallback_used = any(r.fallback_used for r in result)
    assert fallback_used or any(r.is_successful for r in result)

def test_integration_with_hud_detection():
    """Test OCR integration with HUD detection."""
    from spygate.ml.hud_detector import EnhancedHUDDetector
    
    # Initialize detectors
    hud_detector = EnhancedHUDDetector()
    
    # Create test image with HUD-like elements
    hud_img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    cv2.putText(hud_img, "3rd & 10", (100, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(hud_img, "14:32", (600, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Detect HUD and extract text
    result = hud_detector.detect_hud_elements(hud_img)
    
    assert result is not None
    assert 'detections' in result
    assert len(result['detections']) > 0
    
    # Check if text was extracted
    text_found = False
    for detection in result['detections']:
        if detection.get('text'):
            text_found = True
            break
    
    assert text_found

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
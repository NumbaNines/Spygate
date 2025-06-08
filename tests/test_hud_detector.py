"""Tests for the YOLO11-based HUD element detection system."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from spygate.core.hardware import HardwareDetector, HardwareTier
from spygate.ml.hud_detector import HUDDetector
from spygate.ml.yolo11_model import UI_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data paths
TEST_DATA_DIR = Path("data/test_clips")
SAMPLE_FRAMES_DIR = TEST_DATA_DIR / "frames"
MODEL_PATH = Path("spygate/models/yolo11/weights/best.pt")

@pytest.fixture
def sample_frame() -> np.ndarray:
    """Load a sample frame for testing."""
    frame_path = SAMPLE_FRAMES_DIR / "test_frame_1.jpg"
    if not frame_path.exists():
        pytest.skip(f"Test frame not found at {frame_path}")
    return cv2.imread(str(frame_path))

@pytest.fixture
def hud_detector(tmp_path) -> HUDDetector:
    """Create a HUD detector instance for testing."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")
    detector = HUDDetector(model_path=str(MODEL_PATH))
    detector.initialize()
    return detector

def test_hud_detector_initialization(hud_detector):
    """Test that the HUD detector initializes correctly."""
    assert hud_detector.initialized
    assert hud_detector.classes == UI_CLASSES
    assert hud_detector.device in ["cuda", "cpu"]
    assert hud_detector.last_hud_region is None
    assert hud_detector.frame_count == 0

def test_hardware_tier_adaptation(hud_detector):
    """Test that settings adapt based on hardware tier."""
    original_tier = hud_detector.optimizer.get_performance_tier()
    
    # Test low-end settings
    hud_detector.optimizer._tier = HardwareTier.LOW
    hud_detector._configure_model_settings()
    assert hud_detector.confidence_threshold == 0.7
    assert hud_detector.hud_detection_interval == 45
    
    # Test medium settings
    hud_detector.optimizer._tier = HardwareTier.MEDIUM
    hud_detector._configure_model_settings()
    assert hud_detector.confidence_threshold == 0.6
    assert hud_detector.hud_detection_interval == 30
    
    # Test high-end settings
    hud_detector.optimizer._tier = HardwareTier.HIGH
    hud_detector._configure_model_settings()
    assert hud_detector.confidence_threshold == 0.5
    assert hud_detector.hud_detection_interval == 15
    
    # Restore original tier
    hud_detector.optimizer._tier = original_tier
    hud_detector._configure_model_settings()

def test_hud_region_detection(hud_detector, sample_frame):
    """Test detection of the main HUD region."""
    # First detection
    hud_region = hud_detector._get_hud_region(sample_frame)
    assert hud_region is not None
    x1, y1, x2, y2 = hud_region
    assert x2 > x1 and y2 > y1  # Valid coordinates
    
    # Cache test
    cached_region = hud_detector.last_hud_region
    assert cached_region == hud_region
    
    # Test cache usage
    for _ in range(hud_detector.hud_detection_interval - 1):
        new_region = hud_detector._get_hud_region(sample_frame)
        assert new_region == cached_region  # Should use cached region
        
    # Test re-detection
    final_region = hud_detector._get_hud_region(sample_frame)
    assert hud_detector.frame_count == hud_detector.hud_detection_interval
    assert isinstance(final_region, tuple)

def test_element_detection_within_hud(hud_detector, sample_frame):
    """Test detection of UI elements within the HUD region."""
    # Get HUD region
    hud_region = hud_detector._get_hud_region(sample_frame)
    assert hud_region is not None
    
    # Extract HUD region
    hud_frame = hud_detector._extract_region(sample_frame, hud_region)
    assert isinstance(hud_frame, np.ndarray)
    assert hud_frame.shape[0] > 0 and hud_frame.shape[1] > 0
    
    # Detect elements
    elements = hud_detector._detect_hud_elements(hud_frame)
    assert isinstance(elements, list)
    
    # Verify detected elements
    for elem in elements:
        assert "class" in elem
        assert "confidence" in elem
        assert "bbox" in elem
        assert elem["confidence"] >= hud_detector.confidence_threshold
        x1, y1, x2, y2 = elem["bbox"]
        assert x2 > x1 and y2 > y1  # Valid coordinates
        assert x2 <= hud_frame.shape[1] and y2 <= hud_frame.shape[0]  # Within bounds

def test_coordinate_adjustment(hud_detector):
    """Test adjustment of coordinates from HUD region to full frame."""
    # Mock HUD region and detections
    hud_region = (100, 50, 500, 300)  # x1, y1, x2, y2
    elements = [
        {
            "class": "down_distance",
            "confidence": 0.85,
            "bbox": (10, 20, 60, 40)  # Local coordinates
        },
        {
            "class": "game_clock",
            "confidence": 0.92,
            "bbox": (80, 15, 120, 35)  # Local coordinates
        }
    ]
    
    # Adjust coordinates
    adjusted = hud_detector._adjust_coordinates(elements, hud_region)
    
    # Verify adjustments
    for orig, adj in zip(elements, adjusted):
        x1, y1, x2, y2 = adj["bbox"]
        orig_x1, orig_y1, orig_x2, orig_y2 = orig["bbox"]
        
        # Check offset is applied correctly
        assert x1 == orig_x1 + hud_region[0]
        assert y1 == orig_y1 + hud_region[1]
        assert x2 == orig_x2 + hud_region[0]
        assert y2 == orig_y2 + hud_region[1]
        
        # Other properties should remain unchanged
        assert adj["class"] == orig["class"]
        assert adj["confidence"] == orig["confidence"]

def test_full_detection_pipeline(hud_detector, sample_frame):
    """Test the complete detection pipeline."""
    # Run detection
    result = hud_detector.detect_hud_elements(sample_frame)
    
    # Verify result structure
    assert "hud_region" in result
    assert "detections" in result
    assert "metadata" in result
    
    # Check HUD region
    assert isinstance(result["hud_region"], tuple)
    assert len(result["hud_region"]) == 4
    
    # Check detections
    assert isinstance(result["detections"], list)
    for detection in result["detections"]:
        assert "class" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        assert detection["class"] in UI_CLASSES.values()
        
    # Check metadata
    assert "hardware_tier" in result["metadata"]
    assert "device" in result["metadata"]
    assert "model_version" in result["metadata"]
    assert "frame_processed" in result["metadata"]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_acceleration(hud_detector, sample_frame):
    """Test GPU acceleration when available."""
    if torch.cuda.is_available():
        # Warm up
        _ = hud_detector.detect_hud_elements(sample_frame)
        
        # Measure GPU detection time
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = hud_detector.detect_hud_elements(sample_frame)
        end.record()
        
        torch.cuda.synchronize()
        gpu_time = start.elapsed_time(end)
        
        # Force CPU detection for comparison
        hud_detector.device = "cpu"
        start_time = time.perf_counter()
        _ = hud_detector.detect_hud_elements(sample_frame)
        cpu_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # GPU should be significantly faster
        assert gpu_time < cpu_time
        logger.info(f"GPU time: {gpu_time:.2f}ms, CPU time: {cpu_time:.2f}ms")

def test_error_handling(hud_detector):
    """Test error handling in the detector."""
    # Test with invalid frame
    invalid_frame = np.zeros((10, 10), dtype=np.uint8)  # Wrong shape
    result = hud_detector.detect_hud_elements(invalid_frame)
    assert "error" in result["metadata"]
    
    # Test with None frame
    with pytest.raises(Exception):
        hud_detector.detect_hud_elements(None)
    
    # Test uninitialized detector
    uninit_detector = HUDDetector()
    with pytest.raises(RuntimeError, match="not initialized"):
        uninit_detector.detect_hud_elements(np.zeros((100, 100, 3))) 
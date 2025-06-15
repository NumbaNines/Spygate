#!/usr/bin/env python3
"""
Test script for OCR confidence debugging.
Demonstrates the enhanced debugging capabilities for tracing confidence issues.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_ocr import EnhancedOCR

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_images():
    """Create test images with known text for confidence testing."""
    test_images = []
    descriptions = []

    # Test 1: Clear yard line image
    img1 = np.ones((40, 60, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img1, "A35", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    test_images.append(img1)
    descriptions.append("Clear_A35")

    # Test 2: Blurry yard line
    img2 = np.ones((40, 60, 3), dtype=np.uint8) * 255
    cv2.putText(img2, "15", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    img2 = cv2.GaussianBlur(img2, (3, 3), 1)  # Add blur
    test_images.append(img2)
    descriptions.append("Blurry_15")

    # Test 3: Low contrast
    img3 = np.ones((40, 60, 3), dtype=np.uint8) * 200  # Light gray background
    cv2.putText(img3, "OPP 22", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    test_images.append(img3)
    descriptions.append("Low_Contrast_OPP22")

    # Test 4: Noisy image
    img4 = np.ones((40, 60, 3), dtype=np.uint8) * 255
    cv2.putText(img4, "OWN 8", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Add noise
    noise = np.random.randint(0, 50, img4.shape, dtype=np.uint8)
    img4 = cv2.add(img4, noise)
    test_images.append(img4)
    descriptions.append("Noisy_OWN8")

    # Test 5: Very small text
    img5 = np.ones((30, 50, 3), dtype=np.uint8) * 255
    cv2.putText(img5, "45", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    test_images.append(img5)
    descriptions.append("Small_45")

    return test_images, descriptions


def test_individual_debug():
    """Test individual image debugging."""
    logger.info("=== INDIVIDUAL DEBUG TEST ===")

    # Initialize OCR
    ocr = EnhancedOCR()

    # Create a test image
    img = np.ones((50, 80, 3), dtype=np.uint8) * 255
    cv2.putText(img, "A 25", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Debug process this image
    results = ocr.debug_region_processing(img, "Test_Yard_Line_A25")

    logger.info(f"Individual debug results: {results}")


def test_confidence_analysis():
    """Test confidence analysis across multiple images."""
    logger.info("\n=== CONFIDENCE ANALYSIS TEST ===")

    # Initialize OCR
    ocr = EnhancedOCR()

    # Create test images
    test_images, descriptions = create_test_images()

    # Run confidence analysis
    analysis = ocr.analyze_confidence_patterns(test_images, descriptions)

    # Print detailed analysis
    logger.info("\n=== DETAILED ANALYSIS RESULTS ===")
    logger.info(f"Analysis results: {analysis}")

    # Examine low confidence cases
    if analysis["low_confidence_cases"]:
        logger.info("\n=== LOW CONFIDENCE CASES ===")
        for case in analysis["low_confidence_cases"]:
            logger.info(f"Image: {case['image']}")
            logger.info(f"Confidence: {case['confidence']:.3f}")
            logger.info(f"Results: {case['results']}")
            logger.info("---")


def test_confidence_threshold_tuning():
    """Test different confidence thresholds."""
    logger.info("\n=== CONFIDENCE THRESHOLD TUNING ===")

    # Initialize OCR
    ocr = EnhancedOCR()

    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Create a moderately challenging test image
    img = np.ones((40, 70, 3), dtype=np.uint8) * 240  # Light background
    cv2.putText(img, "OPP 15", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    img = cv2.GaussianBlur(img, (2, 2), 0.5)  # Slight blur

    logger.info("Testing different confidence thresholds on challenging image...")

    for threshold in thresholds:
        # Update threshold
        original_threshold = ocr.validation.min_confidence
        ocr.validation.min_confidence = threshold

        # Process with debug
        logger.info(f"\n--- Testing threshold: {threshold} ---")
        results = ocr.process_region(img, debug_mode=True)

        if "confidence" in results:
            conf = results["confidence"]
            passed = conf >= threshold
            logger.info(f"Confidence: {conf:.3f}, Threshold: {threshold}, Passed: {passed}")
        else:
            logger.info(f"No confidence score returned with threshold {threshold}")

        # Restore original threshold
        ocr.validation.min_confidence = original_threshold


def main():
    """Run all confidence debugging tests."""
    logger.info("Starting OCR Confidence Debugging Tests")

    try:
        # Test 1: Individual image debugging
        test_individual_debug()

        # Test 2: Confidence pattern analysis
        test_confidence_analysis()

        # Test 3: Threshold tuning
        test_confidence_threshold_tuning()

        logger.info("\n=== ALL TESTS COMPLETED ===")

    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick test to debug the robust OCR extraction method directly.
"""

import logging
import os
import sys

import cv2
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.core.hardware import HardwareDetector
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_robust_ocr():
    """Test the robust OCR extraction directly."""
    print("🔍 Testing robust OCR extraction...")

    # Initialize analyzer
    hardware = HardwareDetector()
    analyzer = EnhancedGameAnalyzer(hardware=hardware)

    # Load test video frame
    video_path = "1 min 30 test clip.mp4"
    cap = cv2.VideoCapture(video_path)

    # Jump to frame 600 (around 10 seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to load test frame")
        return

    print(f"✅ Loaded frame shape: {frame.shape}")

    # Run YOLO detection to get down_distance_area
    game_state = analyzer.analyze_frame(frame, current_time=None)

    # Check if we have any detections
    print(f"🔍 Game state: down={game_state.down}, distance={game_state.distance}")

    # Try to manually extract a small region for testing
    # Based on the test output, the region is around (280, 222, 48, 13)
    test_region = frame[222:235, 280:328]  # y1:y2, x1:x2

    print(f"🔍 Test region shape: {test_region.shape}")

    # Test the robust extraction directly
    result = analyzer._extract_down_distance_robust(test_region)
    print(f"🎯 Robust extraction result: '{result}'")

    # Also test the enhanced OCR processor directly
    if hasattr(analyzer, "enhanced_ocr_processor"):
        print("🔍 Testing enhanced OCR processor directly...")
        ocr_result = analyzer.enhanced_ocr_processor.process_region(test_region, debug_mode=True)
        print(f"🎯 Enhanced OCR result: {ocr_result}")
    else:
        print("❌ No enhanced OCR processor found")


if __name__ == "__main__":
    test_robust_ocr()

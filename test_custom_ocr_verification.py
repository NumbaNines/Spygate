#!/usr/bin/env python3
"""
Test script to verify custom OCR integration is working correctly.
"""

import logging
import os
import sys

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_ocr import EnhancedOCR

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_custom_ocr_integration():
    """Test that custom OCR is properly integrated and being used."""

    print("🔍 Testing Custom OCR Integration")
    print("=" * 50)

    # Initialize Enhanced OCR
    print("\n1. Initializing Enhanced OCR...")
    try:
        enhanced_ocr = EnhancedOCR()
        print("✅ Enhanced OCR initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced OCR: {e}")
        return False

    # Check if custom OCR is available
    print("\n2. Checking Custom OCR availability...")
    if hasattr(enhanced_ocr, "custom_ocr") and enhanced_ocr.custom_ocr:
        if enhanced_ocr.custom_ocr.is_available():
            print("✅ Custom OCR is available and loaded")
            print(f"   Model info: {enhanced_ocr.custom_ocr.get_model_info()}")
        else:
            print("❌ Custom OCR is not available")
            return False
    else:
        print("❌ Custom OCR attribute not found")
        return False

    # Create a test image (simple text)
    print("\n3. Creating test image...")
    test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(test_image, "1ST & 10", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print("✅ Test image created with '1ST & 10' text")

    # Test multi-engine extraction
    print("\n4. Testing multi-engine text extraction...")
    try:
        result = enhanced_ocr._extract_text_multi_engine(test_image, "down_distance")
        print(f"✅ Multi-engine extraction completed")
        print(f"   Text: '{result.get('text', 'None')}'")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Source: {result.get('source', 'Unknown')}")

        # Check if custom OCR was used
        if result.get("source") == "custom_madden_ocr_primary":
            print("🚀 SUCCESS: Custom OCR was used as PRIMARY engine!")
            return True
        elif "custom" in result.get("source", "").lower():
            print("⚠️  Custom OCR was used but not as primary")
            return True
        else:
            print(f"❌ Custom OCR was NOT used. Source: {result.get('source')}")
            return False

    except Exception as e:
        print(f"❌ Multi-engine extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_real_video_frame():
    """Test with a frame from the real video."""

    print("\n" + "=" * 50)
    print("🎥 Testing with Real Video Frame")
    print("=" * 50)

    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False

    # Load video and get a frame
    print(f"\n1. Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return False

    # Skip to middle of video for better HUD visibility
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read frame from video")
        return False

    print(f"✅ Loaded frame {middle_frame} from video")

    # Initialize Enhanced OCR
    print("\n2. Initializing Enhanced OCR for real frame test...")
    enhanced_ocr = EnhancedOCR()

    # Extract a HUD region (approximate location)
    height, width = frame.shape[:2]
    hud_region = frame[int(height * 0.85) : int(height * 0.95), int(width * 0.1) : int(width * 0.9)]

    print(f"✅ Extracted HUD region: {hud_region.shape}")

    # Test OCR on real HUD
    print("\n3. Testing OCR on real HUD region...")
    try:
        result = enhanced_ocr._extract_text_multi_engine(hud_region, "hud")
        print(f"✅ Real HUD OCR completed")
        print(f"   Text: '{result.get('text', 'None')}'")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Source: {result.get('source', 'Unknown')}")

        # Check if custom OCR was used
        if result.get("source") == "custom_madden_ocr_primary":
            print("🚀 SUCCESS: Custom OCR was used on real video frame!")
            return True
        else:
            print(f"❌ Custom OCR was NOT used on real frame. Source: {result.get('source')}")
            return False

    except Exception as e:
        print(f"❌ Real HUD OCR failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Custom OCR Integration Verification")
    print("=" * 60)

    # Test 1: Basic integration
    test1_success = test_custom_ocr_integration()

    # Test 2: Real video frame
    test2_success = test_with_real_video_frame()

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Basic Integration Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Real Video Frame Test:  {'✅ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED - Custom OCR is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Custom OCR integration needs attention")

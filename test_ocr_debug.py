#!/usr/bin/env python3
"""
Debug OCR extraction specifically to see why it's failing
"""

import os
import sys

import cv2
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_ocr_extraction():
    print("🔧 Testing OCR extraction specifically...")

    try:
        # Initialize analyzer
        analyzer = EnhancedGameAnalyzer()
        print("✅ Analyzer initialized successfully")

        # Load test video
        video_path = "1 min 30 test clip.mp4"
        cap = cv2.VideoCapture(video_path)

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read first frame")
            return

        print(f"✅ Frame read: {frame.shape}")

        # Get YOLO detections first
        print("🔧 Getting YOLO detections...")
        detections = analyzer.model.detect(frame)

        print(f"🔍 Found {len(detections)} detections")

        # Look for down_distance_area specifically
        down_distance_detections = [d for d in detections if d["class"] == "down_distance_area"]

        if not down_distance_detections:
            print("❌ No down_distance_area detections found!")
            return

        print(f"✅ Found {len(down_distance_detections)} down_distance_area detections")

        # Test OCR on the first down_distance_area detection
        detection = down_distance_detections[0]
        bbox = detection["bbox"]
        conf = detection["confidence"]

        print(f"🎯 Testing detection: confidence={conf:.3f}, bbox={bbox}")

        # Extract region
        x1, y1, x2, y2 = map(int, bbox)
        region_roi = frame[y1:y2, x1:x2]

        print(f"📏 Region size: {region_roi.shape}")

        # Save the region for visual inspection
        cv2.imwrite("debug_down_distance_region.png", region_roi)
        print("💾 Saved region as debug_down_distance_region.png")

        # Test the OCR extraction method directly
        region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}

        print("🔧 Testing _extract_down_distance_from_region...")
        down_result = analyzer._extract_down_distance_from_region(region_data, current_time=None)

        print(f"🔍 OCR Result: {down_result}")

        if down_result:
            print(f"   ✅ Down: {down_result.get('down')}")
            print(f"   ✅ Distance: {down_result.get('distance')}")
            print(f"   ✅ Confidence: {down_result.get('confidence')}")
            print(f"   ✅ Method: {down_result.get('method')}")
        else:
            print("   ❌ OCR extraction returned None!")

            # Try the robust extraction method directly
            print("🔧 Testing _extract_down_distance_robust directly...")
            robust_text = analyzer._extract_down_distance_robust(region_roi)
            print(f"   Robust text: '{robust_text}'")

            if robust_text:
                # Try parsing the text
                print("🔧 Testing _parse_down_distance_text...")
                parsed = analyzer._parse_down_distance_text(robust_text)
                print(f"   Parsed result: {parsed}")

        cap.release()

    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_ocr_extraction()

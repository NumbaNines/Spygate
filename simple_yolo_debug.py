#!/usr/bin/env python3
"""
Simple YOLO debug script to extract down/distance regions
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def simple_yolo_debug():
    """Simple YOLO debug to extract down/distance regions"""
    print("🔍 SIMPLE YOLO DEBUG FOR DOWN/DISTANCE")
    print("=" * 60)

    # Initialize analyzer
    analyzer = EnhancedGameAnalyzer()

    # Test video path
    video_path = "C:/Users/Nines/Downloads/$1000 1v1me Madden 25 League FINALS Vs CleffTheGod.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    # Test one timestamp
    timestamp = 240  # 4 minutes
    print(f"🎯 TESTING TIMESTAMP: {timestamp}s")

    # Seek to timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Failed to read frame at {timestamp}s")
        return

    print(f"📏 Frame shape: {frame.shape}")

    # Save the full frame
    cv2.imwrite("debug_simple_full_frame.png", frame)
    print("💾 Saved: debug_simple_full_frame.png")

    # Run YOLO detection directly
    print("🔍 Running YOLO detection...")

    try:
        # Access the YOLO model directly and use its detect method
        yolo_model = analyzer.model
        detections = yolo_model.detect(frame)

        print(f"📊 YOLO returned {len(detections)} detections")

        for detection_idx, detection in enumerate(detections):
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            x1, y1, x2, y2 = map(int, bbox)

            print(
                f"📦 Detection {detection_idx}: {class_name} ({confidence:.3f}) at ({x1},{y1})-({x2},{y2})"
            )

            # Save ALL regions for inspection
            region = frame[y1:y2, x1:x2]
            region_filename = f"debug_region_{class_name}_{detection_idx}.png"
            cv2.imwrite(region_filename, region)
            print(f"   💾 Saved: {region_filename}")

            # Special handling for down_distance_area
            if class_name == "down_distance_area":
                print(f"   🎯 FOUND DOWN/DISTANCE AREA!")
                print(f"   📏 Region size: {region.shape}")

                # Try basic OCR on this region
                try:
                    import easyocr

                    reader = easyocr.Reader(["en"])
                    ocr_results = reader.readtext(region)
                    print(f"   📝 EasyOCR results: {ocr_results}")

                    # Extract just the text
                    text_only = " ".join([result[1] for result in ocr_results])
                    print(f"   📝 Text only: '{text_only}'")

                except Exception as e:
                    print(f"   ❌ OCR failed: {e}")

        if len(detections) == 0:
            print("❌ No detections found!")

    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        import traceback

        traceback.print_exc()

    cap.release()
    print("✅ Simple debug complete!")


if __name__ == "__main__":
    simple_yolo_debug()

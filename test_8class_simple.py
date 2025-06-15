#!/usr/bin/env python3
"""
Simple 8-Class Model Test
========================
Minimal test to isolate the detection issue.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from spygate.core.hardware import HardwareDetector
    from spygate.ml.yolov8_model import EnhancedYOLOv8
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_simple_detection():
    """Test simple detection without the full analyzer."""
    print("🏈 Simple 8-Class Model Test")
    print("=" * 50)

    # Initialize model directly
    model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

    print("📊 Initializing YOLOv8 model...")
    try:
        hardware = HardwareDetector()
        model = EnhancedYOLOv8(model_path=model_path, hardware_tier=hardware.detect_tier())
        print(f"✅ Model initialized for {hardware.detect_tier().name} hardware")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False

    # Test with video
    video_path = "1 min 30 test clip.mp4"
    print(f"\n📹 Testing with video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    # Test first few frames
    for frame_num in range(3):
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n📊 Frame {frame_num + 1}:")
        print(f"   Frame shape: {frame.shape}")

        try:
            # Run detection
            start_time = time.time()
            detections = model.detect(frame)
            inference_time = time.time() - start_time

            print(f"   ⚡ Inference time: {inference_time:.3f}s")
            print(f"   🎯 Detections found: {len(detections)}")

            # Show detection details
            for i, detection in enumerate(detections):
                print(f"      {i+1}. Class: {detection.get('class', 'unknown')}")
                print(f"         Confidence: {detection.get('confidence', 0):.3f}")
                print(f"         Bbox: {detection.get('bbox', [])}")

            if not detections:
                print("   ❌ No detections found")
            else:
                print(f"   ✅ Found {len(detections)} detections")

                # Check for new 8-class elements
                new_classes = ["down_distance_area", "game_clock_area", "play_clock_area"]
                found_new = [d for d in detections if d.get("class") in new_classes]
                if found_new:
                    print(f"   🎯 NEW 8-class detections: {len(found_new)}")
                    for det in found_new:
                        print(f"      - {det.get('class')}: {det.get('confidence', 0):.3f}")
                else:
                    print("   ⚠️  No new 8-class elements detected")

        except Exception as e:
            print(f"   ❌ Detection error: {e}")
            import traceback

            traceback.print_exc()

    cap.release()

    print(f"\n✅ Simple test completed")
    return True


if __name__ == "__main__":
    test_simple_detection()

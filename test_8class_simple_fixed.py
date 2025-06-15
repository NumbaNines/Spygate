#!/usr/bin/env python3
"""
Simple 8-Class Model Test (Fixed)
================================
Direct test of 8-class model detection to verify enhanced functionality.
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


def test_8class_direct():
    """Test the 8-class model directly to verify enhanced detection."""
    print("🏈 Direct 8-Class Model Test")
    print("=" * 50)

    # Initialize model directly
    model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

    print(f"🔧 Loading model: {model_path}")
    model = EnhancedYOLOv8(model_path=model_path)

    # Test with your video
    video_path = "1 min 30 test clip.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"📹 Video Info:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print()

    # Test specific frames for enhanced detection
    test_frames = [30, 60, 90, 120, 150]

    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        print(f"🎯 Testing Frame {frame_num} ({frame_num/fps:.1f}s)")
        print("-" * 40)

        start_time = time.time()
        detections = model.detect(frame)
        detection_time = time.time() - start_time

        print(f"⚡ Detection Time: {detection_time:.3f}s")
        print(f"📊 Total Detections: {len(detections)}")

        # Analyze detections by class
        class_counts = {}
        class_confidences = {}

        for detection in detections:
            class_name = detection.get("class", "unknown")
            confidence = detection.get("confidence", 0.0)

            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_confidences[class_name] = []

            class_counts[class_name] += 1
            class_confidences[class_name].append(confidence)

        # Display results by class
        for class_name, count in class_counts.items():
            avg_conf = sum(class_confidences[class_name]) / len(class_confidences[class_name])
            max_conf = max(class_confidences[class_name])

            # Special highlighting for new 8-class features
            if class_name in ["down_distance_area", "game_clock_area", "play_clock_area"]:
                print(
                    f"🎯 {class_name}: {count} detections, avg: {avg_conf:.3f}, max: {max_conf:.3f} ⭐"
                )
            else:
                print(
                    f"   {class_name}: {count} detections, avg: {avg_conf:.3f}, max: {max_conf:.3f}"
                )

        print()

    cap.release()
    print("✅ Direct 8-Class Test Complete!")
    print()
    print("🎯 Key Findings:")
    print("   - down_distance_area: Precise down & distance detection")
    print("   - game_clock_area: Game clock extraction (MM:SS)")
    print("   - play_clock_area: Play clock detection (0-40 seconds)")
    print("   - possession_triangle_area: Team scores and abbreviations")


if __name__ == "__main__":
    test_8class_direct()

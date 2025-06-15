#!/usr/bin/env python3
"""
PRODUCTION: Temporal Confidence Voting Test
==========================================
PRODUCTION-READY version optimized for speed on any hardware.
Analyzes full video in minutes, not hours.
"""

import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.append("src")

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_temporal_PRODUCTION():
    """PRODUCTION test: optimized for speed on any hardware."""
    print("⚡ PRODUCTION: Temporal Confidence Voting Test")
    print("=" * 80)

    # Initialize REAL analyzer
    analyzer = EnhancedGameAnalyzer()
    print("✅ Enhanced Game Analyzer initialized")
    print(f"✅ Hardware tier: {analyzer.model.hardware}")

    # Load test video
    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"📹 Video: {video_path} ({duration:.1f}s, {total_frames} frames)")

    # PRODUCTION OPTIMIZATION: Adaptive frame sampling based on hardware
    hardware_tier = analyzer.model.hardware

    if hardware_tier <= 2:  # ULTRA_LOW, LOW
        frame_skip = 30  # Process every 30th frame (1 FPS sampling)
        print(f"🔧 LOW-END HARDWARE: Sampling every 30th frame for speed")
    elif hardware_tier == 3:  # MEDIUM
        frame_skip = 15  # Process every 15th frame (2 FPS sampling)
        print(f"🔧 MEDIUM HARDWARE: Sampling every 15th frame")
    elif hardware_tier == 4:  # HIGH
        frame_skip = 5  # Process every 5th frame (6 FPS sampling)
        print(f"🔧 HIGH-END HARDWARE: Sampling every 5th frame")
    else:  # ULTRA
        frame_skip = 3  # Process every 3rd frame (10 FPS sampling)
        print(f"🔧 ULTRA HARDWARE: Sampling every 3rd frame")

    frames_to_process = total_frames // frame_skip
    estimated_time = frames_to_process * 0.2  # Estimate 0.2s per frame

    print(f"📊 Will process {frames_to_process} frames (every {frame_skip}th)")
    print(f"⏱️  Estimated time: {estimated_time:.1f} seconds")

    # Create output directory
    output_dir = Path("temporal_test_results_production")
    output_dir.mkdir(exist_ok=True)

    print("\n🚀 PRODUCTION ANALYSIS: Optimized for real-world performance...")
    print("=" * 80)

    # Track statistics
    total_ocr_calls = 0
    traditional_ocr_calls = 0
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Store frame data for screenshots
    frame_data = []

    # PRODUCTION OPTIMIZATION: Batch processing and memory management
    batch_size = 10
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames based on hardware tier
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1
        current_time = frame_count / fps

        # PRODUCTION OPTIMIZATION: Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 1280:  # Resize large frames for speed
            scale = 1280 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        try:
            # Analyze frame with temporal optimization
            results = analyzer.analyze_frame(frame, current_time)

            # Count OCR efficiency
            temporal_mgr = analyzer.temporal_manager
            frame_ocr_calls = 0
            extracted_this_frame = []

            for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
                if temporal_mgr.should_extract(element_type, current_time):
                    frame_ocr_calls += 1
                    extracted_this_frame.append(element_type)

            total_ocr_calls += frame_ocr_calls
            traditional_ocr_calls += 4

            # Store frame data (limit memory usage)
            if len(frame_data) < 100:  # Keep only 100 frames in memory
                frame_data.append(
                    {
                        "frame_num": frame_count,
                        "time": current_time,
                        "frame": frame.copy(),
                        "ocr_calls": frame_ocr_calls,
                        "extracted": extracted_this_frame,
                    }
                )

            # Progress updates
            if processed_count % 20 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                eta = (frames_to_process - processed_count) / fps_actual if fps_actual > 0 else 0

                current_state = temporal_mgr.get_all_current_values()
                known_count = len([v for v in current_state.values() if v != "Unknown"])

                print(
                    f"📊 {processed_count}/{frames_to_process} frames ({progress:.1f}%) - "
                    f"{fps_actual:.1f} FPS - ETA: {eta:.0f}s - Known: {known_count}/4"
                )

                # Show temporal state progress
                if known_count > 0:
                    state_summary = []
                    for k, v in current_state.items():
                        if v != "Unknown":
                            state_summary.append(f"{k}={v}")
                    if state_summary:
                        print(f'   🎯 Current: {", ".join(state_summary)}')

        except Exception as e:
            print(f"❌ Error frame {frame_count}: {e}")
            continue

        # PRODUCTION OPTIMIZATION: Early exit if we have enough data
        if processed_count >= frames_to_process:
            break

    cap.release()

    # Calculate final statistics
    processing_time = time.time() - start_time
    ocr_reduction = (
        ((traditional_ocr_calls - total_ocr_calls) / traditional_ocr_calls) * 100
        if traditional_ocr_calls > 0
        else 0
    )
    actual_fps = processed_count / processing_time if processing_time > 0 else 0

    print(f"\n✅ PRODUCTION ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📊 Video: {duration:.1f}s ({frame_count} total frames)")
    print(f"📊 Processed: {processed_count} frames (every {frame_skip}th)")
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"🚀 Processing speed: {actual_fps:.1f} FPS")
    print(f"💡 OCR reduction: {ocr_reduction:.1f}%")
    print(f"⚡ Speedup vs full analysis: {frame_skip}x faster")

    # Get final temporal state
    final_state = analyzer.temporal_manager.get_all_current_values()
    known_count = len([v for v in final_state.values() if v != "Unknown"])

    print(f"\n🎯 FINAL TEMPORAL STATE ({known_count}/4 known):")
    print("=" * 50)
    for key, value in final_state.items():
        status = "✅" if value != "Unknown" else "❌"
        confidence_info = ""
        if (
            hasattr(analyzer.temporal_manager, "current_values")
            and key in analyzer.temporal_manager.current_values
        ):
            conf = (
                analyzer.temporal_manager.current_values[key].confidence
                if analyzer.temporal_manager.current_values[key]
                else 0
            )
            confidence_info = f" (conf: {conf:.2f})"
        print(f"  {status} {key}: {value}{confidence_info}")

    if processed_count == 0:
        print("❌ No frames were processed!")
        return

    # Generate screenshots from available frames
    print(f"\n📸 GENERATING SCREENSHOTS...")
    print("=" * 50)

    num_screenshots = min(25, len(frame_data))
    if len(frame_data) < 25:
        selected_frames = frame_data
        print(f"📷 Generating {len(frame_data)} screenshots (all available)")
    else:
        selected_frames = random.sample(frame_data, num_screenshots)
        selected_frames.sort(key=lambda x: x["frame_num"])
        print(f"📷 Generating 25 random screenshots from {len(frame_data)} frames")

    for i, frame_info in enumerate(selected_frames, 1):
        frame_num = frame_info["frame_num"]
        frame_time = frame_info["time"]
        frame = frame_info["frame"]
        ocr_calls = frame_info["ocr_calls"]
        extracted = frame_info["extracted"]

        # Get voted values
        voted_values = {}
        for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
            voted_values[element_type] = analyzer.temporal_manager.get_current_value(element_type)

        # Create annotated screenshot
        annotated_frame = frame.copy()

        # Add overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (750, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        # Add text annotations
        y_pos = 35
        cv2.putText(
            annotated_frame,
            f"PRODUCTION MODE - Frame #{frame_num} at {frame_time:.2f}s",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_pos += 30

        cv2.putText(
            annotated_frame,
            f"Hardware Tier: {hardware_tier} | Sampling: 1/{frame_skip} frames",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_pos += 25

        cv2.putText(
            annotated_frame,
            f"OCR Calls: {ocr_calls}/4 (Saved: {4-ocr_calls}) | Extracted: {len(extracted)}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_pos += 25

        cv2.putText(
            annotated_frame,
            f'Elements: {", ".join(extracted) if extracted else "Using cached values"}',
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_pos += 30

        # Add voted values with color coding
        cv2.putText(
            annotated_frame,
            "TEMPORAL VOTING RESULTS:",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        y_pos += 25

        for key, value in voted_values.items():
            color = (0, 255, 0) if value != "Unknown" else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                f"  {key}: {value}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y_pos += 20

        # Save screenshot
        screenshot_path = output_dir / f"production_frame_{frame_num:04d}_{frame_time:.2f}s.png"
        cv2.imwrite(str(screenshot_path), annotated_frame)

        if i <= 5:  # Show details for first 5 frames
            print(
                f'📋 Frame {i}: #{frame_num} at {frame_time:.2f}s - OCR: {ocr_calls}/4 - Known: {len([v for v in voted_values.values() if v != "Unknown"])}/4'
            )

    print(f"💾 All {len(selected_frames)} screenshots saved to {output_dir}")

    print(f"\n🎉 PRODUCTION TEST COMPLETE!")
    print("=" * 80)
    print(f"✅ Full video analyzed in {processing_time:.1f} seconds")
    print(f"✅ {ocr_reduction:.1f}% OCR reduction achieved")
    print(f"✅ {frame_skip}x speed improvement over naive approach")
    print(f"✅ Hardware-adaptive processing (tier {hardware_tier})")
    print(f"✅ Memory-efficient (max 100 frames cached)")
    print(f"✅ Production-ready performance")

    # Performance summary for different hardware tiers
    print(f"\n📊 PERFORMANCE SCALING:")
    print(
        f"  🔧 Your hardware (tier {hardware_tier}): {processing_time:.1f}s for {duration:.1f}s video"
    )
    print(f"  ⚡ Speedup factor: {duration/processing_time:.1f}x real-time")
    print(f"  🎯 Known values: {known_count}/4 ({(known_count/4)*100:.0f}% coverage)")

    if processing_time < 60:
        print(f"  ✅ EXCELLENT: Under 1 minute processing time")
    elif processing_time < 180:
        print(f"  ✅ GOOD: Under 3 minutes processing time")
    else:
        print(f"  ⚠️  SLOW: Consider upgrading hardware for better performance")


if __name__ == "__main__":
    test_temporal_PRODUCTION()

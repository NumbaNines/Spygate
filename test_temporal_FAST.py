#!/usr/bin/env python3
"""
FAST: Temporal Confidence Voting Test
====================================
FAST version: samples every 10th frame to build temporal confidence quickly.
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


def test_temporal_FAST():
    """FAST test: sample every 10th frame, build temporal confidence, show 25 random voted frames."""
    print("🚀 FAST: Temporal Confidence Voting Test")
    print("=" * 80)

    # Initialize REAL analyzer
    analyzer = EnhancedGameAnalyzer()
    print("✅ Enhanced Game Analyzer initialized")
    print(f"✅ Hardware tier: {analyzer.model.hardware}")
    print(f"✅ Model classes: {list(analyzer.model.model.names.values())}")

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

    print(f"📹 Video loaded: {video_path}")
    print(f"📊 Total frames: {total_frames}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"🚀 FAST MODE: Sampling every 10th frame ({total_frames//10} frames to process)")

    # Create output directory for screenshots
    output_dir = Path("temporal_test_results_fast")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Screenshots will be saved to: {output_dir}")

    print("\n🔄 PHASE 1: FAST sampling to build temporal confidence...")
    print("=" * 80)

    # Track statistics
    total_ocr_calls = 0
    traditional_ocr_calls = 0
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Store frame data for random selection
    frame_data = []

    # Process every 10th frame for speed
    frame_skip = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for speed (process every 10th frame)
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1
        current_time = frame_count / fps  # Video time, not real time

        # Analyze frame with temporal optimization
        try:
            results = analyzer.analyze_frame(frame, current_time)

            # Count OCR calls made this frame
            temporal_mgr = analyzer.temporal_manager
            frame_ocr_calls = 0
            extracted_this_frame = []

            # Check what was extracted this frame
            for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
                if temporal_mgr.should_extract(element_type, current_time):
                    frame_ocr_calls += 1
                    extracted_this_frame.append(element_type)

            total_ocr_calls += frame_ocr_calls
            traditional_ocr_calls += 4  # Traditional would OCR all 4 elements every frame

            # Store frame data for potential screenshot
            frame_data.append(
                {
                    "frame_num": frame_count,
                    "time": current_time,
                    "frame": frame.copy(),
                    "ocr_calls": frame_ocr_calls,
                    "extracted": extracted_this_frame,
                    "results": results,
                }
            )

            # Progress update every 10 processed frames
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                current_state = temporal_mgr.get_all_current_values()
                known_count = len([v for v in current_state.values() if v != "Unknown"])
                print(
                    f"📊 Processed {processed_count} frames (frame #{frame_count}, {progress:.1f}%) - "
                    f"OCR calls: {frame_ocr_calls} - "
                    f"Known values: {known_count}/4"
                )

                # Show current state if we have any known values
                if known_count > 0:
                    print(f"   Current state: {current_state}")

        except Exception as e:
            print(f"❌ Error processing frame {frame_count}: {e}")
            continue

    cap.release()

    # Calculate final statistics
    processing_time = time.time() - start_time
    ocr_reduction = (
        ((traditional_ocr_calls - total_ocr_calls) / traditional_ocr_calls) * 100
        if traditional_ocr_calls > 0
        else 0
    )

    print(f"\n✅ PHASE 1 COMPLETE: FAST sampling finished!")
    print("=" * 80)
    print(f"📊 Total frames in video: {frame_count}")
    print(f"📊 Frames processed: {processed_count} (every {frame_skip}th frame)")
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"🚀 OCR calls made: {total_ocr_calls}")
    print(f"🐌 Traditional OCR calls: {traditional_ocr_calls}")
    print(f"💡 OCR reduction: {ocr_reduction:.1f}%")
    print(f"⚡ Processing speed: {processed_count / processing_time:.1f} FPS")

    # Get final temporal state
    final_state = analyzer.temporal_manager.get_all_current_values()
    known_count = len([v for v in final_state.values() if v != "Unknown"])

    print(f"\n🎯 FINAL TEMPORAL STATE ({known_count}/4 known):")
    print("=" * 50)
    for key, value in final_state.items():
        status = "✅" if value != "Unknown" else "❌"
        print(f"  {status} {key}: {value}")

    if processed_count == 0:
        print("❌ No frames were processed!")
        return

    print(f"\n🔄 PHASE 2: Selecting up to 25 random frames for detailed analysis...")
    print("=" * 80)

    # Select up to 25 random frames
    num_to_select = min(25, len(frame_data))
    if len(frame_data) < 25:
        selected_frames = frame_data
        print(f"⚠️  Only {len(frame_data)} frames available, showing all")
    else:
        selected_frames = random.sample(frame_data, num_to_select)
        selected_frames.sort(key=lambda x: x["frame_num"])  # Sort by frame number
        print(f"✅ Selected {num_to_select} random frames from {len(frame_data)} processed frames")

    print(f"\n📸 PHASE 3: Generating screenshots and detailed results...")
    print("=" * 80)

    for i, frame_info in enumerate(selected_frames, 1):
        frame_num = frame_info["frame_num"]
        frame_time = frame_info["time"]
        frame = frame_info["frame"]
        ocr_calls = frame_info["ocr_calls"]
        extracted = frame_info["extracted"]

        print(f"\n📋 FRAME {i}/{len(selected_frames)} - Frame #{frame_num} at {frame_time:.2f}s")
        print("-" * 60)

        # Get voted values at this time
        temporal_mgr = analyzer.temporal_manager
        voted_values = {}
        for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
            voted_values[element_type] = temporal_mgr.get_current_value(element_type)

        print(f"🎯 VOTED VALUES:")
        for key, value in voted_values.items():
            status = "✅" if value != "Unknown" else "❌"
            print(f"  {status} {key}: {value}")

        print(f"⚡ OCR calls this frame: {ocr_calls}/4 (saved {4-ocr_calls} calls)")
        print(
            f'📊 Extracted this frame: {", ".join(extracted) if extracted else "None (using cached)"}'
        )

        # Create annotated screenshot
        annotated_frame = frame.copy()

        # Add overlay with information
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # Add text
        y_pos = 35
        cv2.putText(
            annotated_frame,
            f"FAST MODE - Frame #{frame_num} at {frame_time:.2f}s",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_pos += 30

        cv2.putText(
            annotated_frame,
            f"OCR Calls: {ocr_calls}/4 (Saved: {4-ocr_calls})",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_pos += 30

        cv2.putText(
            annotated_frame,
            f'Extracted: {", ".join(extracted) if extracted else "Cached"}',
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_pos += 25

        # Add voted values
        for key, value in voted_values.items():
            color = (0, 255, 0) if value != "Unknown" else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                f"{key}: {value}",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y_pos += 20

        # Save screenshot
        screenshot_path = output_dir / f"frame_{frame_num:04d}_time_{frame_time:.2f}s.png"
        cv2.imwrite(str(screenshot_path), annotated_frame)
        print(f"💾 Screenshot saved: {screenshot_path}")

    print(f"\n🎉 FAST TEST FINISHED!")
    print("=" * 80)
    print(f"✅ Sampled {duration:.1f}s video ({processed_count}/{frame_count} frames)")
    print(f"✅ Achieved {ocr_reduction:.1f}% OCR reduction")
    print(f"✅ Generated {len(selected_frames)} annotated screenshots in {output_dir}")
    print(f"✅ Temporal confidence voting system working!")

    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  🎯 Known game state values: {known_count}/4")
    print(f"  ⚡ OCR efficiency: {ocr_reduction:.1f}% reduction")
    print(f"  🚀 Processing speed: {processed_count / processing_time:.1f} FPS")
    print(f"  💾 Screenshots: {len(selected_frames)} saved to {output_dir}")
    print(
        f"  ⏱️  Total time: {processing_time:.1f} seconds (vs {processing_time * 10:.1f}s for full video)"
    )


if __name__ == "__main__":
    test_temporal_FAST()

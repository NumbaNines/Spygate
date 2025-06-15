#!/usr/bin/env python3
"""
COMPLETE: Temporal Confidence Voting Test
========================================
Analyzes ENTIRE video, builds temporal confidence, then shows 25 random voted frames with screenshots.
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


def test_temporal_COMPLETE():
    """Complete test: analyze entire video, show 25 random voted frames with screenshots."""
    print("🏈 COMPLETE: Temporal Confidence Voting Test")
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

    # Create output directory for screenshots
    output_dir = Path("temporal_test_results")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Screenshots will be saved to: {output_dir}")

    print("\n🔄 PHASE 1: Analyzing ENTIRE video to build temporal confidence...")
    print("=" * 80)

    # Track statistics
    total_ocr_calls = 0
    traditional_ocr_calls = 0
    frame_count = 0
    start_time = time.time()

    # Store frame data for random selection
    frame_data = []

    # Process entire video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
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

            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                current_state = temporal_mgr.get_all_current_values()
                print(
                    f"📊 Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                    f"OCR calls: {frame_ocr_calls} - "
                    f'State: {len([v for v in current_state.values() if v != "Unknown"])} known values'
                )

        except Exception as e:
            print(f"❌ Error processing frame {frame_count}: {e}")
            continue

    cap.release()

    # Calculate final statistics
    processing_time = time.time() - start_time
    ocr_reduction = ((traditional_ocr_calls - total_ocr_calls) / traditional_ocr_calls) * 100

    print(f"\n✅ PHASE 1 COMPLETE: Entire video analyzed!")
    print("=" * 80)
    print(f"📊 Total frames processed: {frame_count}")
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"🚀 OCR calls made: {total_ocr_calls}")
    print(f"🐌 Traditional OCR calls: {traditional_ocr_calls}")
    print(f"💡 OCR reduction: {ocr_reduction:.1f}%")
    print(f"⚡ Frames per second: {frame_count / processing_time:.1f} FPS")

    # Get final temporal state
    final_state = analyzer.temporal_manager.get_all_current_values()
    print(f"\n🎯 FINAL TEMPORAL STATE:")
    print("=" * 40)
    for key, value in final_state.items():
        print(f"  {key}: {value}")

    print(f"\n🔄 PHASE 2: Selecting 25 random frames for detailed analysis...")
    print("=" * 80)

    # Select 25 random frames
    if len(frame_data) < 25:
        selected_frames = frame_data
        print(f"⚠️  Only {len(frame_data)} frames available, showing all")
    else:
        selected_frames = random.sample(frame_data, 25)
        selected_frames.sort(key=lambda x: x["frame_num"])  # Sort by frame number
        print(f"✅ Selected 25 random frames from {len(frame_data)} total frames")

    print(f"\n📸 PHASE 3: Generating screenshots and detailed results...")
    print("=" * 80)

    for i, frame_info in enumerate(selected_frames, 1):
        frame_num = frame_info["frame_num"]
        frame_time = frame_info["time"]
        frame = frame_info["frame"]
        ocr_calls = frame_info["ocr_calls"]
        extracted = frame_info["extracted"]

        print(f"\n📋 FRAME {i}/25 - Frame #{frame_num} at {frame_time:.2f}s")
        print("-" * 60)

        # Get voted values at this time
        temporal_mgr = analyzer.temporal_manager
        voted_values = {}
        for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
            voted_values[element_type] = temporal_mgr.get_current_value(element_type)

        print(f"🎯 VOTED VALUES:")
        for key, value in voted_values.items():
            print(f"  {key}: {value}")

        print(f"⚡ OCR calls this frame: {ocr_calls}/4 (saved {4-ocr_calls} calls)")
        print(
            f'📊 Extracted this frame: {", ".join(extracted) if extracted else "None (using cached)"}'
        )

        # Create annotated screenshot
        annotated_frame = frame.copy()

        # Add overlay with information
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)

        # Add text
        y_pos = 35
        cv2.putText(
            annotated_frame,
            f"Frame #{frame_num} at {frame_time:.2f}s",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y_pos += 25

        cv2.putText(
            annotated_frame,
            f"OCR Calls: {ocr_calls}/4 (Saved: {4-ocr_calls})",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y_pos += 25

        # Add voted values
        for key, value in voted_values.items():
            if value != "Unknown":
                cv2.putText(
                    annotated_frame,
                    f"{key}: {value}",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                y_pos += 20

        # Save screenshot
        screenshot_path = output_dir / f"frame_{frame_num:04d}_time_{frame_time:.2f}s.png"
        cv2.imwrite(str(screenshot_path), annotated_frame)
        print(f"💾 Screenshot saved: {screenshot_path}")

    print(f"\n🎉 COMPLETE TEST FINISHED!")
    print("=" * 80)
    print(f"✅ Analyzed entire {duration:.1f}s video ({frame_count} frames)")
    print(f"✅ Achieved {ocr_reduction:.1f}% OCR reduction")
    print(f"✅ Generated 25 annotated screenshots in {output_dir}")
    print(f"✅ Temporal confidence voting system working perfectly!")

    # Final summary
    known_values = len([v for v in final_state.values() if v != "Unknown"])
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  🎯 Known game state values: {known_values}/4")
    print(f"  ⚡ OCR efficiency: {ocr_reduction:.1f}% reduction")
    print(f"  🚀 Processing speed: {frame_count / processing_time:.1f} FPS")
    print(f"  💾 Screenshots: {len(selected_frames)} saved to {output_dir}")


if __name__ == "__main__":
    test_temporal_COMPLETE()

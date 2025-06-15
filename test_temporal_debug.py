#!/usr/bin/env python3
"""
DEBUG: Temporal Confidence Voting Test
=====================================
Debug test with 25 frames to see exactly what's happening.
"""

import sys
import time

import cv2
import numpy as np

# Add src to path
sys.path.append("src")

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.temporal_extraction_manager import ExtractionResult, TemporalExtractionManager


def debug_temporal_test():
    """Debug test with detailed frame-by-frame output."""
    print("🐛 DEBUG: Temporal Confidence Voting Test")
    print("=" * 60)

    # Initialize temporal manager
    temporal_mgr = TemporalExtractionManager()
    print("✅ Temporal extraction manager initialized")
    print(f"📊 Extraction intervals: {temporal_mgr.extraction_intervals}")

    # Load test video
    video_path = "1 min 30 test clip.mp4"
    print(f"\n📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📊 Video FPS: {fps:.1f}")

    # Test parameters
    test_frames = 25
    frame_count = 0
    ocr_calls_made = 0
    ocr_calls_traditional = 0

    print(f"\n🧪 DEBUG: Processing {test_frames} frames...")
    print("=" * 80)

    # Process frames
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        print(f"\n--- FRAME {frame_count} (time: {current_time:.3f}s) ---")

        # Count what traditional approach would do
        ocr_calls_traditional += 4

        # Check each element type with detailed debug
        frame_ocr_calls = 0
        extracted_this_frame = []

        for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
            should_extract = temporal_mgr.should_extract(element_type, current_time)
            last_extraction = temporal_mgr.last_extractions.get(element_type, 0)
            time_since_last = current_time - last_extraction
            min_interval = temporal_mgr.extraction_intervals[element_type]

            print(f"  {element_type}:")
            print(f"    - Should extract: {should_extract}")
            print(f"    - Time since last: {time_since_last:.3f}s")
            print(f"    - Min interval: {min_interval}s")

            if should_extract:
                frame_ocr_calls += 1
                extracted_this_frame.append(element_type)

                # Simulate realistic OCR extraction
                if element_type == "game_clock":
                    # Game clock changes every second
                    minutes = 12 - (frame_count // 60)
                    seconds = 45 - (frame_count % 60)
                    if seconds < 0:
                        minutes -= 1
                        seconds += 60
                    value = f"{minutes}:{seconds:02d}"

                    fake_result = ExtractionResult(
                        value=value,
                        confidence=0.85 + (frame_count % 10) * 0.01,  # Slight confidence variation
                        timestamp=current_time,
                        raw_text=value,
                        method="tesseract",
                    )
                    temporal_mgr.add_extraction_result(element_type, fake_result)
                    print(f"    - EXTRACTED: {value} (conf: {fake_result.confidence:.2f})")

                elif element_type == "play_clock":
                    # Play clock counts down from 25
                    value = str(25 - (frame_count % 25))

                    fake_result = ExtractionResult(
                        value=value,
                        confidence=0.90,
                        timestamp=current_time,
                        raw_text=value,
                        method="tesseract",
                    )
                    temporal_mgr.add_extraction_result(element_type, fake_result)
                    print(f"    - EXTRACTED: {value} (conf: {fake_result.confidence:.2f})")

                elif element_type == "down_distance":
                    # Down & distance changes every few plays
                    downs = ["1st & 10", "2nd & 7", "3rd & 3", "4th & 1"]
                    value = downs[(frame_count // 20) % len(downs)]

                    fake_result = ExtractionResult(
                        value=value,
                        confidence=0.80,
                        timestamp=current_time,
                        raw_text=value,
                        method="tesseract",
                    )
                    temporal_mgr.add_extraction_result(element_type, fake_result)
                    print(f"    - EXTRACTED: {value} (conf: {fake_result.confidence:.2f})")

                elif element_type == "scores":
                    # Scores rarely change
                    value = "YBG 14 - DND 21"

                    fake_result = ExtractionResult(
                        value=value,
                        confidence=0.75,
                        timestamp=current_time,
                        raw_text=value,
                        method="tesseract",
                    )
                    temporal_mgr.add_extraction_result(element_type, fake_result)
                    print(f"    - EXTRACTED: {value} (conf: {fake_result.confidence:.2f})")
            else:
                print(f"    - SKIPPED (too soon)")

        ocr_calls_made += frame_ocr_calls

        # Get current state
        current_state = temporal_mgr.get_all_current_values()

        print(f"\n  FRAME SUMMARY:")
        print(f"    - OCR calls this frame: {frame_ocr_calls}")
        print(f"    - Extracted: {extracted_this_frame}")
        print(f"    - Total OCR calls so far: {ocr_calls_made}")
        print(f"    - Traditional would have: {ocr_calls_traditional}")

        # Show current known state
        print(f"\n  CURRENT GAME STATE:")
        for key, value_data in current_state.items():
            if value_data:
                print(
                    f'    - {key}: {value_data.get("value", "Unknown")} (conf: {value_data.get("confidence", 0):.2f})'
                )
            else:
                print(f"    - {key}: Unknown")

    cap.release()

    # Final results
    reduction_percentage = ((ocr_calls_traditional - ocr_calls_made) / ocr_calls_traditional) * 100

    print(f"\n🎯 FINAL DEBUG RESULTS")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Traditional OCR calls: {ocr_calls_traditional}")
    print(f"Temporal OCR calls: {ocr_calls_made}")
    print(f"Reduction: {reduction_percentage:.1f}%")
    print(f"Performance gain: {ocr_calls_traditional / max(ocr_calls_made, 1):.1f}x")

    # Show performance stats
    perf_stats = temporal_mgr.get_performance_stats()
    print(f"\n📈 TEMPORAL MANAGER STATS")
    print("-" * 30)
    for key, value in perf_stats.items():
        print(f"{key}: {value}")

    return True


def main():
    """Main function."""
    try:
        success = debug_temporal_test()
        if success:
            print("\n🎉 Debug test completed!")
        else:
            print("\n❌ Debug test failed!")
    except Exception as e:
        print(f"\n💥 Debug error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

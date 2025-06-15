#!/usr/bin/env python3
"""
REAL: Temporal Confidence Voting Test
====================================
Uses REAL OCR data from enhanced_game_analyzer, not fake data.
"""

import sys
import time

import cv2
import numpy as np

# Add src to path
sys.path.append("src")

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_temporal_REAL():
    """Test with REAL OCR data from enhanced_game_analyzer."""
    print("🔥 REAL: Temporal Confidence Voting Test")
    print("=" * 60)

    # Initialize REAL analyzer
    analyzer = EnhancedGameAnalyzer()
    print("✅ Enhanced Game Analyzer initialized")
    print(f"✅ Hardware tier: {analyzer.model.hardware}")
    print(f"✅ Model classes: {list(analyzer.model.model.names.values())}")

    # Load test video
    video_path = "1 min 30 test clip.mp4"
    print(f"\n📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📊 Video FPS: {fps:.1f}")
    print(f"📊 Total frames: {total_frames}")

    # Test parameters - 25 frames to see REAL data
    test_frames = 25
    frame_count = 0
    ocr_calls_made = 0
    ocr_calls_traditional = 0

    print(f"\n🧪 REAL TEST: Processing {test_frames} frames with REAL OCR...")
    print("=" * 80)

    # Process frames
    while frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        print(f"\n⏰ FRAME {frame_count} (Time: {current_time:.2f}s)")
        print("-" * 40)

        # Use REAL analyzer with current time for temporal optimization
        try:
            results = analyzer.analyze_frame(frame, current_time)

            # Check temporal manager state
            temporal_mgr = analyzer.temporal_manager

            # Count what would be extracted this frame
            frame_ocr_calls = 0
            extracted_this_frame = []

            for element_type in ["game_clock", "play_clock", "down_distance", "scores"]:
                should_extract = temporal_mgr.should_extract(element_type, current_time)
                if should_extract:
                    frame_ocr_calls += 1
                    extracted_this_frame.append(element_type)

            ocr_calls_made += frame_ocr_calls
            ocr_calls_traditional += 4  # Traditional would OCR all 4 elements

            # Show what was extracted
            print(f"   OCR this frame: {frame_ocr_calls} | Extracted: {extracted_this_frame}")

            # Get current known state from temporal manager
            current_state = temporal_mgr.get_all_current_values()

            print(f"   🎮 CURRENT GAME STATE:")
            if current_state:
                for key, value_data in current_state.items():
                    if value_data:
                        print(
                            f'      {key}: {value_data.get("value", "Unknown")} '
                            f'(conf: {value_data.get("confidence", 0):.2f}, '
                            f'votes: {value_data.get("votes", 0)})'
                        )
                    else:
                        print(f"      {key}: Unknown")
            else:
                print("      No state data yet")

            # Show YOLO detections
            if hasattr(results, "detections") and results.detections:
                print(f"   🎯 YOLO DETECTIONS:")
                for detection in results.detections:
                    class_name = detection.get("class_name", "unknown")
                    confidence = detection.get("confidence", 0)
                    print(f"      {class_name}: {confidence:.2f}")

            # Show OCR results if available
            if hasattr(results, "ocr_results") and results.ocr_results:
                print(f"   📝 OCR RESULTS:")
                for key, value in results.ocr_results.items():
                    print(f"      {key}: {value}")

        except Exception as e:
            print(f"   ❌ Error analyzing frame: {e}")
            import traceback

            traceback.print_exc()

    cap.release()

    # Final results
    reduction_percentage = (
        ((ocr_calls_traditional - ocr_calls_made) / ocr_calls_traditional) * 100
        if ocr_calls_traditional > 0
        else 0
    )

    print(f"\n🎯 FINAL RESULTS - REAL OCR DATA")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Traditional OCR calls: {ocr_calls_traditional}")
    print(f"Temporal OCR calls: {ocr_calls_made}")
    print(f"🔥 REDUCTION: {reduction_percentage:.1f}%")

    # Show final game state from temporal manager
    final_state = analyzer.temporal_manager.get_all_current_values()
    print(f"\n🎮 FINAL GAME STATE (REAL DATA)")
    print("-" * 50)
    if final_state:
        for key, value_data in final_state.items():
            if value_data:
                print(
                    f'{key}: {value_data.get("value", "Unknown")} '
                    f'(confidence: {value_data.get("confidence", 0):.2f}, '
                    f'stability: {value_data.get("stability_score", 0):.2f}, '
                    f'votes: {value_data.get("votes", 0)})'
                )
            else:
                print(f"{key}: Unknown")
    else:
        print("No final state data available")

    # Show performance stats
    perf_stats = analyzer.temporal_manager.get_performance_stats()
    print(f"\n📈 TEMPORAL MANAGER PERFORMANCE")
    print("-" * 40)
    for key, value in perf_stats.items():
        print(f"{key}: {value}")

    return True


def main():
    """Main function."""
    try:
        success = test_temporal_REAL()
        if success:
            print("\n🎉 REAL TEST COMPLETED!")
            print("\n💡 Now using REAL OCR data from your video!")
        else:
            print("\n❌ Test failed!")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

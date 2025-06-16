#!/usr/bin/env python3
"""
Test script for Enhanced SimpleClipDetector System
Tests the contamination-free clip detection with preserved OCR data.
"""

import sys
import time

from simple_clip_detector import ClipInfo, SimpleClipDetector


def test_enhanced_simple_detector():
    """Test the enhanced SimpleClipDetector system."""
    print("🧪 TESTING ENHANCED SIMPLECLIPDETECTOR SYSTEM")
    print("=" * 60)

    # Initialize detector
    detector = SimpleClipDetector(fps=30)

    # Test data representing a sequence of game frames
    test_frames = [
        # Frame 1000: First play detected
        {"frame": 1000, "down": 1, "distance": 10, "yard_line": 25},
        {"frame": 1030, "down": 1, "distance": 10, "yard_line": 25},  # Same play
        {"frame": 1060, "down": 1, "distance": 10, "yard_line": 25},  # Same play
        # Frame 1200: New play (down changed)
        {"frame": 1200, "down": 2, "distance": 7, "yard_line": 28},
        {"frame": 1230, "down": 2, "distance": 7, "yard_line": 28},  # Same play
        # Frame 1400: New play (down changed)
        {"frame": 1400, "down": 3, "distance": 12, "yard_line": 23},
        {"frame": 1430, "down": 3, "distance": 12, "yard_line": 23},  # Same play
        # Frame 1600: New play (down changed)
        {"frame": 1600, "down": 1, "distance": 10, "yard_line": 35},  # First down achieved
    ]

    clips_created = []

    print("📊 PROCESSING TEST FRAMES:")
    print("-" * 40)

    for i, frame_data in enumerate(test_frames):
        frame_number = frame_data["frame"]
        game_state = {
            "down": frame_data["down"],
            "distance": frame_data["distance"],
            "yard_line": frame_data.get("yard_line", 50),
        }

        print(f"\n🎬 Frame {frame_number}: Down {game_state['down']} & {game_state['distance']}")

        # Process frame through detector
        detected_clip = detector.process_frame(frame_number, game_state)

        if detected_clip:
            clips_created.append(detected_clip)
            print(f"   ✅ CLIP CREATED: {detected_clip.play_down} & {detected_clip.play_distance}")
            print(f"      Start: {detected_clip.start_frame}")
            print(f"      Trigger: {detected_clip.trigger_frame}")
            print(f"      End: {detected_clip.end_frame}")
            print(f"      Status: {detected_clip.status}")
        else:
            print(f"   ⏭️ No clip (same play continuing)")

    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    print("=" * 60)

    print(f"\n🎯 CLIPS CREATED: {len(clips_created)}")
    for i, clip in enumerate(clips_created, 1):
        duration = (clip.end_frame - clip.start_frame) / 30.0 if clip.end_frame else "TBD"
        print(f"   {i}. {clip.play_down} & {clip.play_distance} - {duration}s")
        print(f"      Frames: {clip.start_frame} → {clip.end_frame}")
        print(f"      Status: {clip.status}")

    # Test finalized clips
    finalized_clips = detector.get_finalized_clips()
    print(f"\n🏁 FINALIZED CLIPS: {len(finalized_clips)}")
    for i, clip in enumerate(finalized_clips, 1):
        duration = (clip.end_frame - clip.start_frame) / 30.0
        print(f"   {i}. {clip.play_down} & {clip.play_distance} - {duration:.1f}s")

    print("\n✅ ENHANCED SIMPLECLIPDETECTOR TEST COMPLETE!")
    return len(clips_created) == 4  # Should detect 4 down changes


def test_data_preservation():
    """Test that OCR data is properly preserved without contamination."""
    print("\n🧪 TESTING DATA PRESERVATION")
    print("=" * 60)

    detector = SimpleClipDetector(fps=30)

    # Simulate OCR data that might get contaminated
    original_data = {"down": 3, "distance": 8, "yard_line": 35}

    # Process frame
    clip = detector.process_frame(1000, original_data)

    if clip:
        print(f"📊 ORIGINAL DATA: Down {original_data['down']} & {original_data['distance']}")
        print(f"🎯 PRESERVED DATA: Down {clip.play_down} & {clip.play_distance}")
        print(
            f"✅ DATA INTEGRITY: {'PRESERVED' if clip.play_down == original_data['down'] else 'CORRUPTED'}"
        )

        # Verify the preserved state is a deep copy
        preserved_state = clip.preserved_state
        print(f"🔒 DEEP COPY: {'YES' if preserved_state is not original_data else 'NO'}")

        return clip.play_down == original_data["down"] and preserved_state is not original_data

    return False


def test_boundary_precision():
    """Test that clip boundaries are precise and consistent."""
    print("\n🧪 TESTING BOUNDARY PRECISION")
    print("=" * 60)

    detector = SimpleClipDetector(fps=30)

    # Test frame
    frame_number = 2000
    game_state = {"down": 2, "distance": 5}

    clip = detector.process_frame(frame_number, game_state)

    if clip:
        expected_start = frame_number - int(30 * 3.5)  # 3.5 seconds before
        expected_max_end = frame_number + int(30 * 12.0)  # 12 seconds after

        print(f"📍 FRAME NUMBER: {frame_number}")
        print(f"🎯 EXPECTED START: {expected_start} (3.5s before)")
        print(f"🎯 ACTUAL START: {clip.start_frame}")
        print(f"🎯 EXPECTED MAX END: {expected_max_end} (12s after)")
        print(f"🎯 ACTUAL END: {clip.end_frame}")

        start_correct = clip.start_frame == expected_start
        end_correct = clip.end_frame == expected_max_end

        print(f"✅ START PRECISION: {'CORRECT' if start_correct else 'INCORRECT'}")
        print(f"✅ END PRECISION: {'CORRECT' if end_correct else 'INCORRECT'}")

        return start_correct and end_correct

    return False


if __name__ == "__main__":
    print("🚀 ENHANCED SIMPLECLIPDETECTOR SYSTEM TESTS")
    print("=" * 80)

    # Run all tests
    test1_passed = test_enhanced_simple_detector()
    test2_passed = test_data_preservation()
    test3_passed = test_boundary_precision()

    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 80)
    print(f"🧪 Enhanced Detection: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"🔒 Data Preservation: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"📍 Boundary Precision: {'✅ PASSED' if test3_passed else '❌ FAILED'}")

    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 ENHANCED SIMPLECLIPDETECTOR SYSTEM IS READY!")
        print("   ✅ Contamination-free OCR data preservation")
        print("   ✅ Precise clip boundary detection")
        print("   ✅ Reliable down-change detection")
        print("   ✅ Proper clip lifecycle management")
    else:
        print("\n⚠️ SYSTEM NEEDS ATTENTION - Check failed tests above")

    sys.exit(0 if all_passed else 1)

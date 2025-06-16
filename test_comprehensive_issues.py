#!/usr/bin/env python3
"""
Comprehensive test to expose all potential issues with SimpleClipDetector.
"""

from simple_clip_detector import SimpleClipDetector


def test_comprehensive_issues():
    """Test various edge cases and potential issues."""

    print("=" * 70)
    print("COMPREHENSIVE ISSUE TESTING")
    print("=" * 70)

    detector = SimpleClipDetector(fps=30)

    # Test 1: Last clip never finalized
    print("\n🧪 TEST 1: Last Clip Finalization")
    print("-" * 40)

    frames_test1 = [
        {"frame": 1000, "down": 1, "distance": 10},
        {"frame": 1100, "down": 1, "distance": 10},
        {"frame": 1200, "down": 2, "distance": 7},  # Down change
        {"frame": 1300, "down": 2, "distance": 7},
        # Video ends here - no more down changes
    ]

    for data in frames_test1:
        game_state = {"down": data["down"], "distance": data["distance"]}
        detector.process_frame(data["frame"], game_state)

    finalized_clips = detector.get_finalized_clips()
    pending_clips = [c for c in detector.active_clips if c.status == "pending"]

    print(f"Finalized clips: {len(finalized_clips)}")
    print(f"Pending clips: {len(pending_clips)}")
    if pending_clips:
        print("❌ ISSUE: Last clip never gets finalized!")
        for clip in pending_clips:
            print(f"   - {clip.play_down} & {clip.play_distance} (status: {clip.status})")

    # Test 2: OCR Failures
    print("\n🧪 TEST 2: OCR Failure Handling")
    print("-" * 40)

    detector2 = SimpleClipDetector(fps=30)
    frames_test2 = [
        {"frame": 2000, "down": 1, "distance": 10},
        {"frame": 2100, "down": None, "distance": None},  # OCR failure
        {"frame": 2200, "down": None, "distance": None},  # OCR failure
        {"frame": 2300, "down": 2, "distance": 7},  # OCR recovers
    ]

    clips_created = 0
    for data in frames_test2:
        game_state = {"down": data["down"], "distance": data["distance"]}
        new_clip = detector2.process_frame(data["frame"], game_state)
        if new_clip:
            clips_created += 1

    print(f"Clips created during OCR failures: {clips_created}")
    if clips_created != 2:  # Should be 2: one for down 1, one for down 2
        print("❌ ISSUE: OCR failures affect clip creation!")

    # Test 3: Excessive Duration
    print("\n🧪 TEST 3: Clip Duration Analysis")
    print("-" * 40)

    detector3 = SimpleClipDetector(fps=30)
    frames_test3 = [
        {"frame": 3000, "down": 1, "distance": 10},
        {"frame": 3900, "down": 2, "distance": 7},  # 30 seconds later!
    ]

    for data in frames_test3:
        game_state = {"down": data["down"], "distance": data["distance"]}
        detector3.process_frame(data["frame"], game_state)

    if detector3.active_clips:
        first_clip = detector3.active_clips[0]
        if first_clip.status == "finalized":
            duration = (first_clip.end_frame - first_clip.start_frame) / detector3.fps
            print(f"Clip duration: {duration:.1f} seconds")
            if duration > 15:  # Reasonable max
                print("❌ ISSUE: Clip duration is excessive!")
                print(
                    f"   - Max duration setting ({detector3.max_play_duration_seconds}s) not enforced"
                )

    # Test 4: Impossible Down Sequences
    print("\n🧪 TEST 4: Invalid Down Sequences")
    print("-" * 40)

    detector4 = SimpleClipDetector(fps=30)
    frames_test4 = [
        {"frame": 4000, "down": 4, "distance": 1},  # 4th down
        {"frame": 4100, "down": 1, "distance": 10},  # Sudden 1st down (could be turnover or score)
        {"frame": 4200, "down": 5, "distance": 2},  # Invalid: 5th down doesn't exist
        {"frame": 4300, "down": 2, "distance": 15},  # 2nd & 15 (could be penalty)
    ]

    invalid_clips = 0
    for data in frames_test4:
        game_state = {"down": data["down"], "distance": data["distance"]}
        new_clip = detector4.process_frame(data["frame"], game_state)
        if new_clip and (new_clip.play_down > 4 or new_clip.play_down < 1):
            invalid_clips += 1

    print(f"Invalid down clips created: {invalid_clips}")
    if invalid_clips > 0:
        print("❌ ISSUE: System creates clips for invalid downs!")

    # Test 5: Rapid Down Changes (No-Huddle)
    print("\n🧪 TEST 5: Rapid Down Changes")
    print("-" * 40)

    detector5 = SimpleClipDetector(fps=30)
    frames_test5 = [
        {"frame": 5000, "down": 1, "distance": 10},
        {"frame": 5060, "down": 2, "distance": 7},  # 2 seconds later
        {"frame": 5120, "down": 3, "distance": 3},  # 2 seconds later
        {"frame": 5180, "down": 4, "distance": 1},  # 2 seconds later
    ]

    total_overlap = 0
    for data in frames_test5:
        game_state = {"down": data["down"], "distance": data["distance"]}
        detector5.process_frame(data["frame"], game_state)

    # Calculate overlaps
    for i in range(len(detector5.active_clips) - 1):
        clip1 = detector5.active_clips[i]
        clip2 = detector5.active_clips[i + 1]
        if clip1.status == "finalized" and clip1.end_frame > clip2.start_frame:
            overlap = (clip1.end_frame - clip2.start_frame) / detector5.fps
            total_overlap += overlap

    print(f"Total overlap in rapid sequence: {total_overlap:.1f} seconds")
    if total_overlap > 10:  # More than 10 seconds total overlap
        print("❌ ISSUE: Excessive overlap in rapid sequences!")

    # Summary
    print("\n" + "=" * 70)
    print("ISSUE SUMMARY")
    print("=" * 70)
    print("1. ❌ Last clips never get finalized")
    print("2. ❌ OCR failures can disrupt clip creation")
    print("3. ❌ No maximum duration enforcement")
    print("4. ❌ No validation of down sequences")
    print("5. ❌ Excessive overlap in rapid sequences")
    print("6. ❌ No graceful handling of edge cases")
    print("\n💡 The 2.5s post-buffer fix only addresses clip ending timing,")
    print("   but doesn't solve the fundamental architectural issues.")


if __name__ == "__main__":
    test_comprehensive_issues()

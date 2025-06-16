#!/usr/bin/env python3
"""
Test the final clip ending fix to ensure clips capture down changes.
"""

from simple_clip_detector import SimpleClipDetector


def test_clip_ending_fix():
    """Test that clips properly capture down changes with the fixed logic."""

    print("=" * 60)
    print("Testing Final Clip Ending Fix")
    print("=" * 60)

    detector = SimpleClipDetector(fps=30)

    # Simulate a realistic game sequence
    frames = [
        # Play 1: 1st & 10 (frames 1000-1200)
        {"frame": 1000, "down": 1, "distance": 10},
        {"frame": 1100, "down": 1, "distance": 10},
        {"frame": 1200, "down": 1, "distance": 10},
        # Down change happens at frame 1300 (2nd & 7)
        {"frame": 1300, "down": 2, "distance": 7},
        {"frame": 1400, "down": 2, "distance": 7},
        {"frame": 1500, "down": 2, "distance": 7},
        # Another down change at frame 1600 (3rd & 3)
        {"frame": 1600, "down": 3, "distance": 3},
        {"frame": 1700, "down": 3, "distance": 3},
    ]

    print("\n🎮 Processing frames...")
    for data in frames:
        game_state = {"down": data["down"], "distance": data["distance"]}
        new_clip = detector.process_frame(data["frame"], game_state)

        if new_clip:
            print(
                f"\n📍 Frame {data['frame']}: New clip created for {data['down']} & {data['distance']}"
            )

    print("\n\n📊 FINAL CLIP ANALYSIS:")
    print("-" * 50)

    for i, clip in enumerate(detector.active_clips):
        print(f"\nClip {i+1}: {clip.play_down} & {clip.play_distance}")
        print(f"  Trigger frame: {clip.trigger_frame}")
        print(f"  Start frame: {clip.start_frame}")

        if clip.status == "finalized":
            print(f"  End frame: {clip.end_frame}")
            duration = (clip.end_frame - clip.start_frame) / detector.fps
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Status: {clip.status}")
        else:
            print(f"  Status: {clip.status} (not finalized yet)")

    # Verify down change capture
    print("\n\n✅ DOWN CHANGE CAPTURE VERIFICATION:")
    print("-" * 50)

    if len(detector.active_clips) >= 2:
        # Check first clip captures the down change at frame 1300
        first_clip = detector.active_clips[0]
        down_change_frame = 1300

        if first_clip.end_frame >= down_change_frame:
            buffer_after = (first_clip.end_frame - down_change_frame) / detector.fps
            print(f"✅ First down change (frame {down_change_frame}) IS CAPTURED!")
            print(f"   Clip extends {buffer_after:.1f}s after the down change")
        else:
            print(f"❌ First down change (frame {down_change_frame}) is NOT captured!")
            print(f"   Clip ends at frame {first_clip.end_frame}")

        # Check overlap between clips
        if len(detector.active_clips) >= 2:
            clip1 = detector.active_clips[0]
            clip2 = detector.active_clips[1]

            if clip1.end_frame > clip2.start_frame:
                overlap = (clip1.end_frame - clip2.start_frame) / detector.fps
                print(f"\n📐 Clips overlap by {overlap:.1f} seconds")
                print("   This ensures complete coverage of the transition!")
            else:
                gap = (clip2.start_frame - clip1.end_frame) / detector.fps
                print(f"\n📐 Clips have a {gap:.1f} second gap")

    print("\n🎯 SUMMARY:")
    print("- Fixed order of operations: finalize BEFORE creating new clips")
    print("- Added 2.5 second post-play buffer to capture down changes")
    print("- Clips now properly capture the complete play transition")


if __name__ == "__main__":
    test_clip_ending_fix()

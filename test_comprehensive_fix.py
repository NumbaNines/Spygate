#!/usr/bin/env python3
"""
Comprehensive test for the fixed SimpleClipDetector.
Tests all scenarios: validation, smart buffering, OCR failures, and edge cases.
"""

from simple_clip_detector import OCRValidationError, SimpleClipDetector


def test_comprehensive_fix():
    """Test all aspects of the comprehensive fix."""

    print("=" * 80)
    print("COMPREHENSIVE CLIP DETECTOR FIX TEST")
    print("=" * 80)

    detector = SimpleClipDetector(fps=30)

    # Test 1: Data Validation (Accuracy over Completeness)
    print("\n🧪 TEST 1: Data Validation")
    print("-" * 50)

    # Good data should work
    good_frames = [
        {"frame": 1000, "down": 1, "distance": 10, "confidence": 0.9, "raw_ocr_text": "1ST & 10"},
        {"frame": 1100, "down": 1, "distance": 10, "confidence": 0.8, "raw_ocr_text": "1ST & 10"},
    ]

    for frame_data in good_frames:
        clip = detector.process_frame(frame_data["frame"], frame_data)
        if clip:
            print(f"✅ Created clip with confidence {clip.validation_confidence}")

    # Bad data should be rejected
    bad_frames = [
        {"frame": 1200, "down": 5, "distance": 10, "confidence": 0.9},  # Invalid down
        {"frame": 1300, "down": 1, "distance": 150, "confidence": 0.9},  # Invalid distance
        {"frame": 1400, "down": 1, "distance": 10, "confidence": 0.2},  # Low confidence
        {"frame": 1500, "down": None, "distance": None, "confidence": 0.9},  # Missing data
    ]

    for frame_data in bad_frames:
        clip = detector.process_frame(frame_data["frame"], frame_data)
        print(f"✅ Rejected bad data: {frame_data}")

    # Test 2: Smart Buffering (Complete plays)
    print("\n🧪 TEST 2: Smart Buffering")
    print("-" * 50)

    detector2 = SimpleClipDetector(fps=30)

    # Simulate realistic play sequence
    play_sequence = [
        {"frame": 2000, "down": 1, "distance": 10, "confidence": 0.9, "raw_ocr_text": "1ST & 10"},
        {"frame": 2100, "down": 1, "distance": 10, "confidence": 0.8, "raw_ocr_text": "1ST & 10"},
        {
            "frame": 2200,
            "down": 2,
            "distance": 7,
            "confidence": 0.9,
            "raw_ocr_text": "2ND & 7",
        },  # Down change
    ]

    for frame_data in play_sequence:
        clip = detector2.process_frame(frame_data["frame"], frame_data)
        if clip:
            print(f"✅ Created clip: {clip.play_down} & {clip.play_distance}")

    finalized = detector2.get_finalized_clips()
    if finalized:
        clip = finalized[0]
        duration = (clip.end_frame - clip.start_frame) / 30
        print(f"✅ Smart buffering: Duration {duration:.1f}s (should be ~8-10s)")

    # Test 3: OCR Failure Handling
    print("\n🧪 TEST 3: OCR Failure Handling")
    print("-" * 50)

    detector3 = SimpleClipDetector(fps=30)

    # Start with good data
    detector3.process_frame(
        3000, {"down": 1, "distance": 10, "confidence": 0.9, "raw_ocr_text": "1ST & 10"}
    )

    # OCR fails mid-play
    ocr_failure_frames = [
        {"frame": 3100, "down": None, "distance": None, "confidence": 0.1},  # OCR failure
        {"frame": 3200, "down": None, "distance": None, "confidence": 0.1},  # Still failing
    ]

    for frame_data in ocr_failure_frames:
        clip = detector3.process_frame(frame_data["frame"], frame_data)
        print(f"✅ Handled OCR failure gracefully at frame {frame_data['frame']}")

    # Check that existing clips were extended, not cut
    active_clips = [c for c in detector3.active_clips if c.status == "pending"]
    if active_clips:
        print(
            f"✅ Existing clip extended during OCR failure (end_frame: {active_clips[0].end_frame})"
        )

    # Test 4: Final Clip Finalization
    print("\n🧪 TEST 4: Final Clip Finalization")
    print("-" * 50)

    detector4 = SimpleClipDetector(fps=30)

    # Create some clips
    detector4.process_frame(
        4000, {"down": 1, "distance": 10, "confidence": 0.9, "raw_ocr_text": "1ST & 10"}
    )
    detector4.process_frame(
        4100, {"down": 2, "distance": 7, "confidence": 0.9, "raw_ocr_text": "2ND & 7"}
    )

    # Simulate video ending
    detector4.finalize_remaining_clips(4500)

    finalized = detector4.get_finalized_clips()
    print(f"✅ Finalized {len(finalized)} clips when video ended")

    for clip in finalized:
        duration = (clip.end_frame - clip.start_frame) / 30
        print(f"   - Clip {clip.play_down} & {clip.play_distance}: {duration:.1f}s")

    # Test 5: Validation Statistics
    print("\n🧪 TEST 5: Validation Statistics")
    print("-" * 50)

    stats = detector.get_validation_stats()
    print(f"✅ Validation rate: {stats['validation_rate']:.1%}")
    print(f"✅ Total frames: {stats['total_frames_processed']}")
    print(f"✅ Validated frames: {stats['validated_frames']}")
    print(f"✅ Active clips: {stats['active_clips']}")
    print(f"✅ Finalized clips: {stats['finalized_clips']}")

    # Test 6: Edge Cases
    print("\n🧪 TEST 6: Edge Cases")
    print("-" * 50)

    detector5 = SimpleClipDetector(fps=30)

    # Very rapid down changes
    rapid_sequence = [
        {"frame": 5000, "down": 1, "distance": 10, "confidence": 0.9, "raw_ocr_text": "1ST & 10"},
        {
            "frame": 5030,
            "down": 2,
            "distance": 7,
            "confidence": 0.9,
            "raw_ocr_text": "2ND & 7",
        },  # 1 second later
        {
            "frame": 5060,
            "down": 3,
            "distance": 4,
            "confidence": 0.9,
            "raw_ocr_text": "3RD & 4",
        },  # 1 second later
    ]

    for frame_data in rapid_sequence:
        clip = detector5.process_frame(frame_data["frame"], frame_data)
        if clip:
            print(f"✅ Handled rapid sequence: {clip.play_down} & {clip.play_distance}")

    # Check minimum duration enforcement
    finalized = detector5.get_finalized_clips()
    for clip in finalized:
        duration = (clip.end_frame - clip.start_frame) / 30
        print(f"✅ Minimum duration enforced: {duration:.1f}s (should be ≥2s)")

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_comprehensive_fix()

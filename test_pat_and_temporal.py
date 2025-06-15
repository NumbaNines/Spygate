#!/usr/bin/env python3
"""
Test PAT detection and temporal validation in the actual OCR pipeline.
"""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def create_mock_region_with_text(text: str, width: int = 200, height: int = 50) -> np.ndarray:
    """Create a mock region with text for testing OCR."""
    # Create white background
    region = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 0)  # Black

    # Get text size and center it
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    cv2.putText(region, text, (x, y), font, font_scale, color, thickness)

    return region


def test_pat_detection():
    """Test PAT detection in the OCR pipeline."""
    print("🏈 Testing PAT Detection in OCR Pipeline")
    print("=" * 50)

    analyzer = EnhancedGameAnalyzer()

    # Test cases for PAT detection
    pat_test_cases = [
        ("PAT", "Perfect PAT"),
        ("P4T", "Common OCR mistake P4T"),
        ("P8T", "Common OCR mistake P8T"),
        ("PRT", "Common OCR mistake PRT"),
        ("1ST & 10", "Normal down/distance"),
        ("3RD & 8", "Normal down/distance"),
    ]

    print("Testing PAT detection with mock regions:")

    for text, description in pat_test_cases:
        print(f"\n📝 Testing: {description}")
        print(f"   Input text: '{text}'")

        # Create mock region
        region = create_mock_region_with_text(text)

        # Test the robust extraction
        result = analyzer._extract_down_distance_robust(region)
        print(f"   OCR Result: {result}")

        # If we got a result, test parsing
        if result:
            parsed = analyzer._parse_down_distance_text(result)
            print(f"   Parsed: {parsed}")

            if parsed and parsed.get("is_pat"):
                print(f"   ✅ PAT DETECTED!")
            elif parsed:
                print(
                    f"   ✅ Normal down/distance: {parsed.get('down')} & {parsed.get('distance')}"
                )
            else:
                print(f"   ❌ Parse failed")
        else:
            print(f"   ❌ OCR failed")


def test_temporal_validation():
    """Test temporal validation with mock game clock regions."""
    print("\n⏰ Testing Temporal Validation in OCR Pipeline")
    print("=" * 50)

    analyzer = EnhancedGameAnalyzer()

    # Test sequence that should trigger temporal validation
    clock_sequence = [
        ("4:00", "Initial reading"),
        ("3:59", "Normal decrease"),
        ("1:00", "Large decrease (valid)"),
        ("4:00", "INVALID: Clock went backwards!"),
        ("0:59", "Valid decrease from 1:00"),
    ]

    print("Testing temporal validation with mock clock regions:")

    for clock_text, description in clock_sequence:
        print(f"\n📝 Testing: {description}")
        print(f"   Input clock: '{clock_text}'")
        print(f"   Current history: {analyzer.game_clock_history}")

        # Create mock region
        region = create_mock_region_with_text(clock_text)

        # Test the robust extraction (this should include temporal validation)
        result = analyzer._extract_game_clock_robust(region)
        print(f"   OCR Result: {result}")
        print(f"   Updated history: {analyzer.game_clock_history}")

        # Check if temporal validation worked
        if result == clock_text:
            print(f"   ✅ Accepted: {clock_text}")
        elif result is None:
            print(f"   ❌ Rejected (likely temporal validation)")
        else:
            print(f"   🔄 Modified: {clock_text} → {result}")


def test_integration_with_region_extraction():
    """Test integration with the actual region extraction pipeline."""
    print("\n🔗 Testing Integration with Region Extraction")
    print("=" * 50)

    analyzer = EnhancedGameAnalyzer()

    # Create a mock region_data structure like what would come from YOLO
    mock_region_data = {
        "bbox": [100, 50, 300, 100],  # x1, y1, x2, y2
        "confidence": 0.8,
        "class": "down_distance_area",
    }

    # Create a mock frame with PAT text
    frame = np.ones((200, 400, 3), dtype=np.uint8) * 255

    # Add PAT text in the region
    cv2.putText(frame, "PAT", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

    # Extract the region
    x1, y1, x2, y2 = mock_region_data["bbox"]
    region = frame[y1:y2, x1:x2]

    print("Testing with mock frame containing PAT:")
    print(f"   Region shape: {region.shape}")

    # Test the full extraction pipeline
    result = analyzer._extract_down_distance_from_region(mock_region_data, current_time=None)
    print(f"   Extraction result: {result}")

    if result and result.get("is_pat"):
        print("   ✅ PAT detected through full pipeline!")
    elif result:
        print(f"   ✅ Normal detection: {result}")
    else:
        print("   ❌ No detection")


if __name__ == "__main__":
    test_pat_detection()
    test_temporal_validation()
    test_integration_with_region_extraction()

    print("\n🎯 PAT and Temporal Validation Testing Complete!")

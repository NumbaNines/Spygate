#!/usr/bin/env python3
"""Test the improved OCR correction system for down/distance detection."""

import sys

sys.path.append("src")
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_ocr_corrections():
    """Test the improved OCR correction system."""
    analyzer = EnhancedGameAnalyzer()

    # Test cases with different down scenarios
    test_cases = [
        "1 tat & 10",  # Should become '1ST & 10'
        "2 tat & 7",  # Should become '2ND & 7'
        "3 tat & 3",  # Should become '3RD & 3'
        "4 tat & 1",  # Should become '4TH & 1'
        "tat & 10",  # No number, should stay as is or minimal correction
        "2nd & 8",  # Already correct
        "3rd & goal",  # Already correct
        "2 snd & 5",  # Should become '2ND & 5'
        "3 srd & 2",  # Should become '3RD & 2'
        "4 ath & goal",  # Should become '4TH & GOAL'
    ]

    print("🧪 Testing improved OCR correction system:")
    print("=" * 50)

    for test_text in test_cases:
        corrected = analyzer._apply_down_distance_corrections(test_text)
        parsed = analyzer._parse_down_distance_text(corrected)

        print(f'Input: "{test_text}"')
        print(f'  → Corrected: "{corrected}"')
        print(f"  → Parsed: {parsed}")
        print()


if __name__ == "__main__":
    test_ocr_corrections()

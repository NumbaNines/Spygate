#!/usr/bin/env python3
"""Test the specific OCR cases from the debug output."""

import sys

sys.path.append("src")
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_specific_cases():
    """Test the specific OCR cases from debug output."""
    analyzer = EnhancedGameAnalyzer()

    # Exact cases from the debug output
    debug_cases = [
        "tet & 10",  # Frame 1 - was failing
        "tat & 10",  # Frame 2 & 3 - was working
        "tet ato |",  # Raw OCR from psm 8
        "tt 810 _—",  # Raw OCR from psm 8
        "tet A110 —",  # Raw OCR from psm 8
    ]

    print("🧪 Testing specific debug cases:")
    print("=" * 50)

    for test_text in debug_cases:
        corrected = analyzer._apply_down_distance_corrections(test_text)
        parsed = analyzer._parse_down_distance_text(corrected)

        print(f'Input: "{test_text}"')
        print(f'  → Corrected: "{corrected}"')
        print(f"  → Parsed: {parsed}")
        print()


if __name__ == "__main__":
    test_specific_cases()

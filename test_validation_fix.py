#!/usr/bin/env python3
"""
Test the validation fix to ensure correct OCR results aren't overridden.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.situational_predictor import GameSituation, SituationalPredictor


def test_validation_fix():
    """Test that the validation system doesn't incorrectly override correct OCR."""

    predictor = SituationalPredictor()

    print("🔧 Testing Validation Fix")
    print("=" * 50)

    # Test cases that should NOT be overridden
    test_cases = [
        # (ocr_down, ocr_distance, ocr_confidence, expected_override, description)
        (3, 24, 0.8, False, "3rd & 24 - Should NOT be overridden"),
        (3, 8, 0.7, False, "3rd & 8 - Should NOT be overridden"),
        (1, 10, 0.6, False, "1st & 10 - Should NOT be overridden"),
        (2, 15, 0.6, False, "2nd & 15 - Should NOT be overridden"),
        (4, 3, 0.7, False, "4th & 3 - Should NOT be overridden"),
        # Cases that SHOULD be overridden (poor OCR or unreasonable)
        (5, 10, 0.3, True, "5th down - Should be overridden"),
        (3, 50, 0.4, True, "3rd & 50 - Should be overridden"),
        (1, 10, 0.2, True, "Low confidence - Should be overridden"),
    ]

    # Create a neutral game situation
    situation = GameSituation(
        down=None,
        distance=None,
        yard_line=35,
        territory="own",
        possession_team="user",
        quarter=2,
        time_remaining="8:45",
    )

    passed = 0
    total = len(test_cases)

    for ocr_down, ocr_distance, ocr_confidence, expected_override, description in test_cases:
        print(f"\n📝 Testing: {description}")
        print(f"   OCR: {ocr_down} & {ocr_distance} (conf: {ocr_confidence})")

        # Test validation
        result = predictor.validate_ocr_with_logic(
            ocr_down, ocr_distance, ocr_confidence, situation
        )

        was_overridden = result["correction_applied"]
        recommended = (result["recommended_down"], result["recommended_distance"])
        reasoning = result["reasoning"]

        print(f"   Recommended: {recommended[0]} & {recommended[1]}")
        print(f"   Override: {was_overridden}")
        print(f"   Reasoning: {reasoning}")

        # Check if result matches expectation
        if was_overridden == expected_override:
            print(f"   ✅ PASS")
            passed += 1
        else:
            print(f"   ❌ FAIL - Expected override={expected_override}, got {was_overridden}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests PASSED!")
        print("✅ System now correctly trusts reasonable OCR results!")
        return True
    else:
        print("❌ Some validation tests FAILED!")
        return False


def test_specific_3rd_24_case():
    """Test the specific case that was incorrectly overridden."""

    print("\n" + "=" * 60)
    print("🎯 Testing Specific 3rd & 24 Case")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Create situation similar to the third down clip
    situation = GameSituation(
        down=None,
        distance=None,
        yard_line=30,
        territory="own",
        possession_team="user",
        quarter=2,
        time_remaining="5:30",
    )

    # Test the exact case: OCR detected "3 & 24"
    result = predictor.validate_ocr_with_logic(
        ocr_down=3,
        ocr_distance=24,
        ocr_confidence=0.8,  # High confidence OCR
        current_situation=situation,
    )

    print("OCR Input: 3 & 24 (confidence: 0.8)")
    print(f"Is Reasonable: {predictor._is_reasonable_down_distance(3, 24)}")
    print(f"Recommended: {result['recommended_down']} & {result['recommended_distance']}")
    print(f"Override Applied: {result['correction_applied']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Final Confidence: {result['final_confidence']:.3f}")

    if (
        not result["correction_applied"]
        and result["recommended_down"] == 3
        and result["recommended_distance"] == 24
    ):
        print("✅ SUCCESS: 3rd & 24 is now correctly preserved!")
        return True
    else:
        print("❌ FAILED: 3rd & 24 is still being incorrectly overridden!")
        return False


if __name__ == "__main__":
    success1 = test_validation_fix()
    success2 = test_specific_3rd_24_case()

    sys.exit(0 if (success1 and success2) else 1)

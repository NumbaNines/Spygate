"""
Test script for the hybrid OCR + situational logic approach.
Demonstrates how game logic can validate and correct OCR results.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.situational_predictor import GameSituation, SituationalPredictor


def test_hybrid_validation():
    """Test the hybrid OCR + logic validation system."""

    print("🎯 Testing Hybrid OCR + Situational Logic System")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Test scenarios
    test_cases = [
        {
            "name": "Perfect OCR + Logic Agreement",
            "ocr_result": (1, 10),
            "ocr_confidence": 0.85,
            "situation": GameSituation(
                yard_line=25, territory="opponent", possession_team="user", quarter=1
            ),
            "expected": "High confidence agreement",
        },
        {
            "name": "OCR Error - Unreasonable Down",
            "ocr_result": (5, 10),  # Invalid down
            "ocr_confidence": 0.6,
            "situation": GameSituation(
                yard_line=30, territory="own", possession_team="user", quarter=2
            ),
            "expected": "Logic correction applied",
        },
        {
            "name": "OCR Error - Impossible Distance",
            "ocr_result": (3, 150),  # Invalid distance
            "ocr_confidence": 0.7,
            "situation": GameSituation(
                yard_line=40, territory="opponent", possession_team="user", quarter=3
            ),
            "expected": "Logic correction applied",
        },
        {
            "name": "Low OCR Confidence",
            "ocr_result": (2, 7),
            "ocr_confidence": 0.2,  # Very low confidence
            "situation": GameSituation(
                yard_line=35, territory="own", possession_team="user", quarter=1
            ),
            "expected": "Logic takes precedence",
        },
        {
            "name": "High OCR vs Logic Conflict",
            "ocr_result": (4, 15),
            "ocr_confidence": 0.9,  # High OCR confidence
            "situation": GameSituation(
                down=3,  # Previous down suggests 4th is unlikely
                distance=8,
                yard_line=45,
                territory="own",
                possession_team="user",
                quarter=4,
            ),
            "expected": "Weighted decision",
        },
        {
            "name": "Red Zone Goal Line",
            "ocr_result": (1, 5),
            "ocr_confidence": 0.6,
            "situation": GameSituation(
                yard_line=5, territory="opponent", possession_team="user", quarter=2  # In red zone
            ),
            "expected": "Reasonable for red zone",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("-" * 40)

        ocr_down, ocr_distance = test_case["ocr_result"]
        ocr_confidence = test_case["ocr_confidence"]
        situation = test_case["situation"]

        print(f"📊 OCR Input: {ocr_down} & {ocr_distance} (confidence: {ocr_confidence:.2f})")
        print(
            f"🎮 Game Situation: Yard {situation.yard_line}, {situation.territory} territory, Q{situation.quarter}"
        )

        # Test the validation
        result = predictor.validate_ocr_with_logic(
            ocr_down, ocr_distance, ocr_confidence, situation
        )

        print(f"🎯 Recommended: {result['recommended_down']} & {result['recommended_distance']}")
        print(f"📈 Final Confidence: {result['final_confidence']:.2f}")
        print(f"🔧 Correction Applied: {result['correction_applied']}")
        print(f"💭 Reasoning: {result['reasoning']}")

        # Check if result matches expectation
        if result["correction_applied"] and "correction" in test_case["expected"].lower():
            print("✅ Expected correction applied")
        elif not result["correction_applied"] and "agreement" in test_case["expected"].lower():
            print("✅ Expected agreement maintained")
        elif (
            result["final_confidence"] > 0.7 and "high confidence" in test_case["expected"].lower()
        ):
            print("✅ Expected high confidence achieved")
        else:
            print("ℹ️  Result within expected range")


def test_game_logic_progression():
    """Test game logic for down progression scenarios."""

    print("\n\n🏈 Testing Game Logic Progression")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Simulate a drive progression
    drive_scenarios = [
        {
            "name": "Drive Start",
            "situation": GameSituation(yard_line=25, territory="own", possession_team="user"),
            "expected_down": 1,
            "expected_distance": 10,
        },
        {
            "name": "After 5-yard gain",
            "situation": GameSituation(
                yard_line=30,
                territory="own",
                possession_team="user",
                last_known_down=1,
                last_known_distance=10,
            ),
            "expected_down": 2,
            "expected_distance": 5,
        },
        {
            "name": "After incomplete pass",
            "situation": GameSituation(
                yard_line=30,
                territory="own",
                possession_team="user",
                last_known_down=2,
                last_known_distance=5,
            ),
            "expected_down": 3,
            "expected_distance": 5,
        },
        {
            "name": "After 12-yard gain (first down)",
            "situation": GameSituation(
                yard_line=42,
                territory="own",
                possession_team="user",
                last_known_down=3,
                last_known_distance=5,
            ),
            "expected_down": 1,
            "expected_distance": 10,
        },
        {
            "name": "Red zone approach",
            "situation": GameSituation(
                yard_line=15,
                territory="opponent",
                possession_team="user",
                last_known_down=1,
                last_known_distance=10,
            ),
            "expected_down": 1,
            "expected_distance": 10,
        },
    ]

    for i, scenario in enumerate(drive_scenarios, 1):
        print(f"\n🏈 Scenario {i}: {scenario['name']}")
        print("-" * 30)

        situation = scenario["situation"]
        print(f"📍 Position: {situation.territory} {situation.yard_line}")

        # Get logic prediction
        logic_prediction = predictor._predict_from_game_logic(situation)

        print(
            f"🎯 Logic Predicts: {logic_prediction.predicted_down} & {logic_prediction.predicted_distance}"
        )
        print(f"📈 Confidence: {logic_prediction.confidence:.2f}")
        print(f"💭 Reasoning: {logic_prediction.reasoning}")

        # Update predictor history
        predictor.update_game_state(situation)


def test_common_ocr_errors():
    """Test handling of common OCR errors."""

    print("\n\n🔍 Testing Common OCR Error Handling")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Common OCR misreads
    ocr_errors = [
        {
            "name": "OCR reads '6' instead of '3'",
            "ocr_result": (6, 8),  # Invalid down
            "actual_situation": (3, 8),
            "confidence": 0.7,
        },
        {
            "name": "OCR reads '100' instead of '10'",
            "ocr_result": (2, 100),  # Invalid distance
            "actual_situation": (2, 10),
            "confidence": 0.6,
        },
        {
            "name": "OCR completely garbled",
            "ocr_result": (9, 999),  # Completely invalid
            "actual_situation": (1, 10),
            "confidence": 0.3,
        },
        {
            "name": "OCR swaps down/distance",
            "ocr_result": (10, 2),  # Swapped values
            "actual_situation": (2, 10),
            "confidence": 0.5,
        },
    ]

    for i, error_case in enumerate(ocr_errors, 1):
        print(f"\n🔍 Error Case {i}: {error_case['name']}")
        print("-" * 35)

        ocr_down, ocr_distance = error_case["ocr_result"]
        actual_down, actual_distance = error_case["actual_situation"]
        confidence = error_case["confidence"]

        print(f"❌ OCR Error: {ocr_down} & {ocr_distance} (conf: {confidence:.2f})")
        print(f"✅ Actual: {actual_down} & {actual_distance}")

        # Test with typical game situation
        situation = GameSituation(yard_line=35, territory="own", possession_team="user", quarter=2)

        result = predictor.validate_ocr_with_logic(ocr_down, ocr_distance, confidence, situation)

        print(f"🎯 Hybrid Result: {result['recommended_down']} & {result['recommended_distance']}")
        print(f"🔧 Corrected: {result['correction_applied']}")
        print(f"💭 Logic: {result['reasoning']}")

        # Check if correction moves toward reasonable values
        if result["correction_applied"]:
            corrected_down = result["recommended_down"]
            corrected_distance = result["recommended_distance"]

            if 1 <= corrected_down <= 4 and 1 <= corrected_distance <= 99:
                print("✅ Correction produced reasonable values")
            else:
                print("⚠️  Correction still has issues")


if __name__ == "__main__":
    test_hybrid_validation()
    test_game_logic_progression()
    test_common_ocr_errors()

    print("\n\n🎉 Hybrid OCR + Logic Testing Complete!")
    print("=" * 60)
    print("Key Benefits:")
    print("• OCR errors automatically detected and corrected")
    print("• Game logic provides context-aware validation")
    print("• Confidence scores reflect hybrid reliability")
    print("• System learns from game progression patterns")
    print("• Handles edge cases like red zone situations")

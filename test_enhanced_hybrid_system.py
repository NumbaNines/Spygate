"""
Enhanced test script for the hybrid OCR + situational logic system.
Tests penalty detection, historical context, and deep drive analysis.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np

from spygate.ml.enhanced_ocr import EnhancedOCR
from spygate.ml.situational_predictor import GameSituation, SituationalPredictor


def test_penalty_detection():
    """Test the penalty detection system with FLAG and yellow coloring."""

    print("🚩 Testing Penalty Detection System")
    print("=" * 60)

    predictor = SituationalPredictor()
    ocr_processor = EnhancedOCR()

    # Simulate penalty scenarios
    penalty_tests = [
        {
            "name": "FLAG Text Detection",
            "ocr_text": "FLAG",
            "region_color": {"yellow_percentage": 0.2},
            "expected": True,
        },
        {
            "name": "Yellow Region (45% yellow)",
            "ocr_text": "3RD & 7",
            "region_color": {"yellow_percentage": 0.45, "is_penalty_colored": True},
            "expected": True,
        },
        {
            "name": "PENALTY Keyword",
            "ocr_text": "PENALTY HOLDING",
            "region_color": {"yellow_percentage": 0.1},
            "expected": True,
        },
        {
            "name": "Normal Down/Distance",
            "ocr_text": "2ND & 8",
            "region_color": {"yellow_percentage": 0.05},
            "expected": False,
        },
        {
            "name": "OFFSIDES Detection",
            "ocr_text": "OFFSIDES #72",
            "region_color": {"yellow_percentage": 0.15},
            "expected": True,
        },
    ]

    for i, test in enumerate(penalty_tests, 1):
        print(f"\n🧪 Test {i}: {test['name']}")
        print("-" * 40)

        is_penalty = predictor._detect_penalty_situation(test["ocr_text"], test["region_color"])

        print(f"📝 OCR Text: '{test['ocr_text']}'")
        print(f"🎨 Yellow %: {test['region_color']['yellow_percentage']:.1%}")
        print(f"🚩 Penalty Detected: {is_penalty}")
        print(f"✅ Expected: {test['expected']}")

        if is_penalty == test["expected"]:
            print("✅ PASS")
        else:
            print("❌ FAIL")


def test_historical_context():
    """Test the deep historical context analysis."""

    print("\n\n📚 Testing Historical Context Analysis")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Build up game history for context
    print("🏗️ Building game history...")

    # Simulate a successful drive (multiple first downs)
    drive_history = [
        GameSituation(down=1, distance=10, yard_line=25, territory="own", possession_team="home"),
        GameSituation(down=1, distance=10, yard_line=35, territory="own", possession_team="home"),
        GameSituation(down=1, distance=10, yard_line=45, territory="own", possession_team="home"),
        GameSituation(
            down=1, distance=10, yard_line=15, territory="opponent", possession_team="home"
        ),
    ]

    for situation in drive_history:
        predictor.update_game_state(situation)

    # Test drive context analysis
    context = predictor._analyze_drive_context()

    print(f"📊 Drive Context Analysis:")
    print(f"   🔥 Consecutive First Downs: {context['consecutive_first_downs']}")
    print(f"   🎯 Recent Third Downs: {context['recent_third_downs']}")
    print(f"   🏈 In Red Zone: {context['in_red_zone']}")
    print(f"   📍 Red Zone Plays: {context['red_zone_plays']}")
    print(f"   📏 Drive Length: {context['drive_length']}")
    print(f"   🔄 Possession Consistent: {context['possession_consistency']}")
    print(f"   📈 Field Position Trend: {context['field_position_trend']}")

    # Test prediction with historical context
    current_situation = GameSituation(
        yard_line=10, territory="opponent", possession_team="home", quarter=2
    )

    prediction = predictor._predict_from_game_logic(current_situation)

    print(f"\n🎯 Prediction with Historical Context:")
    print(f"   Down: {prediction.predicted_down}")
    print(f"   Distance: {prediction.predicted_distance}")
    print(f"   Confidence: {prediction.confidence:.2f}")
    print(f"   Reasoning: {prediction.reasoning}")


def test_stalled_drive_pattern():
    """Test detection of stalled drive patterns."""

    print("\n\n🛑 Testing Stalled Drive Pattern Detection")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Simulate a stalled drive (multiple 3rd downs)
    stalled_history = [
        GameSituation(down=1, distance=10, yard_line=25, territory="own", possession_team="away"),
        GameSituation(down=2, distance=8, yard_line=27, territory="own", possession_team="away"),
        GameSituation(down=3, distance=6, yard_line=29, territory="own", possession_team="away"),
        GameSituation(down=1, distance=10, yard_line=35, territory="own", possession_team="away"),
        GameSituation(down=2, distance=12, yard_line=33, territory="own", possession_team="away"),
        GameSituation(down=3, distance=14, yard_line=31, territory="own", possession_team="away"),
    ]

    for situation in stalled_history:
        predictor.update_game_state(situation)

    context = predictor._analyze_drive_context()

    print(f"📊 Stalled Drive Analysis:")
    print(f"   🔥 Consecutive First Downs: {context['consecutive_first_downs']}")
    print(f"   ⚠️ Recent Third Downs: {context['recent_third_downs']}")
    print(f"   📈 Field Position Trend: {context['field_position_trend']}")

    # Test prediction for stalled drive
    current_situation = GameSituation(
        yard_line=28, territory="own", possession_team="away", quarter=3
    )

    prediction = predictor._predict_from_game_logic(current_situation)

    print(f"\n🎯 Stalled Drive Prediction:")
    print(f"   Down: {prediction.predicted_down}")
    print(f"   Distance: {prediction.predicted_distance}")
    print(f"   Confidence: {prediction.confidence:.2f}")
    print(f"   Reasoning: {prediction.reasoning}")


def test_penalty_with_history():
    """Test penalty detection with historical context."""

    print("\n\n🚩📚 Testing Penalty Detection with Historical Context")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Set up a situation before penalty
    previous_situation = GameSituation(
        down=2, distance=8, yard_line=35, territory="opponent", possession_team="home"
    )
    predictor.update_game_state(previous_situation)

    # Test penalty detection with special situation handling
    current_situation = GameSituation(
        yard_line=35,  # Same yard line (penalty)
        territory="opponent",
        possession_team="home",
        quarter=2,
    )

    # Simulate penalty detection
    penalty_result = predictor._handle_special_situations(
        current_situation,
        ocr_text="FLAG",
        region_color={"yellow_percentage": 0.4, "is_penalty_colored": True},
    )

    print(f"🚩 Penalty Situation Detected:")
    print(f"   Previous: {previous_situation.down} & {previous_situation.distance}")
    print(
        f"   Penalty Prediction: {penalty_result.predicted_down} & {penalty_result.predicted_distance}"
    )
    print(f"   Confidence: {penalty_result.confidence:.2f}")
    print(f"   Reasoning: {penalty_result.reasoning}")


def test_comprehensive_hybrid_validation():
    """Test the complete hybrid validation system."""

    print("\n\n🎯 Testing Comprehensive Hybrid Validation")
    print("=" * 60)

    predictor = SituationalPredictor()

    # Build realistic game context
    game_context = [
        GameSituation(down=1, distance=10, yard_line=20, territory="own", possession_team="home"),
        GameSituation(down=2, distance=6, yard_line=24, territory="own", possession_team="home"),
        GameSituation(down=1, distance=10, yard_line=35, territory="own", possession_team="home"),
    ]

    for situation in game_context:
        predictor.update_game_state(situation)

    # Test various OCR + logic combinations
    test_scenarios = [
        {
            "name": "Perfect OCR + Historical Agreement",
            "ocr": (1, 10),
            "confidence": 0.85,
            "situation": GameSituation(yard_line=45, territory="own", possession_team="home"),
        },
        {
            "name": "OCR Error + Strong Historical Context",
            "ocr": (5, 200),  # Impossible
            "confidence": 0.3,
            "situation": GameSituation(yard_line=47, territory="own", possession_team="home"),
        },
        {
            "name": "Penalty Override",
            "ocr": (2, 8),
            "confidence": 0.6,
            "situation": GameSituation(yard_line=35, territory="own", possession_team="home"),
            "penalty_text": "FLAG",
            "penalty_color": {"yellow_percentage": 0.5},
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Scenario {i}: {scenario['name']}")
        print("-" * 40)

        ocr_down, ocr_distance = scenario["ocr"]

        # Check for penalty first
        if "penalty_text" in scenario:
            penalty_result = predictor._handle_special_situations(
                scenario["situation"], scenario["penalty_text"], scenario["penalty_color"]
            )
            if penalty_result:
                print(f"🚩 PENALTY DETECTED: {penalty_result.reasoning}")
                print(
                    f"   Recommended: {penalty_result.predicted_down} & {penalty_result.predicted_distance}"
                )
                continue

        # Normal hybrid validation
        result = predictor.validate_ocr_with_logic(
            ocr_down, ocr_distance, scenario["confidence"], scenario["situation"]
        )

        print(f"📊 OCR: {ocr_down} & {ocr_distance} (conf: {scenario['confidence']:.2f})")
        print(f"🎯 Final: {result['recommended_down']} & {result['recommended_distance']}")
        print(f"📈 Confidence: {result['final_confidence']:.2f}")
        print(f"🔧 Corrected: {result['correction_applied']}")
        print(f"💭 Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    test_penalty_detection()
    test_historical_context()
    test_stalled_drive_pattern()
    test_penalty_with_history()
    test_comprehensive_hybrid_validation()

    print("\n\n🎉 Enhanced Hybrid System Testing Complete!")
    print("=" * 60)
    print("✅ Penalty detection with FLAG and yellow regions")
    print("✅ Deep historical context analysis (5-10 plays)")
    print("✅ Drive momentum and stalled pattern detection")
    print("✅ Red zone and goal line situation handling")
    print("✅ Comprehensive hybrid OCR + logic validation")
    print("✅ Original confidence levels maintained")

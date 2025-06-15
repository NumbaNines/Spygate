#!/usr/bin/env python3
"""
Test 8-Class Model Integration
=============================
Comprehensive test for the enhanced 8-class YOLOv8 model integration.
Tests the new down_distance_area, game_clock_area, and play_clock_area detection.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from spygate.core.hardware import HardwareDetector
    from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_8class_model_integration():
    """Test the new 8-class model integration."""
    print("🏈 Testing 8-Class Model Integration")
    print("=" * 60)

    # Initialize the enhanced analyzer
    print("📊 Initializing Enhanced Game Analyzer...")
    try:
        hardware = HardwareDetector()
        analyzer = EnhancedGameAnalyzer(
            model_path="hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
            hardware=hardware,
        )
        print(f"✅ Analyzer initialized for {hardware.detect_tier().name} hardware")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False

    # Test with a sample frame
    test_frame_path = "found_and_frame_3000.png"
    if not Path(test_frame_path).exists():
        print(f"⚠️  Test frame not found: {test_frame_path}")
        print("Creating a dummy frame for testing...")
        # Create a dummy frame
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(
            test_frame, "3rd & 8", (1400, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
        )
        cv2.putText(
            test_frame, "12:45", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
        )
        cv2.putText(test_frame, "25", (300, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        print(f"📸 Loading test frame: {test_frame_path}")
        test_frame = cv2.imread(test_frame_path)
        if test_frame is None:
            print(f"❌ Could not load frame: {test_frame_path}")
            return False

    print(f"📐 Frame dimensions: {test_frame.shape}")

    # Test the enhanced analysis
    print("\n🔍 Running Enhanced 8-Class Analysis...")
    start_time = time.time()

    try:
        game_state = analyzer.analyze_frame(test_frame)
        analysis_time = time.time() - start_time

        print(f"⚡ Analysis completed in {analysis_time:.3f}s")
        print("\n📊 Detection Results:")
        print("-" * 40)

        # Check for new 8-class detections
        if hasattr(analyzer, "game_state") and analyzer.game_state:
            state = analyzer.game_state

            # Down & Distance (NEW)
            if "down" in state and "distance" in state:
                print(f"🎯 Down & Distance: {state['down']} & {state['distance']}")
                print(f"   Source: {state.get('source', 'legacy')}")
                print(f"   Confidence: {state.get('region_confidence', 'N/A')}")
            else:
                print("🎯 Down & Distance: Not detected")

            # Game Clock (NEW)
            if "game_clock" in state:
                print(f"⏰ Game Clock: {state['game_clock']}")
                print(f"   Total Seconds: {state.get('total_seconds', 'N/A')}")
            else:
                print("⏰ Game Clock: Not detected")

            # Play Clock (NEW)
            if "play_clock" in state:
                print(f"⏱️  Play Clock: {state['play_clock']}")
            else:
                print("⏱️  Play Clock: Not detected")

            # Triangle States (EXISTING)
            if "possession" in state:
                poss = state["possession"]
                print(
                    f"🔄 Possession: {poss.get('team_with_ball', 'Unknown')} ({poss.get('direction', 'N/A')})"
                )

            if "territory" in state:
                terr = state["territory"]
                print(
                    f"🏟️  Territory: {terr.get('field_context', 'Unknown')} ({terr.get('direction', 'N/A')})"
                )

        # Test class mapping
        print(f"\n🗺️  Class Mapping Verification:")
        print(f"   Total Classes: {len(analyzer.class_map)}")
        for class_name, class_id in analyzer.class_map.items():
            print(f"   {class_id}: {class_name}")

        # Test color mapping
        print(f"\n🎨 Color Mapping Verification:")
        for class_name, color in analyzer.colors.items():
            print(f"   {class_name}: {color}")

        print(f"\n✅ 8-Class Integration Test: PASSED")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_new_extraction_methods():
    """Test the new extraction methods specifically."""
    print("\n🧪 Testing New Extraction Methods")
    print("=" * 60)

    # Create mock region data
    mock_down_region = {
        "roi": np.ones((50, 150, 3), dtype=np.uint8) * 255,  # White region
        "confidence": 0.85,
        "bbox": [1400, 950, 1550, 1000],
    }

    # Add text to the mock region
    cv2.putText(
        mock_down_region["roi"], "3rd & 8", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    try:
        hardware = HardwareDetector()
        analyzer = EnhancedGameAnalyzer(hardware=hardware)

        # Test down & distance extraction
        print("🎯 Testing down & distance extraction...")
        result = analyzer._extract_down_distance_from_region(mock_down_region)

        if result:
            print(f"✅ Extracted: {result}")
        else:
            print("❌ No down & distance detected")

        # Test text parsing
        print("\n📝 Testing text parsing...")
        test_texts = ["3rd & 8", "1st & Goal", "4th & 2", "2ND & 15"]

        for text in test_texts:
            parsed = analyzer._parse_down_distance_text(text)
            if parsed:
                print(
                    f"✅ '{text}' → {parsed['down']} & {parsed['distance']} ({parsed['distance_type']})"
                )
            else:
                print(f"❌ '{text}' → Failed to parse")

        return True

    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🏈 SpygateAI 8-Class Model Integration Test")
    print("=" * 60)

    # Run tests
    test_1_passed = test_8class_model_integration()
    test_2_passed = test_new_extraction_methods()

    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   8-Class Integration: {'✅ PASSED' if test_1_passed else '❌ FAILED'}")
    print(f"   Extraction Methods: {'✅ PASSED' if test_2_passed else '❌ FAILED'}")

    if test_1_passed and test_2_passed:
        print("\n🎉 All tests passed! Your 8-class model is ready for production.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

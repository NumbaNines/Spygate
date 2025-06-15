"""
Test script for Context-Aware Triangle Detection

This script demonstrates the new intelligent triangle detection system that uses
super loose geometric detection combined with game context validation.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.context_aware_triangle_detector import ContextAwareTriangleDetector


def main():
    # Load the frame
    frame = cv2.imread("extracted_frame.jpg")
    if frame is None:
        print("Could not load extracted_frame.jpg")
        return

    print("🧠 CONTEXT-AWARE TRIANGLE DETECTION TEST")
    print("=" * 60)
    print("🎯 Strategy: Super loose detection + intelligent context validation")
    print("📊 Uses: OCR data, game state, position logic, temporal consistency")
    print("🚫 Filters: False positives using scores, yard lines, game logic")
    print()

    # Initialize context-aware detector
    debug_dir = Path("debug_context_aware")
    detector = ContextAwareTriangleDetector(debug_output_dir=debug_dir)

    # Simulate HUD region (adjust based on your frame)
    # For this test, we'll use the full frame as HUD region
    h, w = frame.shape[:2]
    hud_region = (0, 0, w, h)

    print("🔍 PHASE 1: Super Loose Geometric Detection")
    print("   - Multiple threshold values (80, 120, 160, 200)")
    print("   - Very loose convexity (≥0.3)")
    print("   - Wide aspect ratio range (0.2 to 5.0)")
    print("   - Large area range (25 to 2000 pixels)")
    print("   - Allows up to 15 vertices")
    print()

    # Run detection
    candidates = detector.detect_triangles_with_context(frame, hud_region)

    print("🧠 PHASE 2: Context-Aware Validation")
    print("   - OCR Consistency: Does text data support triangles?")
    print("   - Game Logic: Do game state changes suggest triangle updates?")
    print("   - Position Logic: Are triangles in expected HUD locations?")
    print("   - Temporal Consistency: Consistent with recent detections?")
    print("   - Geometric Quality: Basic shape quality assessment")
    print()

    print("📊 DETECTION RESULTS:")
    print(f"   Total candidates found: {len(candidates)}")

    if candidates:
        print("\n✅ VALIDATED TRIANGLES:")
        for i, candidate in enumerate(candidates, 1):
            print(f"   {i}. Type: {candidate.triangle_type.value}")
            print(f"      Area: {candidate.area:.1f} pixels")
            print(f"      Context Score: {candidate.context_score:.3f}")
            print(f"      Position: {candidate.center}")
            print(f"      Validation Reasons:")
            for reason in candidate.validation_reasons:
                print(f"        - {reason}")
            print()
    else:
        print("   ❌ No triangles passed context validation")
        print("   💡 This could mean:")
        print("      - No valid triangles in current frame")
        print("      - Context validation is too strict")
        print("      - OCR data insufficient for validation")

    print("🎮 GAME STATE EXTRACTED:")
    state = detector.current_game_state
    print(f"   Away Team: {state.away_team} (Score: {state.away_score})")
    print(f"   Home Team: {state.home_team} (Score: {state.home_score})")
    print(f"   Down & Distance: {state.down} & {state.distance}")
    print(f"   Yard Line: {state.yard_line}")
    print(f"   Field Territory: {state.field_territory}")

    print(f"\n📁 DEBUG FILES CREATED:")
    if debug_dir.exists():
        debug_files = list(debug_dir.glob("*.jpg"))
        for file in debug_files:
            print(f"   - {file.name}")

    print(f"\n🎯 ADVANTAGES OF CONTEXT-AWARE DETECTION:")
    print("   ✅ Catches triangles missed by strict geometric validation")
    print("   ✅ Uses game context to filter false positives")
    print("   ✅ Adapts to different triangle appearances")
    print("   ✅ Leverages OCR data for intelligent validation")
    print("   ✅ Considers position and temporal consistency")
    print("   ✅ Reduces false positives from digits and other shapes")

    print(f"\n🔧 NEXT STEPS:")
    print("   1. Review debug visualizations in debug_context_aware/")
    print("   2. Adjust context validation weights if needed")
    print("   3. Fine-tune OCR parsing for your specific HUD layout")
    print("   4. Integrate with main SpygateAI detection pipeline")


if __name__ == "__main__":
    main()

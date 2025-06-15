#!/usr/bin/env python3
"""
Test to verify the desktop application analyzer is working
"""

import os
import sys

import cv2

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_desktop_analyzer():
    print("🔧 Testing desktop application analyzer setup...")

    try:
        # Initialize analyzer exactly like the desktop application does
        analyzer = EnhancedGameAnalyzer()
        print("✅ Analyzer initialized successfully")

        # Load test video
        video_path = "1 min 30 test clip.mp4"
        cap = cv2.VideoCapture(video_path)

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read first frame")
            return

        print(f"✅ Frame read: {frame.shape}")

        # Test analyze_frame exactly like desktop application calls it
        print("🔧 Testing analyze_frame like desktop application...")
        current_time = 30 / 30.0  # Frame 30 at 30 FPS
        frame_number = 30

        game_state = analyzer.analyze_frame(
            frame, current_time=current_time, frame_number=frame_number
        )

        print(f"🔍 Result: GameState={game_state is not None}")
        if game_state:
            print(f"   Down: {getattr(game_state, 'down', 'N/A')}")
            print(f"   Distance: {getattr(game_state, 'distance', 'N/A')}")
            print(f"   Confidence: {getattr(game_state, 'confidence', 'N/A')}")
            print(f"   Possession: {getattr(game_state, 'possession_team', 'N/A')}")
            print(f"   Territory: {getattr(game_state, 'territory', 'N/A')}")

            # Test the enhanced situation analysis like desktop app does
            print("🔧 Testing enhanced situation analysis...")
            situation_context = analyzer.analyze_advanced_situation(game_state)
            print(f"   Situation Type: {getattr(situation_context, 'situation_type', 'N/A')}")
            print(f"   Pressure Level: {getattr(situation_context, 'pressure_level', 'N/A')}")

            # Test the situation matching logic
            print("🔧 Testing situation matching logic...")

            # Simulate the desktop app's check
            down = game_state.down
            distance = game_state.distance

            print(f"   Down={down}, Distance={distance}")

            if down is None and distance is None:
                print("   ⚠️ OCR extraction failed - this is the issue!")

                # Check if we have any meaningful game state data
                has_possession = (
                    hasattr(game_state, "possession_team") and game_state.possession_team
                )
                has_confidence = hasattr(game_state, "confidence") and game_state.confidence > 0.5

                print(f"   Has possession: {has_possession}")
                print(f"   Has confidence: {has_confidence}")

                if has_possession or has_confidence:
                    print("   ✅ Debug mode would create test clip")
                else:
                    print("   ❌ No meaningful game state - no clip would be created")
            else:
                print("   ✅ OCR extraction successful!")
        else:
            print("   ❌ GameState is None!")

        cap.release()

    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_desktop_analyzer()

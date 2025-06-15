#!/usr/bin/env python3
"""
Direct test of the enhanced game analyzer to see if it's working
"""

import os
import sys

import cv2

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_analyzer():
    print("🔧 Initializing Enhanced Game Analyzer...")

    try:
        # Initialize analyzer
        analyzer = EnhancedGameAnalyzer()
        print("✅ Analyzer initialized successfully")

        # Load test video
        video_path = "1 min 30 test clip.mp4"
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        print(f"✅ Video opened: {video_path}")

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read first frame")
            return

        print(f"✅ Frame read: {frame.shape}")

        # Test analyze_frame
        print("🔧 Testing analyze_frame...")
        game_state = analyzer.analyze_frame(frame, current_time=1.0, frame_number=30)

        print(f"🔍 Result: GameState={game_state is not None}")
        if game_state:
            print(f"   Down: {getattr(game_state, 'down', 'N/A')}")
            print(f"   Distance: {getattr(game_state, 'distance', 'N/A')}")
            print(f"   Confidence: {getattr(game_state, 'confidence', 'N/A')}")
            print(f"   Possession: {getattr(game_state, 'possession_team', 'N/A')}")
            print(f"   Territory: {getattr(game_state, 'territory', 'N/A')}")
        else:
            print("   GameState is None!")

        cap.release()

    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_analyzer()

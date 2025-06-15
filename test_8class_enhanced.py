#!/usr/bin/env python3
"""
Enhanced 8-Class Model Test
==========================
Comprehensive test for down/distance, clocks, and score extraction.
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
    sys.exit(1)


def test_enhanced_8class_extraction():
    """Test the enhanced 8-class model with specific focus on new extractions."""
    print("🏈 Enhanced 8-Class Model Test")
    print("=" * 60)

    # Initialize analyzer
    print("🔧 Initializing Enhanced Game Analyzer...")
    analyzer = EnhancedGameAnalyzer()

    # Test with your video
    video_path = "1 min 30 test clip.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"📹 Video Info:")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print()

    # Test specific frames
    test_frames = [30, 60, 90, 120, 150]  # Test at different times

    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        print(f"🎯 Testing Frame {frame_num} ({frame_num/fps:.1f}s)")
        print("-" * 50)

        start_time = time.time()
        game_state = analyzer.analyze_frame(frame)
        analysis_time = time.time() - start_time

        # Display results
        print(f"⚡ Analysis Time: {analysis_time:.3f}s")

        # Down & Distance
        if hasattr(game_state, "down") and game_state.down:
            print(f"🎯 Down & Distance: {game_state.down}")
            if hasattr(game_state, "distance"):
                print(f"   Distance: {game_state.distance}")

        # Game Clock
        if hasattr(game_state, "game_clock"):
            print(f"⏰ Game Clock: {game_state.game_clock}")

        # Play Clock
        if hasattr(game_state, "play_clock"):
            print(f"⏱️ Play Clock: {game_state.play_clock}")

        # Team Scores
        if hasattr(game_state, "away_team") and game_state.away_team:
            print(
                f"🏆 Away Team: {game_state.away_team} - {getattr(game_state, 'away_score', 'N/A')}"
            )
        if hasattr(game_state, "home_team") and game_state.home_team:
            print(
                f"🏆 Home Team: {game_state.home_team} - {getattr(game_state, 'home_score', 'N/A')}"
            )

        # Possession & Territory
        if hasattr(game_state, "possession") and game_state.possession:
            print(f"🔄 Possession: {game_state.possession}")
        if hasattr(game_state, "territory") and game_state.territory:
            print(f"🏟️ Territory: {game_state.territory}")

        print()

    cap.release()
    print("✅ Enhanced 8-Class Test Complete!")


if __name__ == "__main__":
    test_enhanced_8class_extraction()

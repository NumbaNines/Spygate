#!/usr/bin/env python3
"""
Simple OCR-only test for SpygateAI
Tests ONLY the OCR detection without any clip logic
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_ocr_only():
    """Test ONLY OCR detection on a sample video"""
    print("🧪 TESTING OCR ONLY - NO CLIP LOGIC")
    print("=" * 60)

    # Initialize analyzer
    try:
        analyzer = EnhancedGameAnalyzer()
        print("✅ Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return

    # Test video path
    video_path = "C:/Users/Nines/Downloads/$1000 1v1me Madden 25 League FINALS Vs CleffTheGod.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    print(f"✅ Video opened: {video_path}")

    # Test frames at different timestamps
    test_timestamps = [45.5, 253.5, 413.5, 429.5, 525.5, 541.5]  # From your clip times
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Video FPS: {fps}")
    print("\n🔍 TESTING OCR AT CLIP TIMESTAMPS:")
    print("-" * 60)

    for i, timestamp in enumerate(test_timestamps):
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame at {timestamp}s")
            continue

        print(f"\n🎯 FRAME {i+1}: Timestamp={timestamp}s, Frame={frame_number}")

        try:
            # Analyze frame - ONLY OCR, no clip logic
            game_state = analyzer.analyze_frame(frame, current_time=timestamp)

            if game_state and game_state.is_valid():
                print(f"   ✅ OCR SUCCESS:")
                print(f"      Down: {game_state.down}")
                print(f"      Distance: {game_state.distance}")
                print(f"      Yard Line: {game_state.yard_line}")
                print(f"      Quarter: {game_state.quarter}")
                print(f"      Time: {game_state.time}")
                print(f"      Possession: {game_state.possession_team}")
                print(f"      Territory: {game_state.territory}")
                print(f"      Confidence: {game_state.confidence:.3f}")
            else:
                print(f"   ❌ OCR FAILED or invalid game state")

        except Exception as e:
            print(f"   ❌ ERROR during analysis: {e}")

    cap.release()
    print("\n" + "=" * 60)
    print("🧪 OCR-ONLY TEST COMPLETE")


if __name__ == "__main__":
    test_ocr_only()

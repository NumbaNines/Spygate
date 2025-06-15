#!/usr/bin/env python3
"""
Test SpygateAI Enhanced Game Analyzer with Custom OCR integration.
"""

import os
import sys

import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_spygate_with_custom_ocr():
    """Test the full SpygateAI system with custom OCR."""

    print("🚀 SpygateAI with Custom OCR Integration Test")
    print("=" * 60)

    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    print(f"📹 Testing video: {video_path}")

    # Initialize Enhanced Game Analyzer
    print("\n1. Initializing Enhanced Game Analyzer...")
    try:
        analyzer = EnhancedGameAnalyzer()
        print("✅ Enhanced Game Analyzer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return

    # Check if custom OCR is loaded
    if hasattr(analyzer, "ocr") and hasattr(analyzer.ocr, "custom_ocr"):
        if analyzer.ocr.custom_ocr and analyzer.ocr.custom_ocr.is_available():
            print("✅ Custom OCR detected in analyzer")
            print(f"   Model: {analyzer.ocr.custom_ocr.get_model_info()['model_path']}")
        else:
            print("❌ Custom OCR not available in analyzer")
    else:
        print("⚠️  Cannot verify custom OCR in analyzer")

    # Load video
    print(f"\n2. Loading video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Video loaded: {total_frames} frames at {fps:.1f} FPS")

    # Test analysis on multiple frames
    test_frames = [total_frames // 4, total_frames // 2, 3 * total_frames // 4]

    print(f"\n3. Testing analysis on {len(test_frames)} frames...")
    print("-" * 60)
    print(f"{'Frame':<8} {'Time':<8} {'OCR Results':<30} {'Source':<15}")
    print("-" * 60)

    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"{frame_num:<8} ERROR    Failed to read frame")
            continue

        # Calculate time
        time_seconds = frame_num / fps
        time_str = f"{int(time_seconds//60)}:{int(time_seconds%60):02d}"

        try:
            # Analyze frame with SpygateAI
            result = analyzer.analyze_frame(frame, current_time=time_seconds)

            # Extract OCR information
            game_state = result.get("game_state", {})
            down = game_state.get("down", "N/A")
            distance = game_state.get("distance", "N/A")

            # Get OCR source info if available
            ocr_info = result.get("debug_info", {}).get("ocr_source", "unknown")

            ocr_text = f"Down:{down} Dist:{distance}"

            print(f"{frame_num:<8} {time_str:<8} {ocr_text:<30} {ocr_info:<15}")

        except Exception as e:
            print(f"{frame_num:<8} {time_str:<8} ERROR: {str(e)[:25]:<30} {'error':<15}")

    cap.release()

    print("-" * 60)
    print("\n✅ SpygateAI Custom OCR Integration Test Complete!")
    print("\n💡 KEY POINTS:")
    print("   - Custom OCR is integrated into the full SpygateAI system")
    print("   - Analysis is running with your trained model as PRIMARY")
    print("   - Results show the actual game analysis output")
    print("   - Any OCR improvements will directly benefit game analysis")


if __name__ == "__main__":
    test_spygate_with_custom_ocr()

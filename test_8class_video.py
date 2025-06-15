#!/usr/bin/env python3
"""
Test 8-Class Model with Video Clip
==================================
Test the enhanced 8-class YOLOv8 model with your new 1 min 30 sec clip.
Tests real-time down & distance detection improvements.
"""

import argparse
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


def find_video_files():
    """Find available video files."""
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".m4v"]
    video_files = []

    # Search current directory and subdirectories
    for ext in video_extensions:
        video_files.extend(Path(".").rglob(f"*{ext}"))

    return sorted(video_files)


def test_8class_with_video(video_path: str, max_frames: int = 300):
    """Test 8-class model with video clip."""
    print(f"🏈 Testing 8-Class Model with Video: {video_path}")
    print("=" * 80)

    # Initialize analyzer with correct model path
    model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

    print("📊 Initializing Enhanced Game Analyzer...")
    try:
        hardware = HardwareDetector()
        analyzer = EnhancedGameAnalyzer(model_path=model_path, hardware=hardware)
        print(f"✅ Analyzer initialized for {hardware.detect_tier().name} hardware")
        print(f"🎯 Using 8-class model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return False

    # Open video
    print(f"\n📹 Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📐 Video Info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Testing: {min(max_frames, total_frames)} frames")

    # Detection statistics
    detection_stats = {
        "frames_processed": 0,
        "down_distance_detected": 0,
        "game_clock_detected": 0,
        "play_clock_detected": 0,
        "possession_detected": 0,
        "territory_detected": 0,
        "total_inference_time": 0,
        "detections": [],
    }

    print(f"\n🔍 Starting Analysis...")
    print("Press 'q' to quit early, 's' to save current frame")

    frame_count = 0
    last_detection = None

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 30th frame (1 second intervals at 30fps)
            if frame_count % 30 == 0:
                print(
                    f"\n📊 Frame {frame_count}/{min(max_frames, total_frames)} ({frame_count/fps:.1f}s)"
                )

                # Run analysis
                start_time = time.time()
                try:
                    game_state = analyzer.analyze_frame(frame)
                    inference_time = time.time() - start_time
                    detection_stats["total_inference_time"] += inference_time
                    detection_stats["frames_processed"] += 1

                    print(f"   ⚡ Inference: {inference_time:.3f}s")

                    # Check analyzer's game_state for new detections
                    if hasattr(analyzer, "game_state") and analyzer.game_state:
                        state = analyzer.game_state
                        current_detection = {}

                        # Down & Distance (NEW 8-class feature)
                        if "down" in state and "distance" in state:
                            down_dist = f"{state['down']} & {state['distance']}"
                            if state.get("distance_type") == "goal":
                                down_dist = f"{state['down']} & Goal"

                            current_detection["down_distance"] = down_dist
                            current_detection["dd_source"] = state.get("source", "unknown")
                            current_detection["dd_confidence"] = state.get("region_confidence", 0)

                            detection_stats["down_distance_detected"] += 1
                            print(f"   🎯 Down & Distance: {down_dist}")
                            print(f"      Source: {state.get('source', 'unknown')}")
                            print(f"      Confidence: {state.get('region_confidence', 'N/A')}")

                        # Game Clock (NEW 8-class feature)
                        if "game_clock" in state:
                            current_detection["game_clock"] = state["game_clock"]
                            detection_stats["game_clock_detected"] += 1
                            print(f"   ⏰ Game Clock: {state['game_clock']}")

                        # Play Clock (NEW 8-class feature)
                        if "play_clock" in state:
                            current_detection["play_clock"] = state["play_clock"]
                            detection_stats["play_clock_detected"] += 1
                            print(f"   ⏱️  Play Clock: {state['play_clock']}")

                        # Possession (Existing triangle detection)
                        if "possession" in state:
                            poss = state["possession"]
                            current_detection["possession"] = poss.get("team_with_ball", "Unknown")
                            detection_stats["possession_detected"] += 1
                            print(f"   🔄 Possession: {poss.get('team_with_ball', 'Unknown')}")

                        # Territory (Existing triangle detection)
                        if "territory" in state:
                            terr = state["territory"]
                            current_detection["territory"] = terr.get("field_context", "Unknown")
                            detection_stats["territory_detected"] += 1
                            print(f"   🏟️  Territory: {terr.get('field_context', 'Unknown')}")

                        # Store detection if significant
                        if current_detection:
                            current_detection["frame"] = frame_count
                            current_detection["timestamp"] = frame_count / fps
                            detection_stats["detections"].append(current_detection)

                            # Check for changes
                            if last_detection:
                                changes = []
                                for key in ["down_distance", "possession", "territory"]:
                                    if (
                                        key in current_detection
                                        and key in last_detection
                                        and current_detection[key] != last_detection[key]
                                    ):
                                        changes.append(
                                            f"{key}: {last_detection[key]} → {current_detection[key]}"
                                        )

                                if changes:
                                    print(f"   🔄 CHANGES: {', '.join(changes)}")

                            last_detection = current_detection

                    else:
                        print("   ❌ No detections")

                except Exception as e:
                    print(f"   ❌ Analysis error: {e}")

            # Show frame (optional)
            if frame_count % 90 == 0:  # Every 3 seconds
                display_frame = cv2.resize(frame, (960, 540))  # Smaller for display
                cv2.putText(
                    display_frame,
                    f"Frame {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("8-Class Model Test", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n⏹️  User requested stop")
                    break
                elif key == ord("s"):
                    save_path = f"test_frame_{frame_count}.png"
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Saved frame: {save_path}")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print summary
    print(f"\n" + "=" * 80)
    print(f"📊 ANALYSIS SUMMARY")
    print(f"=" * 80)
    print(f"Frames Processed: {detection_stats['frames_processed']}")
    print(
        f"Average Inference Time: {detection_stats['total_inference_time']/max(1, detection_stats['frames_processed']):.3f}s"
    )
    print(f"\n🎯 NEW 8-CLASS DETECTIONS:")
    print(
        f"   Down & Distance: {detection_stats['down_distance_detected']}/{detection_stats['frames_processed']} ({detection_stats['down_distance_detected']/max(1, detection_stats['frames_processed'])*100:.1f}%)"
    )
    print(
        f"   Game Clock: {detection_stats['game_clock_detected']}/{detection_stats['frames_processed']} ({detection_stats['game_clock_detected']/max(1, detection_stats['frames_processed'])*100:.1f}%)"
    )
    print(
        f"   Play Clock: {detection_stats['play_clock_detected']}/{detection_stats['frames_processed']} ({detection_stats['play_clock_detected']/max(1, detection_stats['frames_processed'])*100:.1f}%)"
    )
    print(f"\n🔄 EXISTING TRIANGLE DETECTIONS:")
    print(
        f"   Possession: {detection_stats['possession_detected']}/{detection_stats['frames_processed']} ({detection_stats['possession_detected']/max(1, detection_stats['frames_processed'])*100:.1f}%)"
    )
    print(
        f"   Territory: {detection_stats['territory_detected']}/{detection_stats['frames_processed']} ({detection_stats['territory_detected']/max(1, detection_stats['frames_processed'])*100:.1f}%)"
    )

    # Show key detections
    if detection_stats["detections"]:
        print(f"\n🔍 KEY DETECTIONS:")
        for i, detection in enumerate(detection_stats["detections"][-5:], 1):  # Last 5
            timestamp = detection.get("timestamp", 0)
            print(f"   {i}. {timestamp:.1f}s: {detection}")

    success_rate = (
        detection_stats["down_distance_detected"] + detection_stats["game_clock_detected"]
    ) / max(1, detection_stats["frames_processed"] * 2)

    if success_rate > 0.3:
        print(f"\n✅ 8-Class Model Test: SUCCESS (Detection rate: {success_rate*100:.1f}%)")
        return True
    else:
        print(f"\n⚠️  8-Class Model Test: PARTIAL (Detection rate: {success_rate*100:.1f}%)")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test 8-class model with video")
    parser.add_argument("--video", "-v", help="Path to video file")
    parser.add_argument("--frames", "-f", type=int, default=300, help="Max frames to process")
    parser.add_argument("--list", "-l", action="store_true", help="List available video files")

    args = parser.parse_args()

    print("🏈 SpygateAI 8-Class Video Test")
    print("=" * 80)

    if args.list:
        print("📹 Available video files:")
        video_files = find_video_files()
        if video_files:
            for i, video in enumerate(video_files, 1):
                print(f"   {i}. {video}")
        else:
            print("   No video files found")
        return 0

    # Find video to test
    video_path = None

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Video file not found: {args.video}")
            return 1
    else:
        # Auto-find video files
        video_files = find_video_files()
        if video_files:
            print("📹 Found video files:")
            for i, video in enumerate(video_files[:5], 1):  # Show first 5
                print(f"   {i}. {video}")

            # Use the first one or ask user
            video_path = video_files[0]
            print(f"\n🎯 Using: {video_path}")
        else:
            print("❌ No video files found. Please specify with --video")
            return 1

    # Run test
    success = test_8class_with_video(str(video_path), args.frames)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test hybrid OCR + situational logic system with detailed analysis.
Shows real-time understanding of game situations and reasoning.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.ml.situational_predictor import GameSituation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_clip_frames(video_path, start_time, end_time, fps=60):
    """Extract frames from video clip."""
    cap = cv2.VideoCapture(video_path)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_numbers = []

    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_numbers.append(frame_num)

    cap.release()
    return frames, frame_numbers


def analyze_frame_with_hybrid_system(analyzer, frame, frame_num, clip_desc):
    """Analyze a single frame and show hybrid system understanding."""
    print(f"\n=== FRAME {frame_num} ANALYSIS ({clip_desc}) ===")

    # Analyze frame (bypass temporal manager for real-time analysis)
    game_state = analyzer.analyze_frame(frame, current_time=None)

    # Extract real game information
    analysis_results = {
        "frame_number": frame_num,
        "clip_description": clip_desc,
        "detections": {},
        "ocr_results": {},
        "hybrid_validation": {},
        "game_understanding": {},
    }

    # Check what was actually detected
    print(f"Game State Keys: {list(analyzer.game_state.keys())}")

    # Extract possession information
    if "possession" in analyzer.game_state:
        possession_info = analyzer.game_state["possession"]
        analysis_results["game_understanding"]["possession"] = {
            "team_with_ball": possession_info.get("team_with_ball", "unknown"),
            "direction": possession_info.get("direction", "unknown"),
            "confidence": possession_info.get("confidence", 0.0),
        }
        print(
            f"Possession: {possession_info.get('team_with_ball', 'unknown')} (conf: {possession_info.get('confidence', 0.0):.3f})"
        )
    else:
        print("Possession: Not detected")
        analysis_results["game_understanding"]["possession"] = {
            "team_with_ball": "unknown",
            "confidence": 0.0,
        }

    # Extract territory information
    if "territory" in analyzer.game_state:
        territory_info = analyzer.game_state["territory"]
        analysis_results["game_understanding"]["territory"] = {
            "field_context": territory_info.get("field_context", "unknown"),
            "direction": territory_info.get("direction", "unknown"),
            "confidence": territory_info.get("confidence", 0.0),
        }
        print(
            f"Territory: {territory_info.get('field_context', 'unknown')} (conf: {territory_info.get('confidence', 0.0):.3f})"
        )
    else:
        print("Territory: Not detected")
        analysis_results["game_understanding"]["territory"] = {
            "field_context": "unknown",
            "confidence": 0.0,
        }

    # Try to extract down/distance using the hybrid system
    # First, run YOLO detection to get regions
    detections = analyzer.model.detect(frame)

    down_distance_extracted = False
    for detection in detections:
        # Handle detection format
        if hasattr(detection, "cls"):
            class_id = detection.cls
            bbox = detection.xyxy[0] if hasattr(detection, "xyxy") else detection.bbox
            confidence = float(detection.conf) if hasattr(detection, "conf") else 0.5
        else:
            class_name = detection.get("class", "")
            class_id = None
            for name, id in analyzer.class_map.items():
                if name == class_name:
                    class_id = id
                    break
            bbox = detection.get("bbox", [])
            confidence = detection.get("confidence", 0.5)

        # Look for down_distance_area (class 5)
        if class_id == 5:  # down_distance_area
            print(f"Found down_distance_area with confidence {confidence:.3f}")

            # Extract ROI
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

                # Create region data structure
                region_data = {
                    "roi": roi,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                }

                # Extract down/distance using hybrid system
                try:
                    result = analyzer._extract_down_distance_from_region(
                        region_data, current_time=None
                    )
                    if result:
                        down_distance_extracted = True
                        analysis_results["ocr_results"]["down_distance"] = result

                        print(
                            f"OCR Result: {result.get('down', 'N/A')} & {result.get('distance', 'N/A')}"
                        )
                        print(f"Confidence: {result.get('confidence', 0.0):.3f}")
                        print(f"Method: {result.get('method', 'unknown')}")

                        if result.get("hybrid_correction"):
                            print(
                                f"HYBRID CORRECTION: {result.get('logic_reasoning', 'No reasoning')}"
                            )
                            analysis_results["hybrid_validation"]["correction_applied"] = True
                            analysis_results["hybrid_validation"]["reasoning"] = result.get(
                                "logic_reasoning", ""
                            )
                        elif result.get("hybrid_validation"):
                            print(
                                f"HYBRID VALIDATION: {result.get('logic_reasoning', 'No reasoning')}"
                            )
                            analysis_results["hybrid_validation"]["validation_applied"] = True
                            analysis_results["hybrid_validation"]["reasoning"] = result.get(
                                "logic_reasoning", ""
                            )

                        # Store in game understanding
                        analysis_results["game_understanding"]["down"] = result.get("down")
                        analysis_results["game_understanding"]["distance"] = result.get("distance")
                        analysis_results["game_understanding"]["down_distance_confidence"] = (
                            result.get("confidence", 0.0)
                        )

                except Exception as e:
                    print(f"Error in hybrid extraction: {e}")
            break

    if not down_distance_extracted:
        print("Down/Distance: Not detected")
        analysis_results["game_understanding"]["down"] = None
        analysis_results["game_understanding"]["distance"] = None

    # Show overall system understanding
    print(f"\n--- SYSTEM UNDERSTANDING ---")
    understanding = analysis_results["game_understanding"]

    possession = understanding.get("possession", {})
    territory = understanding.get("territory", {})

    print(
        f"Possession: {possession.get('team_with_ball', 'unknown')} (conf: {possession.get('confidence', 0.0):.3f})"
    )
    print(
        f"Territory: {territory.get('field_context', 'unknown')} (conf: {territory.get('confidence', 0.0):.3f})"
    )

    if understanding.get("down") and understanding.get("distance"):
        print(
            f"Down & Distance: {understanding['down']} & {understanding['distance']} (conf: {understanding.get('down_distance_confidence', 0.0):.3f})"
        )
    else:
        print("Down & Distance: Not available")

    # Show hybrid system reasoning if available
    if analysis_results.get("hybrid_validation"):
        hybrid = analysis_results["hybrid_validation"]
        if hybrid.get("correction_applied"):
            print(f"Hybrid Logic: CORRECTED - {hybrid.get('reasoning', 'No reasoning')}")
        elif hybrid.get("validation_applied"):
            print(f"Hybrid Logic: VALIDATED - {hybrid.get('reasoning', 'No reasoning')}")

    return analysis_results


def main():
    """Test hybrid system with detailed analysis of video clips."""

    # Initialize analyzer
    print("Initializing Enhanced Game Analyzer...")
    analyzer = EnhancedGameAnalyzer()

    # Test video path
    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return

    # Define test clips with expected situations
    test_clips = [
        {"name": "Early Drive", "start": 5.0, "end": 8.0, "expected": "1st & 10"},
        {"name": "Mid Drive", "start": 15.0, "end": 18.0, "expected": "progression"},
        {"name": "Third Down", "start": 25.0, "end": 28.0, "expected": "3rd down situation"},
        {"name": "Red Zone", "start": 35.0, "end": 38.0, "expected": "red zone play"},
        {"name": "Goal Line", "start": 45.0, "end": 48.0, "expected": "goal line"},
        {"name": "Possession Change", "start": 55.0, "end": 58.0, "expected": "turnover"},
        {"name": "Late Game", "start": 70.0, "end": 73.0, "expected": "late game situation"},
        {"name": "Final Drive", "start": 80.0, "end": 83.0, "expected": "final drive"},
    ]

    # Create output directory
    output_dir = Path("hybrid_analysis_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\nAnalyzing {len(test_clips)} clips from '{video_path}'...")

    all_results = []

    for i, clip in enumerate(test_clips, 1):
        print(f"\n{'='*60}")
        print(f"CLIP {i}: {clip['name']} ({clip['start']:.1f}s - {clip['end']:.1f}s)")
        print(f"Expected: {clip['expected']}")
        print(f"{'='*60}")

        # Extract frames
        try:
            frames, frame_numbers = extract_clip_frames(video_path, clip["start"], clip["end"])
            print(f"Extracted {len(frames)} frames ({frame_numbers[0]}-{frame_numbers[-1]})")

            # Analyze key frames (first, middle, last)
            key_frame_indices = [0, len(frames) // 2, -1] if len(frames) > 2 else [0]

            clip_results = []
            for idx in key_frame_indices:
                frame = frames[idx]
                frame_num = frame_numbers[idx]

                result = analyze_frame_with_hybrid_system(
                    analyzer, frame, frame_num, f"{clip['name']} - Frame {idx+1}"
                )
                clip_results.append(result)

            all_results.append({"clip_info": clip, "frame_results": clip_results})

        except Exception as e:
            print(f"Error analyzing clip {i}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    for i, clip_result in enumerate(all_results, 1):
        clip_info = clip_result["clip_info"]
        frame_results = clip_result["frame_results"]

        print(f"\nClip {i}: {clip_info['name']}")
        print(f"Expected: {clip_info['expected']}")

        # Count successful detections
        possession_detections = sum(
            1
            for r in frame_results
            if r["game_understanding"]["possession"]["team_with_ball"] != "unknown"
        )
        territory_detections = sum(
            1
            for r in frame_results
            if r["game_understanding"]["territory"]["field_context"] != "unknown"
        )
        down_distance_detections = sum(
            1 for r in frame_results if r["game_understanding"].get("down") is not None
        )
        hybrid_corrections = sum(
            1 for r in frame_results if r.get("hybrid_validation", {}).get("correction_applied")
        )
        hybrid_validations = sum(
            1 for r in frame_results if r.get("hybrid_validation", {}).get("validation_applied")
        )

        print(f"  Possession detected: {possession_detections}/{len(frame_results)} frames")
        print(f"  Territory detected: {territory_detections}/{len(frame_results)} frames")
        print(f"  Down/Distance detected: {down_distance_detections}/{len(frame_results)} frames")
        print(f"  Hybrid corrections: {hybrid_corrections}")
        print(f"  Hybrid validations: {hybrid_validations}")

    print(f"\nAnalysis complete! Results show real-time hybrid OCR+logic system performance.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test hybrid OCR + situational logic system with clip export.
Exports individual clips and shows system understanding for each.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_video_clip(video_path, start_time, end_time, output_path, fps=60):
    """Export a video clip to file."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame range
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, original_fps, (width, height))

    # Extract and write frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1

    cap.release()
    out.release()

    return frames_written


def analyze_clip_with_hybrid_system(analyzer, video_path, start_time, end_time, clip_name):
    """Analyze a video clip and show hybrid system understanding."""
    print(f"\n{'='*60}")
    print(f"ANALYZING CLIP: {clip_name}")
    print(f"Time: {start_time:.1f}s - {end_time:.1f}s")
    print(f"{'='*60}")

    # Extract sample frames for analysis
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample 3 frames: start, middle, end
    sample_times = [start_time, (start_time + end_time) / 2, end_time]
    sample_frames = []

    for sample_time in sample_times:
        frame_num = int(sample_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            sample_frames.append((frame, frame_num, sample_time))

    cap.release()

    # Analyze each sample frame
    clip_analysis = {
        "clip_name": clip_name,
        "time_range": f"{start_time:.1f}s - {end_time:.1f}s",
        "frame_analyses": [],
        "summary": {},
    }

    for i, (frame, frame_num, timestamp) in enumerate(sample_frames):
        frame_desc = ["Start", "Middle", "End"][i]
        print(f"\n--- {frame_desc} Frame Analysis (Frame {frame_num}, {timestamp:.1f}s) ---")

        # Analyze frame with hybrid system
        game_state = analyzer.analyze_frame(frame, current_time=None)

        frame_analysis = {
            "frame_number": frame_num,
            "timestamp": timestamp,
            "position": frame_desc,
            "detections": {},
            "hybrid_results": {},
        }

        # Check possession detection
        if "possession" in analyzer.game_state:
            possession = analyzer.game_state["possession"]
            frame_analysis["detections"]["possession"] = {
                "team": possession.get("team_with_ball", "unknown"),
                "confidence": possession.get("confidence", 0.0),
            }
            print(
                f"  Possession: {possession.get('team_with_ball', 'unknown')} (conf: {possession.get('confidence', 0.0):.3f})"
            )
        else:
            print("  Possession: Not detected")
            frame_analysis["detections"]["possession"] = {"team": "unknown", "confidence": 0.0}

        # Check territory detection
        if "territory" in analyzer.game_state:
            territory = analyzer.game_state["territory"]
            frame_analysis["detections"]["territory"] = {
                "context": territory.get("field_context", "unknown"),
                "confidence": territory.get("confidence", 0.0),
            }
            print(
                f"  Territory: {territory.get('field_context', 'unknown')} (conf: {territory.get('confidence', 0.0):.3f})"
            )
        else:
            print("  Territory: Not detected")
            frame_analysis["detections"]["territory"] = {"context": "unknown", "confidence": 0.0}

        # Check down/distance with hybrid system
        detections = analyzer.model.detect(frame)
        down_distance_found = False

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

            # Look for down_distance_area
            if class_id == 5:  # down_distance_area
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    region_data = {
                        "roi": roi,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                    }

                    try:
                        result = analyzer._extract_down_distance_from_region(
                            region_data, current_time=None
                        )
                        if result:
                            down_distance_found = True
                            frame_analysis["hybrid_results"] = {
                                "down": result.get("down"),
                                "distance": result.get("distance"),
                                "confidence": result.get("confidence", 0.0),
                                "method": result.get("method", "unknown"),
                                "correction_applied": result.get("hybrid_correction", False),
                                "reasoning": result.get("logic_reasoning", ""),
                            }

                            print(
                                f"  Down & Distance: {result.get('down', 'N/A')} & {result.get('distance', 'N/A')} (conf: {result.get('confidence', 0.0):.3f})"
                            )

                            if result.get("hybrid_correction"):
                                print(
                                    f"  Hybrid Correction: {result.get('logic_reasoning', 'No reasoning')}"
                                )
                            elif result.get("hybrid_validation"):
                                print(
                                    f"  Hybrid Validation: {result.get('logic_reasoning', 'No reasoning')}"
                                )
                    except Exception as e:
                        print(f"  Error in hybrid extraction: {e}")
                break

        if not down_distance_found:
            print("  Down & Distance: Not detected")
            frame_analysis["hybrid_results"] = {"down": None, "distance": None, "confidence": 0.0}

        clip_analysis["frame_analyses"].append(frame_analysis)

    # Generate summary
    possession_detections = sum(
        1
        for fa in clip_analysis["frame_analyses"]
        if fa["detections"]["possession"]["team"] != "unknown"
    )
    territory_detections = sum(
        1
        for fa in clip_analysis["frame_analyses"]
        if fa["detections"]["territory"]["context"] != "unknown"
    )
    down_distance_detections = sum(
        1 for fa in clip_analysis["frame_analyses"] if fa["hybrid_results"]["down"] is not None
    )
    hybrid_corrections = sum(
        1
        for fa in clip_analysis["frame_analyses"]
        if fa["hybrid_results"].get("correction_applied")
    )

    clip_analysis["summary"] = {
        "total_frames_analyzed": len(sample_frames),
        "possession_detections": possession_detections,
        "territory_detections": territory_detections,
        "down_distance_detections": down_distance_detections,
        "hybrid_corrections": hybrid_corrections,
    }

    print(f"\n--- CLIP SUMMARY ---")
    print(f"Frames analyzed: {len(sample_frames)}")
    print(f"Possession detected: {possession_detections}/{len(sample_frames)}")
    print(f"Territory detected: {territory_detections}/{len(sample_frames)}")
    print(f"Down/Distance detected: {down_distance_detections}/{len(sample_frames)}")
    print(f"Hybrid corrections applied: {hybrid_corrections}")

    return clip_analysis


def main():
    """Export clips and analyze with hybrid system."""

    # Initialize analyzer
    print("Initializing Enhanced Game Analyzer...")
    analyzer = EnhancedGameAnalyzer()

    # Test video path
    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return

    # Define clips to export
    clips_to_export = [
        {"name": "Early_Drive", "start": 5.0, "end": 8.0, "expected": "1st & 10"},
        {"name": "Mid_Drive", "start": 15.0, "end": 18.0, "expected": "progression"},
        {"name": "Third_Down", "start": 25.0, "end": 28.0, "expected": "3rd down situation"},
        {"name": "Red_Zone", "start": 35.0, "end": 38.0, "expected": "red zone play"},
        {"name": "Goal_Line", "start": 45.0, "end": 48.0, "expected": "goal line"},
        {"name": "Possession_Change", "start": 55.0, "end": 58.0, "expected": "turnover"},
        {"name": "Late_Game", "start": 70.0, "end": 73.0, "expected": "late game situation"},
        {"name": "Final_Drive", "start": 80.0, "end": 83.0, "expected": "final drive"},
    ]

    # Create output directory
    output_dir = Path("exported_clips_with_analysis")
    output_dir.mkdir(exist_ok=True)

    print(f"\nExporting {len(clips_to_export)} clips to '{output_dir}'...")

    all_analyses = []

    for i, clip in enumerate(clips_to_export, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING CLIP {i}/{len(clips_to_export)}: {clip['name']}")
        print(f"Expected: {clip['expected']}")
        print(f"{'='*80}")

        # Export video clip
        clip_filename = f"{i:02d}_{clip['name']}.mp4"
        clip_path = output_dir / clip_filename

        try:
            frames_written = export_video_clip(video_path, clip["start"], clip["end"], clip_path)
            print(f"Exported {frames_written} frames to: {clip_path}")

            # Analyze clip with hybrid system
            analysis = analyze_clip_with_hybrid_system(
                analyzer, video_path, clip["start"], clip["end"], clip["name"]
            )
            analysis["exported_file"] = str(clip_path)
            analysis["frames_exported"] = frames_written
            analysis["expected_situation"] = clip["expected"]

            all_analyses.append(analysis)

        except Exception as e:
            print(f"Error processing clip {i}: {e}")
            import traceback

            traceback.print_exc()

    # Generate final report
    print(f"\n{'='*80}")
    print("EXPORT AND ANALYSIS COMPLETE")
    print(f"{'='*80}")

    print(f"\nClips exported to: {output_dir.absolute()}")
    print(f"Total clips: {len(all_analyses)}")

    # Summary statistics
    total_possession = sum(a["summary"]["possession_detections"] for a in all_analyses)
    total_territory = sum(a["summary"]["territory_detections"] for a in all_analyses)
    total_down_distance = sum(a["summary"]["down_distance_detections"] for a in all_analyses)
    total_corrections = sum(a["summary"]["hybrid_corrections"] for a in all_analyses)
    total_frames = sum(a["summary"]["total_frames_analyzed"] for a in all_analyses)

    print(f"\nOverall Detection Statistics:")
    print(f"  Total frames analyzed: {total_frames}")
    print(
        f"  Possession detections: {total_possession}/{total_frames} ({total_possession/total_frames*100:.1f}%)"
    )
    print(
        f"  Territory detections: {total_territory}/{total_frames} ({total_territory/total_frames*100:.1f}%)"
    )
    print(
        f"  Down/Distance detections: {total_down_distance}/{total_frames} ({total_down_distance/total_frames*100:.1f}%)"
    )
    print(f"  Hybrid corrections applied: {total_corrections}")

    print(f"\nHybrid OCR+Logic System Performance:")
    if total_down_distance > 0:
        correction_rate = total_corrections / total_down_distance * 100
        print(f"  Correction rate: {correction_rate:.1f}% of successful detections")
        print(f"  System is actively using game logic to validate/correct OCR results!")
    else:
        print(f"  No down/distance detections to analyze")

    print(f"\nFiles exported:")
    for analysis in all_analyses:
        print(f"  {analysis['exported_file']} ({analysis['frames_exported']} frames)")


if __name__ == "__main__":
    main()

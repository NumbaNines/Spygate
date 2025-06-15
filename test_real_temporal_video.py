#!/usr/bin/env python3
"""
REAL Temporal Confidence Voting Test with Actual Video
======================================================
Uses the actual 8-class YOLO model and OCR on real frames from
"1 min 30 test clip.mp4" to demonstrate temporal confidence voting.
"""

import random
import time
from pathlib import Path

import cv2
import numpy as np

from src.spygate.ml.enhanced_ocr import EnhancedOCR

# Import SpygateAI components
from src.spygate.ml.temporal_extraction_manager import ExtractionResult, TemporalExtractionManager
from src.spygate.ml.yolov8_model import EnhancedYOLOv8


def test_real_temporal_video_analysis():
    """Test temporal confidence voting with REAL video analysis."""
    print("🏈 REAL Temporal Confidence Voting Test")
    print("=" * 60)

    video_path = "1 min 30 test clip.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize components with YOUR trained 8-class model
    print("🔧 Initializing AI Components...")
    model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        print("   Using default model for demonstration...")
        model_path = "models/yolov8n.pt"
    else:
        print(f"✅ Using trained 8-class model: {model_path}")

    yolo_model = EnhancedYOLOv8(model_path=model_path)
    ocr_engine = EnhancedOCR()
    temporal_manager = TemporalExtractionManager()

    # Adjust voting windows for our test scenario (random frames across 90+ seconds)
    temporal_manager.voting_windows = {
        "game_clock": 60.0,  # 60 second window for test
        "play_clock": 60.0,  # 60 second window for test
        "down_distance": 60.0,  # 60 second window for test
        "scores": 60.0,  # 60 second window for test
        "team_names": 60.0,  # 60 second window for test
    }

    # Reduce minimum votes for test
    temporal_manager.min_votes = {
        "game_clock": 1,  # Need 1+ vote for test
        "play_clock": 2,  # Need 2+ votes for test
        "down_distance": 1,  # Need 1+ vote for test
        "scores": 1,  # Need 1+ vote for test
        "team_names": 1,  # Need 1+ vote for test
    }

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    print(f"📹 Video: {duration:.1f}s, {fps:.0f} FPS, {frame_count} frames")

    # Generate 15 random frames throughout the video
    random.seed(42)
    random_frames = sorted(random.sample(range(0, frame_count), min(15, frame_count)))

    print(f"🎲 Testing {len(random_frames)} random frames:")
    print(f"   Frames: {random_frames}")
    print(f"   Times: {[f'{f/fps:.1f}s' for f in random_frames]}")

    print("\n🔍 Processing Frames...")
    print("-" * 60)

    total_ocr_calls = 0
    successful_detections = 0

    for i, frame_num in enumerate(random_frames):
        print(f"\n📍 Frame {frame_num} (t={frame_num/fps:.1f}s) - {i+1}/{len(random_frames)}")

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print("   ❌ Could not read frame")
            continue

        # Run YOLO detection
        start_time = time.time()
        detections = yolo_model.detect(frame)
        detection_time = time.time() - start_time

        print(f"   🎯 YOLO: {len(detections)} detections in {detection_time:.3f}s")

        # Process each detected region
        frame_ocr_calls = 0
        frame_time = frame_num / fps

        for detection in detections:
            class_name = detection.get("class", "unknown")
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox", [0, 0, 0, 0])

            # Extract region
            x1, y1, x2, y2 = map(int, bbox)
            region = frame[y1:y2, x1:x2]

            if region.size == 0:
                continue

            print(f"      📦 {class_name}: {confidence:.2f} conf, region {x2-x1}x{y2-y1}")

            # Determine extraction strategy based on class
            should_extract = False
            extraction_type = None

            if class_name == "play_clock_area":
                # Extract every frame (dynamic)
                should_extract = True
                extraction_type = "play_clock"
            elif class_name == "game_clock_area":
                # Extract every frame (dynamic)
                should_extract = True
                extraction_type = "game_clock"
            elif class_name == "down_distance_area":
                # Extract every 3 seconds (semi-static)
                should_extract = temporal_manager.should_extract("down_distance", frame_time)
                extraction_type = "down_distance"
            elif class_name == "possession_triangle_area":
                # Extract every 10 seconds (mostly static)
                should_extract = temporal_manager.should_extract("scores", frame_time)
                extraction_type = "scores"

            if should_extract and extraction_type:
                # Run OCR based on region type
                try:
                    ocr_start = time.time()
                    raw_text = ""
                    ocr_confidence = 0.0

                    # Use specialized extraction methods based on class
                    if class_name == "game_clock_area":
                        extracted_text = ocr_engine.extract_game_clock(region)
                        if extracted_text:
                            raw_text = extracted_text
                            ocr_confidence = 0.8  # Assume good confidence if extracted

                    elif class_name == "play_clock_area":
                        extracted_text = ocr_engine.extract_play_clock(region)
                        if extracted_text:
                            raw_text = extracted_text
                            ocr_confidence = 0.8

                    elif class_name == "down_distance_area":
                        extracted_text = ocr_engine.extract_down_distance(region)
                        if extracted_text:
                            raw_text = extracted_text
                            ocr_confidence = 0.8

                    elif class_name == "possession_triangle_area":
                        # Extract scores from possession area
                        score_data = ocr_engine.extract_scores(region)
                        if score_data:
                            raw_text = f"{score_data['away_team']} {score_data['away_score']} {score_data['home_team']} {score_data['home_score']}"
                            ocr_confidence = 0.8

                        # ALSO detect triangle orientation (keep existing functionality)
                        # TODO: Add triangle detection here if needed

                    elif class_name in [
                        "territory_triangle_area",
                        "hud",
                        "preplay_indicator",
                        "play_call_screen",
                    ]:
                        # For these regions, we might not need OCR or use different methods
                        # Keep triangle detection for territory_triangle_area
                        # Skip OCR for now but preserve the region for other analysis
                        raw_text = f"[{class_name}_detected]"
                        ocr_confidence = 0.5

                    else:
                        # Fallback to original method for unknown regions
                        ocr_result = ocr_engine.process_region(region)
                        if ocr_result:
                            for key in [
                                "text",
                                "down",
                                "distance",
                                "score_home",
                                "score_away",
                                "time",
                            ]:
                                if key in ocr_result and ocr_result[key]:
                                    raw_text = str(ocr_result[key])
                                    ocr_confidence = ocr_result.get("confidence", 0.0)
                                    break

                    ocr_time = time.time() - ocr_start

                    frame_ocr_calls += 1
                    total_ocr_calls += 1

                    if raw_text.strip():
                        print(
                            f'         🔤 OCR: "{raw_text}" ({ocr_confidence:.2f} conf, {ocr_time:.3f}s)'
                        )

                        # Create extraction result for temporal voting
                        extraction_result = ExtractionResult(
                            value=raw_text,
                            confidence=ocr_confidence,
                            timestamp=frame_time,
                            raw_text=raw_text,
                            method=f"{class_name}_extraction",
                        )

                        # Add to temporal manager
                        temporal_manager.add_extraction_result(extraction_type, extraction_result)
                        print(f"         ✅ Parsed: {raw_text}")
                    else:
                        print(f"         ❌ No OCR text extracted")

                except Exception as e:
                    print(f"         ❌ OCR error: {e}")
            else:
                print(f"         ⏭️  Skipped extraction (temporal optimization)")

        print(f"   📊 Frame summary: {frame_ocr_calls} OCR calls")

    cap.release()

    # Get final results from temporal manager
    print(f"\n🎯 TEMPORAL CONFIDENCE VOTING RESULTS")
    print("=" * 60)

    categories = ["play_clock", "game_clock", "down_distance", "scores"]

    print("\n🔍 DEBUG: Temporal Manager State")
    print("-" * 50)

    # Show performance stats
    stats = temporal_manager.get_performance_stats()
    print(f"📊 Performance Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()

    # Show raw voting data
    print("📊 Raw Voting Data:")
    for element_type in categories:
        voting_data = temporal_manager.voting_data.get(element_type, {})
        min_votes = temporal_manager.min_votes.get(element_type, 1)
        if voting_data:
            print(f"   {element_type} (min votes: {min_votes}):")
            for value, results in voting_data.items():
                avg_conf = sum(r.confidence for r in results) / len(results)
                print(f"      '{value}': {len(results)} votes, avg conf: {avg_conf:.2f}")
                for r in results:
                    print(f"         t={r.timestamp:.1f}s, conf={r.confidence:.2f}")
        else:
            print(f"   {element_type}: No voting data")
    print()

    for category in categories:
        result = temporal_manager.get_current_value(category)
        if result and result.get("value"):
            print(f"🏆 {category.upper()}: \"{result['value']}\"")
            print(f"   Confidence: {result.get('confidence', 0)*100:.1f}%")
            print(f"   Votes: {result.get('votes', 0)}")
            print(f"   Stability: {result.get('stability_score', 0):.2f}")
        else:
            print(f"❌ {category.upper()}: No reliable result")
        print()

    # Performance summary
    traditional_ocr_calls = len(random_frames) * 4  # 4 regions per frame
    efficiency = (traditional_ocr_calls - total_ocr_calls) / traditional_ocr_calls * 100

    print(f"⚡ PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"Traditional approach: {traditional_ocr_calls} OCR calls")
    print(f"Temporal approach: {total_ocr_calls} OCR calls")
    print(f"Efficiency gain: {efficiency:.1f}% reduction")
    print(f"Successful extractions: {successful_detections}")
    print(
        f"Success rate: {successful_detections/total_ocr_calls*100:.1f}%"
        if total_ocr_calls > 0
        else "N/A"
    )


def parse_play_clock(text: str) -> str:
    """Parse play clock text (0-40 seconds)."""
    import re

    # Look for numbers 0-40
    match = re.search(r"\b([0-3]?[0-9]|40)\b", text)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 40:
            return str(num)
    return None


def parse_game_clock(text: str) -> str:
    """Parse game clock text (MM:SS format)."""
    import re

    # Look for MM:SS format
    match = re.search(r"\b(\d{1,2}):([0-5]\d)\b", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        if 0 <= minutes <= 15 and 0 <= seconds <= 59:
            return f"{minutes}:{seconds:02d}"
    return None


def parse_down_distance(text: str) -> str:
    """Parse down and distance text."""
    import re

    # Look for patterns like "1st & 10", "3rd & 7", "1st & Goal"
    patterns = [r"\b([1-4])(st|nd|rd|th)\s*&\s*(\d+)\b", r"\b([1-4])(st|nd|rd|th)\s*&\s*(Goal)\b"]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            down = match.group(1)
            suffix = match.group(2).lower()
            distance = match.group(3)

            # Correct suffix
            if down == "1":
                suffix = "st"
            elif down == "2":
                suffix = "nd"
            elif down == "3":
                suffix = "rd"
            else:
                suffix = "th"

            return f"{down}{suffix} & {distance}"

    return None


def parse_scores(text: str) -> dict:
    """Parse team scores from possession area."""
    import re

    # Look for team abbreviations and scores
    # Pattern: TEAM1 ## TEAM2 ##
    match = re.search(r"\b([A-Z]{2,4})\s+(\d{1,2})\s+([A-Z]{2,4})\s+(\d{1,2})\b", text)
    if match:
        return {
            "away_team": match.group(1),
            "away_score": int(match.group(2)),
            "home_team": match.group(3),
            "home_score": int(match.group(4)),
        }
    return None


if __name__ == "__main__":
    test_real_temporal_video_analysis()

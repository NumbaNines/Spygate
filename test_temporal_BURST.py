#!/usr/bin/env python3
"""
BURST: Temporal Confidence Voting with Burst Sampling
====================================================
Smart burst sampling: 3 frames every 10 seconds with consensus voting.
Target: 90-second video in 10-15 seconds with high reliability.
"""

import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.append("src")

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


class BurstSamplingAnalyzer:
    """Burst sampling analyzer with 3-frame consensus voting."""

    def __init__(self):
        """Initialize with burst sampling optimizations."""
        print("💥 Initializing BURST sampling analyzer...")

        # Initialize base analyzer
        self.analyzer = EnhancedGameAnalyzer()

        # SPEED-OPTIMIZED Burst sampling configuration
        self.decision_interval = 10.0  # Make decision every 10 seconds
        self.burst_frames = 2  # Reduced from 3 to 2 for speed
        self.burst_window = 1.5  # Reduced from 2.0 to 1.5 for speed

        # Target classes (no generic "hud")
        self.target_classes = [
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ]

        # BALANCED resolution for speed + OCR quality
        self.target_resolution = 420  # Sweet spot: faster than 480, better than 360

        # Decision tracking
        self.last_decisions = {}
        self.decision_history = []

        print(f"✅ Burst config: {self.burst_frames} frames every {self.decision_interval}s")
        print(f"✅ Burst window: {self.burst_window}s before each decision")
        print(f"✅ Using {len(self.target_classes)} specific classes (no generic hud)")
        print("✅ BURST optimizations initialized")

    def calculate_burst_frames(self, video_duration: float, fps: float) -> list:
        """Calculate which frames to analyze using burst sampling."""
        burst_frames = []

        # Calculate decision points (every 10 seconds)
        decision_times = []
        current_time = self.decision_interval
        while current_time <= video_duration:
            decision_times.append(current_time)
            current_time += self.decision_interval

        print(f"📊 Decision points: {len(decision_times)} (every {self.decision_interval}s)")

        # For each decision point, calculate burst frames
        for decision_time in decision_times:
            # Calculate 2 frames in the 1.5-second window before decision (SPEED OPTIMIZED)
            frame_times = [
                decision_time - 1.0,  # 1 second before
                decision_time - 0.0,  # At decision point
            ]

            # Convert times to frame numbers
            for frame_time in frame_times:
                if frame_time >= 0:  # Don't go negative
                    frame_num = int(frame_time * fps)
                    burst_frames.append(
                        {
                            "frame_num": frame_num,
                            "time": frame_time,
                            "decision_point": decision_time,
                        }
                    )

        # Sort by frame number
        burst_frames.sort(key=lambda x: x["frame_num"])

        print(f"📊 Total burst frames: {len(burst_frames)}")
        return burst_frames

    def preprocess_frame_fast(self, frame: np.ndarray) -> np.ndarray:
        """Fast frame preprocessing for burst analysis."""
        height, width = frame.shape[:2]

        # Aggressive resizing for speed
        if width > self.target_resolution:
            scale = self.target_resolution / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return frame

    def analyze_frame_burst(
        self, frame: np.ndarray, current_time: float, frame_idx: int = 0
    ) -> dict:
        """Analyze single frame for burst sampling."""

        # FIXED: Keep original frame for YOLO detection, only resize for performance after
        original_frame = frame.copy()

        # Analyze with base analyzer using ORIGINAL FRAME - pass None as current_time to disable temporal manager optimization
        # since burst sampling provides its own consensus mechanism
        game_state = self.analyzer.analyze_frame(original_frame, None)

        # IMPORTANT: The analyzer.analyze_frame() calls _process_detections() which still uses the current_time
        # We need to ensure that _process_detections also receives None to fully bypass temporal manager

        # DEBUG: Check if model is actually detecting anything
        if frame_idx <= 3:  # Only for first few frames to avoid spam
            # Try to get raw OCR results directly from the extraction methods
            yolo_model = self.analyzer.model
            if hasattr(yolo_model, "model"):
                # Run inference directly to see what's detected using ORIGINAL FRAME
                results = yolo_model.model(original_frame)
                if hasattr(results[0], "boxes") and results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                    print(f"🔍 YOLO detected {num_detections} objects")
                    if num_detections > 0:
                        for i, box in enumerate(results[0].boxes):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            bbox = box.xyxy[0].tolist()
                            # Map class_id to class name using UI_CLASSES from enhanced_game_analyzer
                            UI_CLASSES = [
                                "hud",  # 0 - Main HUD bar
                                "possession_triangle_area",  # 1 - Left triangle (possession)
                                "territory_triangle_area",  # 2 - Right triangle (territory)
                                "preplay_indicator",  # 3 - Pre-play state
                                "play_call_screen",  # 4 - Play selection overlay
                                "down_distance_area",  # 5 - Down & distance text
                                "game_clock_area",  # 6 - Game clock display
                                "play_clock_area",  # 7 - Play clock display
                            ]
                            class_name = (
                                UI_CLASSES[class_id]
                                if class_id < len(UI_CLASSES)
                                else f"class_{class_id}"
                            )
                            print(f"   Detection {i}: {class_name} (conf: {confidence:.3f})")

                            # For down_distance_area, let's debug the OCR extraction
                            if (
                                class_name == "down_distance_area"
                            ):  # Test any down_distance_area detection
                                x1, y1, x2, y2 = map(int, bbox)
                                region_roi = frame[y1:y2, x1:x2]

                                # FIXED: Scale up tiny regions for better OCR
                                if region_roi.shape[0] < 20 or region_roi.shape[1] < 60:
                                    # Scale up small regions by 3x for OCR
                                    scale_factor = 3
                                    new_height = region_roi.shape[0] * scale_factor
                                    new_width = region_roi.shape[1] * scale_factor
                                    region_roi = cv2.resize(
                                        region_roi,
                                        (new_width, new_height),
                                        interpolation=cv2.INTER_CUBIC,
                                    )
                                    print(
                                        f"   🔧 Scaled region from {frame[y1:y2, x1:x2].shape} to {region_roi.shape}"
                                    )

                                # DEBUG: Save the region image to see what we're actually processing
                                if frame_idx <= 2:  # Only save first few for debugging
                                    debug_filename = f"debug_region_f{frame_idx}_d{i}.png"
                                    try:
                                        cv2.imwrite(debug_filename, region_roi)
                                        print(f"   💾 Saved region to {debug_filename}")
                                    except Exception as e:
                                        print(f"   ❌ Failed to save region: {e}")

                                region_data = {
                                    "roi": region_roi,
                                    "bbox": bbox,
                                    "confidence": confidence,
                                }

                                print(f"   🔍 Testing OCR on {class_name} region:")
                                print(f"      Region shape: {region_roi.shape}")
                                print(f"      Bbox: {bbox}")

                                # Test the extraction method directly
                                try:
                                    # Note: Burst sampling bypasses temporal manager by passing None as current_time
                                    # This ensures OCR extraction always runs for burst sampling consensus

                                    ocr_result = self.analyzer._extract_down_distance_from_region(
                                        region_data, None
                                    )
                                    if ocr_result:
                                        print(
                                            f"      ✅ OCR success: {ocr_result.get('down')} & {ocr_result.get('distance')}"
                                        )
                                    else:
                                        print(f"      ❌ OCR failed")

                                    # DETAILED DEBUGGING: Test each step
                                    processed_roi = self.analyzer._preprocess_region_for_ocr(
                                        region_roi
                                    )
                                    import pytesseract

                                    raw_text = pytesseract.image_to_string(
                                        processed_roi, config=r"--oem 3 --psm 7"
                                    ).strip()
                                    if raw_text:
                                        corrected = self.analyzer._apply_down_distance_corrections(
                                            raw_text
                                        )
                                        print(f"      Raw: '{raw_text}' → Corrected: '{corrected}'")

                                        # Test validation
                                        is_valid = self.analyzer._validate_down_distance(corrected)
                                        print(f"      Validation: {is_valid}")

                                        # Test parsing
                                        parsed = self.analyzer._parse_down_distance_text(corrected)
                                        print(f"      Parsed: {parsed}")

                                        # Test robust extraction directly with detailed debugging
                                        print(f"      === ROBUST DEBUGGING ===")

                                        # Test confidence estimation
                                        conf_est = self.analyzer._estimate_text_confidence(
                                            corrected
                                        )
                                        print(f"      Confidence estimate: {conf_est}")

                                        # Test quality score
                                        quality = self.analyzer._calculate_text_quality_score(
                                            corrected
                                        )
                                        print(f"      Quality score: {quality}")

                                        # Test combined score
                                        combined = conf_est * quality
                                        print(f"      Combined score: {combined}")

                                        robust_result = self.analyzer._extract_down_distance_robust(
                                            region_roi
                                        )
                                        print(f"      Robust result: '{robust_result}'")

                                except Exception as e:
                                    print(f"      OCR error: {e}")
                                    import traceback

                                    print(f"      Traceback: {traceback.format_exc()}")

                                break  # Only test one down_distance_area per frame
                else:
                    print(f"🔍 YOLO detected no objects")
            else:
                print(f"🔍 YOLO model not available")

        # DEBUG: Print what we actually got from the analyzer
        if frame_idx <= 3:  # Only for first few frames to avoid spam
            print(f"🔍 DEBUG Frame analysis:")
            print(f"   Game state type: {type(game_state)}")
            print(f"   Time: {getattr(game_state, 'time', 'None')}")
            print(f"   Down: {getattr(game_state, 'down', 'None')}")
            print(f"   Distance: {getattr(game_state, 'distance', 'None')}")
            print(f"   Score Away: {getattr(game_state, 'score_away', 'None')}")
            print(f"   Score Home: {getattr(game_state, 'score_home', 'None')}")
            print(f"   Away Team: {getattr(game_state, 'away_team', 'None')}")
            print(f"   Home Team: {getattr(game_state, 'home_team', 'None')}")

        # Extract values directly from game_state instead of temporal manager
        # since burst sampling provides its own consensus mechanism
        extracted_values = {
            "game_clock": getattr(game_state, "time", None),
            "play_clock": None,  # Not directly stored in game_state, try temporal manager
            "down_distance": (
                f"{getattr(game_state, 'down', None)} & {getattr(game_state, 'distance', None)}"
                if getattr(game_state, "down", None) and getattr(game_state, "distance", None)
                else None
            ),
            "scores": (
                f"{getattr(game_state, 'away_team', 'UNK')} {getattr(game_state, 'score_away', 0)} - {getattr(game_state, 'home_team', 'UNK')} {getattr(game_state, 'score_home', 0)}"
                if getattr(game_state, "score_away", None) is not None
                else None
            ),
        }

        # Try to get play clock from temporal manager as fallback
        temporal_mgr = self.analyzer.temporal_manager
        play_clock_data = temporal_mgr.get_current_value("play_clock")
        if play_clock_data:
            extracted_values["play_clock"] = play_clock_data.get("value")

        # Clean up None values and format properly
        clean_values = {}
        for key, value in extracted_values.items():
            if value and value != "None & None" and "None" not in str(value):
                clean_values[key] = value
            else:
                clean_values[key] = "Unknown"

        return {"results": game_state, "values": clean_values, "time": current_time}

    def vote_on_burst_results(self, burst_results: list) -> dict:
        """Perform consensus voting on burst results."""
        if not burst_results:
            return {}

        # Collect all values for each element type
        all_values = {"game_clock": [], "play_clock": [], "down_distance": [], "scores": []}

        for result in burst_results:
            values = result["values"]
            for key in all_values.keys():
                if values.get(key) and values[key] != "Unknown":
                    all_values[key].append(values[key])

        # Vote on each element type
        final_decisions = {}
        voting_details = {}

        for element_type, value_list in all_values.items():
            if value_list:
                # Count occurrences
                vote_counts = Counter(value_list)
                # Get most common value
                winner, count = vote_counts.most_common(1)[0]
                confidence = count / len(burst_results)

                final_decisions[element_type] = winner
                voting_details[element_type] = {
                    "winner": winner,
                    "votes": count,
                    "total": len(burst_results),
                    "confidence": confidence,
                    "all_votes": dict(vote_counts),
                }
            else:
                final_decisions[element_type] = "Unknown"
                voting_details[element_type] = {
                    "winner": "Unknown",
                    "votes": 0,
                    "total": len(burst_results),
                    "confidence": 0.0,
                    "all_votes": {},
                }

        return {
            "decisions": final_decisions,
            "voting_details": voting_details,
            "burst_size": len(burst_results),
        }


def test_temporal_BURST():
    """Test enhanced temporal analysis with burst sampling and consensus voting."""
    print("💥 BURST: Temporal Confidence Voting with Consensus")
    print("=" * 80)

    # Quick debug mode for faster testing
    QUICK_DEBUG = False  # Set to True for faster debugging - Disabled to test all decision points for different downs

    if QUICK_DEBUG:
        print("🔧 QUICK DEBUG MODE: Processing only first decision point")

    # Display strategy
    print("🚀 BURST SAMPLING STRATEGY:")
    print("   ✅ 3 frames every 10 seconds")
    print("   ✅ Consensus voting for reliability")
    print("   ✅ 2-second burst window")
    print("   ✅ Ultra-low resolution (480p)")
    print('   ✅ No generic "hud" class')
    print("   ✅ Smart temporal prediction")

    # Initialize burst analyzer
    burst_analyzer = BurstSamplingAnalyzer()

    # Load test video
    video_path = "1 min 30 test clip.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"📹 Video: {video_path} ({duration:.1f}s, {total_frames} frames)")

    # Calculate burst frames
    burst_frames = burst_analyzer.calculate_burst_frames(duration, fps)

    print(f"💥 BURST MODE: Processing {len(burst_frames)} frames total")
    print(f"🎯 Target: 10-15 seconds processing time")

    # Create output directory
    output_dir = Path("temporal_test_results_burst")
    output_dir.mkdir(exist_ok=True)

    print("\n💥 BURST ANALYSIS: Smart sampling with consensus voting...")
    print("=" * 80)

    # Track statistics
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Store burst results
    current_burst = []
    current_decision_point = None
    all_decisions = []
    frame_data = []

    # Process burst frames
    for burst_info in burst_frames:
        target_frame = burst_info["frame_num"]
        frame_time = burst_info["time"]
        decision_point = burst_info["decision_point"]

        # Seek to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret:
            print(f"❌ Could not read frame {target_frame}")
            continue

        processed_count += 1

        try:
            # Quick debug mode: only process first decision point
            if (
                QUICK_DEBUG
                and current_decision_point is not None
                and decision_point > current_decision_point
            ):
                print(
                    f"🔧 QUICK DEBUG: Stopping after first decision point ({current_decision_point:.1f}s)"
                )
                break

            # Analyze frame
            analysis_result = burst_analyzer.analyze_frame_burst(frame, frame_time, processed_count)

            # Group by decision point
            if current_decision_point != decision_point:
                # Process previous burst if exists
                if current_burst and current_decision_point is not None:
                    voting_result = burst_analyzer.vote_on_burst_results(current_burst)
                    all_decisions.append(
                        {
                            "decision_point": current_decision_point,
                            "voting_result": voting_result,
                            "burst_frames": len(current_burst),
                        }
                    )

                    # Show decision
                    decisions = voting_result["decisions"]
                    known_count = len([v for v in decisions.values() if v != "Unknown"])
                    print(
                        f"🎯 Decision at {current_decision_point:.1f}s: {known_count}/4 values decided"
                    )

                # Start new burst
                current_burst = []
                current_decision_point = decision_point

            # Add to current burst
            current_burst.append(analysis_result)

            # Store frame data for screenshots
            if len(frame_data) < 15:  # Limit for speed
                frame_data.append(
                    {
                        "frame_num": target_frame,
                        "time": frame_time,
                        "frame": frame.copy(),
                        "decision_point": decision_point,
                        "analysis": analysis_result,
                    }
                )

            # Progress update
            if processed_count % 5 == 0:
                elapsed = time.time() - start_time
                progress = (processed_count / len(burst_frames)) * 100
                fps_actual = processed_count / elapsed if elapsed > 0 else 0
                eta = (len(burst_frames) - processed_count) / fps_actual if fps_actual > 0 else 0

                print(
                    f"💥 {processed_count}/{len(burst_frames)} frames ({progress:.1f}%) - "
                    f"{fps_actual:.1f} FPS - ETA: {eta:.0f}s"
                )

        except Exception as e:
            print(f"❌ Error frame {target_frame}: {e}")
            continue

    # Process final burst
    if current_burst and current_decision_point is not None:
        voting_result = burst_analyzer.vote_on_burst_results(current_burst)
        all_decisions.append(
            {
                "decision_point": current_decision_point,
                "voting_result": voting_result,
                "burst_frames": len(current_burst),
            }
        )

    cap.release()

    # Calculate final statistics
    processing_time = time.time() - start_time
    actual_fps = processed_count / processing_time if processing_time > 0 else 0
    speedup_factor = duration / processing_time if processing_time > 0 else 0

    print(f"\n💥 BURST ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📊 Video: {duration:.1f}s ({total_frames} total frames)")
    print(f"📊 Processed: {processed_count} frames ({len(all_decisions)} decisions)")
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"🚀 Processing speed: {actual_fps:.1f} FPS")
    print(f"⚡ Real-time speedup: {speedup_factor:.1f}x")

    # Performance assessment
    if processing_time < 15:
        print(f"💥 BURST SUCCESS: {processing_time:.1f}s (Target: 10-15s)")
    elif processing_time < 30:
        print(f"✅ EXCELLENT: {processing_time:.1f}s (Under 30s)")
    else:
        print(f"⚠️  NEEDS MORE: {processing_time:.1f}s (Still too slow)")

    # Show decision results
    print(f"\n🎯 CONSENSUS VOTING RESULTS:")
    print("=" * 50)

    total_decisions = 0
    successful_decisions = 0

    for decision in all_decisions:
        decision_point = decision["decision_point"]
        voting_result = decision["voting_result"]
        decisions = voting_result["decisions"]
        voting_details = voting_result["voting_details"]

        known_count = len([v for v in decisions.values() if v != "Unknown"])
        total_decisions += 4  # 4 possible values
        successful_decisions += known_count

        print(
            f'⏰ {decision_point:.1f}s: {known_count}/4 values ({voting_result["burst_size"]} frames)'
        )

        # Show voting details for known values
        for element_type, details in voting_details.items():
            if details["winner"] != "Unknown":
                confidence = details["confidence"] * 100
                votes = details["votes"]
                total = details["total"]
                print(
                    f'   ✅ {element_type}: {details["winner"]} ({votes}/{total} votes, {confidence:.0f}%)'
                )

    decision_success_rate = (
        (successful_decisions / total_decisions) * 100 if total_decisions > 0 else 0
    )

    print(f"\n📊 DECISION QUALITY:")
    print(
        f"   🎯 Success rate: {decision_success_rate:.1f}% ({successful_decisions}/{total_decisions})"
    )
    print(f"   💥 Decisions made: {len(all_decisions)}")
    if len(all_decisions) > 0:
        print(f"   🔄 Avg frames per decision: {processed_count / len(all_decisions):.1f}")
    else:
        print(f"   🔄 Avg frames per decision: N/A (no successful decisions)")

    # Generate screenshots
    print(f"\n📸 GENERATING SCREENSHOTS...")
    print("=" * 50)

    num_screenshots = min(8, len(frame_data))
    if len(frame_data) <= num_screenshots:
        selected_frames = frame_data
    else:
        selected_frames = random.sample(frame_data, num_screenshots)
        selected_frames.sort(key=lambda x: x["frame_num"])

    for i, frame_info in enumerate(selected_frames, 1):
        frame_num = frame_info["frame_num"]
        frame_time = frame_info["time"]
        frame = frame_info["frame"]
        decision_point = frame_info["decision_point"]
        analysis = frame_info["analysis"]

        # Create annotated screenshot
        annotated_frame = frame.copy()

        # Add overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (475, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        # Add text annotations
        y_pos = 20
        cv2.putText(
            annotated_frame,
            f"BURST - Frame #{frame_num} at {frame_time:.1f}s",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y_pos += 20

        cv2.putText(
            annotated_frame,
            f"Decision point: {decision_point:.1f}s | {processing_time:.1f}s total",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
        y_pos += 15

        # Add current values
        values = analysis["values"]
        known_vals = len([v for v in values.values() if v != "Unknown"])
        cv2.putText(
            annotated_frame,
            f"Current values: {known_vals}/4",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 0),
            1,
        )
        y_pos += 15

        # Add values
        for key, value in values.items():
            if value != "Unknown":
                color = (0, 255, 0)
                cv2.putText(
                    annotated_frame,
                    f"{key}: {value}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
                y_pos += 12

        # Save screenshot
        screenshot_path = output_dir / f"burst_{frame_num:04d}_{frame_time:.1f}s.png"
        cv2.imwrite(str(screenshot_path), annotated_frame)

        if i <= 3:
            print(
                f"📋 Frame {i}: #{frame_num} ({frame_time:.1f}s) - Decision: {decision_point:.1f}s"
            )

    print(f"💾 {len(selected_frames)} screenshots saved to {output_dir}")

    print(f"\n💥 BURST TEST COMPLETE!")
    print("=" * 80)
    print(f"✅ {duration:.1f}s video → {processing_time:.1f}s analysis")
    print(f"✅ {speedup_factor:.1f}x real-time processing")
    print(f"✅ {decision_success_rate:.1f}% decision success rate")
    print(f"✅ {len(all_decisions)} consensus decisions made")
    print(f"✅ {processed_count} total frames processed")

    # Final assessment
    print(f"\n📊 BURST SAMPLING RESULTS:")
    if processing_time < 15:
        print(f"  💥 TARGET ACHIEVED: {processing_time:.1f}s (Goal: 10-15s)")
        print(f"  🚀 Ready for production deployment")
    elif processing_time < 30:
        print(f"  ✅ EXCELLENT: {processing_time:.1f}s (Under 30s)")
        print(f"  ✅ Suitable for real-world use")
    else:
        print(f"  ⚠️  MORE OPTIMIZATION NEEDED: {processing_time:.1f}s")

    print(f"  🎯 Decision quality: {decision_success_rate:.1f}% success")
    print(f"  💥 Consensus voting: {processed_count} frames → {len(all_decisions)} decisions")
    print(f"  ⚡ Efficiency: {processed_count / total_frames * 100:.3f}% of total frames")


if __name__ == "__main__":
    test_temporal_BURST()

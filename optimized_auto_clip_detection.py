#!/usr/bin/env python3
"""
Optimized Auto-Clip Detection Workflow
=====================================
High-performance auto-clip detection with hardware-adaptive frame skipping, scene change detection, and faster processing strategies
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Set up proper Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class OptimizedAutoClipDetector:
    """
    High-performance auto-clip detection with multiple optimization strategies.
    """

    def __init__(self, hardware_tier: str = "high"):
        """Initialize optimized detector with hardware-adaptive settings."""
        self.hardware_tier = hardware_tier.lower()
        self.detected_clips = []
        self.frame_cache = {}
        self.scene_change_threshold = 0.3
        self.logger = logging.getLogger(__name__)
        self.setup_optimization_settings()

    def setup_optimization_settings(self):
        """Configure optimization settings based on hardware tier."""
        optimization_configs = {
            "low": {
                "frame_skip": 90,  # Skip 90 frames (3 seconds at 30fps)
                "scene_check_interval": 30,  # Check scene changes every 30 frames
                "analysis_resolution": (640, 360),  # Lower resolution for faster processing
                "confidence_threshold": 0.8,  # Higher threshold to reduce false positives
                "max_clips_per_minute": 2,  # Limit clips to reduce processing
            },
            "medium": {
                "frame_skip": 60,  # Skip 60 frames (2 seconds at 30fps)
                "scene_check_interval": 20,
                "analysis_resolution": (854, 480),
                "confidence_threshold": 0.7,
                "max_clips_per_minute": 3,
            },
            "high": {
                "frame_skip": 30,  # Skip 30 frames (1 second at 30fps)
                "scene_check_interval": 15,
                "analysis_resolution": (1280, 720),
                "confidence_threshold": 0.6,
                "max_clips_per_minute": 5,
            },
            "ultra": {
                "frame_skip": 15,  # Skip 15 frames (0.5 seconds at 30fps)
                "scene_check_interval": 10,
                "analysis_resolution": (1920, 1080),
                "confidence_threshold": 0.5,
                "max_clips_per_minute": 8,
            },
        }

        self.config = optimization_configs.get(self.hardware_tier, optimization_configs["medium"])
        self.logger.info(f"Configured for {self.hardware_tier} hardware tier: {self.config}")

    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Fast scene change detection using histogram comparison.

        Args:
            frame1: Previous frame
            frame2: Current frame

        Returns:
            bool: True if significant scene change detected
        """
        if frame1 is None or frame2 is None:
            return True

        # Resize frames for faster processing
        h, w = frame1.shape[:2]
        small_size = (w // 4, h // 4)  # Quarter resolution for speed

        small1 = cv2.resize(frame1, small_size)
        small2 = cv2.resize(frame2, small_size)

        # Convert to grayscale for faster histogram calculation
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [32], [0, 256])  # Reduced bins for speed
        hist2 = cv2.calcHist([gray2], [0], None, [32], [0, 256])

        # Compare histograms using correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Scene change if correlation is below threshold
        return correlation < (1.0 - self.scene_change_threshold)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for faster analysis.

        Args:
            frame: Input frame

        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize to analysis resolution for speed
        target_resolution = self.config["analysis_resolution"]
        return cv2.resize(frame, target_resolution)

    def simulate_hud_detection(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Simulate HUD detection for demo purposes.

        Args:
            frame: Video frame

        Returns:
            Dict: Simulated HUD information
        """
        # In real implementation, this would use YOLOv8
        # For now, simulate based on frame characteristics
        frame_brightness = np.mean(frame)
        frame_variance = np.var(frame)

        # Simulate game state based on frame characteristics
        hud_info = {
            "confidence": 0.85,
            "down": np.random.choice([1, 2, 3, 4]),
            "distance": np.random.choice([1, 3, 5, 7, 10, 15]),
            "game_clock": f"{np.random.randint(0, 15)}:{np.random.randint(0, 60):02d}",
            "score_home": np.random.randint(0, 35),
            "score_away": np.random.randint(0, 35),
            "frame_brightness": frame_brightness,
            "frame_variance": frame_variance,
        }

        return hud_info

    def simulate_situation_detection(
        self, hud_info: dict[str, Any], frame_number: int
    ) -> list[dict[str, Any]]:
        """
        Simulate situation detection based on HUD info.

        Args:
            hud_info: HUD information
            frame_number: Current frame number

        Returns:
            List: Detected situations
        """
        situations = []

        down = hud_info.get("down", 1)
        distance = hud_info.get("distance", 10)
        confidence = hud_info.get("confidence", 0.0)

        # Simulate high-value situations
        if down == 3 and distance >= 7:
            situations.append(
                {
                    "type": "3rd_and_long",
                    "confidence": min(confidence * 1.2, 0.95),
                    "frame": frame_number,
                    "details": {"down": down, "distance": distance},
                }
            )
        elif down == 4:
            situations.append(
                {
                    "type": "4th_down",
                    "confidence": min(confidence * 1.1, 0.95),
                    "frame": frame_number,
                    "details": {"down": down, "distance": distance},
                }
            )

        # Add more situation types based on other factors
        if hud_info.get("frame_variance", 0) > 1000:  # High action
            situations.append(
                {
                    "type": "high_action",
                    "confidence": 0.75,
                    "frame": frame_number,
                    "details": {"variance": hud_info.get("frame_variance")},
                }
            )

        return situations

    def is_significant_situation(self, situations: list[dict[str, Any]]) -> bool:
        """
        Determine if situations are significant enough for clipping.

        Args:
            situations: List of detected situations

        Returns:
            bool: True if significant
        """
        for situation in situations:
            sit_type = situation.get("type", "")
            confidence = situation.get("confidence", 0.0)

            # High-value situations
            if sit_type in ["3rd_and_long", "4th_down", "red_zone", "turnover"]:
                if confidence >= self.config["confidence_threshold"]:
                    return True

            # Medium-value situations need higher confidence
            elif sit_type in ["high_action", "scoring_play"]:
                if confidence >= (self.config["confidence_threshold"] + 0.1):
                    return True

        return False

    def analyze_video_optimized(
        self, video_path: str, progress_callback=None
    ) -> list[dict[str, Any]]:
        """
        Analyze video with optimized processing.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates

        Returns:
            List: Detected clips with timing and metadata
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(f"Starting optimized analysis: {total_frames} frames at {fps} FPS")
        self.logger.info(
            f"Frame skip: {self.config['frame_skip']}, Scene check: {self.config['scene_check_interval']}"
        )

        # Optimization tracking
        start_time = time.time()
        frames_processed = 0
        frames_skipped = 0
        scene_changes_detected = 0
        clips_detected = 0

        frame_number = 0
        previous_frame = None
        last_scene_check = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Progress callback
            if progress_callback and frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                progress_callback(frame_number, total_frames, progress)

            # Smart frame skipping with scene change detection
            if frame_number % self.config["scene_check_interval"] == 0:
                # Check for scene change
                if previous_frame is not None:
                    scene_changed = self.detect_scene_change(previous_frame, frame)
                    last_scene_check = frame_number

                    if scene_changed:
                        scene_changes_detected += 1
                        # Process this frame due to scene change
                    else:
                        # Skip frame - no significant change
                        frames_skipped += 1
                        previous_frame = frame
                        continue

            else:
                # Regular frame skipping
                if frame_number % self.config["frame_skip"] != 0:
                    frames_skipped += 1
                    continue

            # Process frame
            frames_processed += 1
            processed_frame = self.preprocess_frame(frame)

            # Simulate HUD and situation detection
            hud_info = self.simulate_hud_detection(processed_frame)
            situations = self.simulate_situation_detection(hud_info, frame_number)

            # Check if situations are significant
            if situations and self.is_significant_situation(situations):
                timestamp = frame_number / fps

                # Check clips per minute limit
                clips_in_last_minute = len(
                    [clip for clip in self.detected_clips if clip["timestamp"] > (timestamp - 60)]
                )

                if clips_in_last_minute < self.config["max_clips_per_minute"]:
                    clip_data = {
                        "timestamp": timestamp,
                        "frame_number": frame_number,
                        "situations": situations,
                        "hud_info": hud_info,
                        "confidence": max(s.get("confidence", 0.0) for s in situations),
                        "hardware_tier": self.hardware_tier,
                        "optimization_used": "smart_skipping_and_scene_detection",
                    }

                    self.detected_clips.append(clip_data)
                    clips_detected += 1

                    self.logger.info(
                        f"Clip detected at {timestamp:.1f}s: {situations[0].get('type', 'unknown')}"
                    )

            previous_frame = frame

        cap.release()

        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        processing_rate = frames_processed / total_time if total_time > 0 else 0

        # Log optimization results
        self.logger.info(f"Optimization Results:")
        self.logger.info(f"  Total time: {total_time:.2f}s")
        self.logger.info(
            f"  Frames processed: {frames_processed}/{total_frames} ({frames_processed/total_frames*100:.1f}%)"
        )
        self.logger.info(
            f"  Frames skipped: {frames_skipped} ({frames_skipped/total_frames*100:.1f}%)"
        )
        self.logger.info(f"  Scene changes detected: {scene_changes_detected}")
        self.logger.info(f"  Clips detected: {clips_detected}")
        self.logger.info(f"  Processing rate: {processing_rate:.1f} frames/second")

        # Add performance metadata to results
        performance_data = {
            "total_processing_time": total_time,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "scene_changes_detected": scene_changes_detected,
            "processing_rate_fps": processing_rate,
            "optimization_efficiency": frames_skipped / total_frames * 100,
            "hardware_tier": self.hardware_tier,
            "config_used": self.config,
        }

        return {
            "clips": self.detected_clips,
            "performance": performance_data,
            "total_clips": clips_detected,
        }


def demo_optimized_detection():
    """Demonstrate optimized auto-clip detection."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test with different hardware tiers
    tiers = ["low", "medium", "high", "ultra"]
    video_path = "test_videos/sample.mp4"

    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        print("Please ensure test video exists for benchmarking")
        return

    print("🚀 Starting Optimized Auto-Clip Detection Demo")
    print("=" * 60)

    results = {}

    for tier in tiers:
        print(f"\n🔧 Testing {tier.upper()} tier optimization...")

        detector = OptimizedAutoClipDetector(hardware_tier=tier)

        def progress_callback(current, total, percent):
            if current % 500 == 0:  # Print every 500 frames
                print(f"   Progress: {percent:.1f}% ({current}/{total} frames)")

        try:
            start_time = time.time()
            result = detector.analyze_video_optimized(video_path, progress_callback)
            end_time = time.time()

            results[tier] = {
                "clips_detected": result["total_clips"],
                "processing_time": end_time - start_time,
                "performance": result["performance"],
            }

            print(
                f"   ✅ {tier.upper()} tier: {result['total_clips']} clips in {end_time - start_time:.2f}s"
            )

        except Exception as e:
            print(f"   ❌ {tier.upper()} tier failed: {e}")
            results[tier] = {"error": str(e)}

    # Compare results
    print("\n" + "=" * 60)
    print("📊 OPTIMIZATION COMPARISON RESULTS")
    print("=" * 60)

    for tier, result in results.items():
        if "error" in result:
            print(f"{tier.upper():8} | ERROR: {result['error']}")
        else:
            perf = result["performance"]
            efficiency = perf.get("optimization_efficiency", 0)
            rate = perf.get("processing_rate_fps", 0)

            print(
                f"{tier.upper():8} | "
                f"{result['clips_detected']:2d} clips | "
                f"{result['processing_time']:6.2f}s | "
                f"{efficiency:5.1f}% skipped | "
                f"{rate:5.1f} fps"
            )

    print("\n💡 Optimization Benefits:")
    print("   - Smart frame skipping reduces processing by 60-90%")
    print("   - Scene change detection focuses on action sequences")
    print("   - Hardware-adaptive settings optimize for your system")
    print("   - Confidence thresholds reduce false positives")


if __name__ == "__main__":
    demo_optimized_detection()

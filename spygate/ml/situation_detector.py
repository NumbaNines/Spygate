"""ML-based situation detection for gameplay clips with enhanced HUD analysis."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from ..video.motion_detector import MotionDetector
from .hud_detector import HUDDetector

logger = logging.getLogger(__name__)


class SituationDetector:
    """Enhanced situation detector with YOLOv8-based HUD analysis and hardware-aware motion detection."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the situation detector."""
        self.initialized = False
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.motion_detector = None
        self.hud_detector = None
        self.motion_history = []
        self.situation_history = []
        self.min_situation_confidence = 0.6
        self.model_path = model_path

    def initialize(self):
        """Initialize components with hardware-aware configuration."""
        try:
            # Initialize motion detector with hardware-optimized settings
            self.motion_detector = MotionDetector(
                use_gpu=self.hardware.has_cuda,
                num_threads=(
                    self.optimizer.get_thread_count()
                    if hasattr(self.optimizer, "get_thread_count")
                    else 4
                ),
            )

            # Initialize HUD detector with YOLOv8
            self.hud_detector = HUDDetector(model_path=self.model_path)
            self.hud_detector.initialize()

            # Configure ROIs for the football field
            field_rois = self._get_default_field_rois()
            self.motion_detector.set_rois(field_rois)

            self.initialized = True
            logger.info("Situation detector initialized with YOLOv8 HUD analysis")

        except Exception as e:
            logger.error(f"Failed to initialize situation detector: {e}")
            # For Phase 1, we can work with just HUD detection if motion detection fails
            try:
                self.hud_detector = HUDDetector(model_path=self.model_path)
                self.hud_detector.initialize()
                self.initialized = True
                logger.info(
                    "Situation detector initialized with HUD analysis only (motion detection disabled)"
                )
            except Exception as e2:
                logger.error(f"Failed to initialize even basic HUD detection: {e2}")
                raise

    def _get_default_field_rois(self) -> list[dict[str, Any]]:
        """Get default ROIs for different areas of the football field."""
        # These are relative coordinates (0-1) that will be scaled to frame size
        return [
            {
                "name": "backfield",
                "points": [(0.4, 0.3), (0.6, 0.3), (0.6, 0.5), (0.4, 0.5)],
            },
            {
                "name": "line_of_scrimmage",
                "points": [(0.2, 0.4), (0.8, 0.4), (0.8, 0.6), (0.2, 0.6)],
            },
            {
                "name": "defensive_secondary",
                "points": [(0.3, 0.6), (0.7, 0.6), (0.7, 0.8), (0.3, 0.8)],
            },
        ]

    def detect_situations(self, frame: np.ndarray, frame_number: int, fps: float) -> dict[str, Any]:
        """
        Detect situations in a video frame using enhanced HUD analysis.

        Args:
            frame: The video frame as a numpy array
            frame_number: The frame number in the video
            fps: Frames per second of the video

        Returns:
            Dict[str, Any]: Detected situations and their details
        """
        if not self.initialized:
            raise RuntimeError("Situation detector not initialized")

        # Get HUD information - this is the core of Phase 1
        hud_info = self.extract_hud_info(frame)

        # Detect motion using hardware-aware detector if available
        motion_result = None
        motion_features = {}

        if self.motion_detector:
            try:
                motion_result_obj = self.motion_detector.detect_motion(frame)
                # Convert MotionDetectionResult to dict format for compatibility
                motion_result = {
                    "motion_detected": motion_result_obj.motion_detected,
                    "score": np.count_nonzero(motion_result_obj.motion_mask)
                    / motion_result_obj.motion_mask.size,
                    "contours": [],  # Convert bounding boxes to contours if needed
                    "bounding_boxes": motion_result_obj.bounding_boxes,
                }
                motion_features = self._extract_motion_features(motion_result, frame.shape)
            except Exception as e:
                logger.warning(f"Motion detection failed, continuing with HUD only: {e}")

        # Detect situations based on HUD info (primary) and motion (secondary)
        situations = self._analyze_situations(motion_features, hud_info, frame_number, fps)

        # Update situation history
        if situations:
            self.situation_history.append(
                {
                    "frame_number": frame_number,
                    "timestamp": frame_number / fps,
                    "situations": situations,
                }
            )

        return {
            "frame_number": frame_number,
            "timestamp": frame_number / fps,
            "situations": situations,
            "hud_info": hud_info,
            "metadata": {
                "motion_score": float(motion_result["score"]) if motion_result else 0.0,
                "hud_confidence": hud_info.get("confidence", 0.0),
                "hardware_tier": self.optimizer.get_performance_tier(),
                "analysis_version": "2.0.0-phase1",
            },
        }

    def _extract_motion_features(
        self, motion_result: dict[str, Any], frame_shape: tuple[int, int, int]
    ) -> dict[str, Any]:
        """Extract meaningful features from motion detection results."""
        features = {
            "overall_motion": motion_result.get("score", 0.0),
            "motion_regions": [],
            "motion_patterns": [],
        }

        if not motion_result.get("motion_detected", False):
            return features

        height, width = frame_shape[:2]

        # Analyze each motion region - use bounding boxes if contours not available
        contours = motion_result.get("contours", [])
        bounding_boxes = motion_result.get("bounding_boxes", [])

        if contours:
            for roi in self._contours_to_roi(contours):
                # Calculate relative position
                rel_x = roi["center"]["x"] / width
                rel_y = roi["center"]["y"] / height
                rel_area = roi["area"] / (width * height)

                # Determine region of the field
                field_region = self._determine_field_region(rel_x, rel_y)

                features["motion_regions"].append(
                    {
                        "region": field_region,
                        "relative_position": (rel_x, rel_y),
                        "relative_area": rel_area,
                        "intensity": motion_result["score"],
                    }
                )
        elif bounding_boxes:
            # Use bounding boxes directly
            for x, y, w, h in bounding_boxes:
                # Calculate center and area from bounding box
                center_x = x + w / 2
                center_y = y + h / 2
                area = w * h

                # Calculate relative position
                rel_x = center_x / width
                rel_y = center_y / height
                rel_area = area / (width * height)

                # Determine region of the field
                field_region = self._determine_field_region(rel_x, rel_y)

                features["motion_regions"].append(
                    {
                        "region": field_region,
                        "relative_position": (rel_x, rel_y),
                        "relative_area": rel_area,
                        "intensity": motion_result["score"],
                    }
                )

        # Analyze motion patterns if we have history (note: basic MotionDetector doesn't store history)
        # This would need to be implemented if motion pattern analysis is required
        # For now, skip motion pattern analysis to avoid dependency on motion_history

        return features

    def _contours_to_roi(self, contours: list) -> list[dict[str, Any]]:
        """Convert contours to ROI dictionaries."""
        rois = []
        for contour in contours:
            try:
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    center_x = moments["m10"] / moments["m00"]
                    center_y = moments["m01"] / moments["m00"]
                    area = cv2.contourArea(contour)

                    rois.append({"center": {"x": center_x, "y": center_y}, "area": area})
            except Exception as e:
                logger.debug(f"Error processing contour: {e}")
                continue

        return rois

    def _determine_field_region(self, rel_x: float, rel_y: float) -> str:
        """Determine which region of the field a point belongs to."""
        if rel_y < 0.4:
            return "backfield"
        elif rel_y < 0.6:
            return "line_of_scrimmage"
        else:
            return "defensive_secondary"

    def _analyze_motion_patterns(
        self, motion_history: list[dict[str, Any]], frame_shape: tuple[int, int, int]
    ) -> list[dict[str, Any]]:
        """Analyze motion patterns from recent history."""
        patterns = []

        # Skip if not enough history
        if len(motion_history) < 2:
            return patterns

        # Get motion centroids from history
        centroids = []
        for entry in motion_history:
            contours = entry.get("contours", [])
            if contours:
                try:
                    moments = cv2.moments(contours[0])
                    if moments["m00"] != 0:
                        cx = moments["m10"] / moments["m00"]
                        cy = moments["m01"] / moments["m00"]
                        centroids.append((cx, cy))
                except Exception:
                    continue

        # Analyze motion direction and speed
        if len(centroids) >= 2:
            dx = centroids[-1][0] - centroids[-2][0]
            dy = centroids[-1][1] - centroids[-2][1]
            speed = np.sqrt(dx * dx + dy * dy)
            angle = np.arctan2(dy, dx) * 180 / np.pi

            patterns.append(
                {
                    "type": "linear_motion",
                    "speed": float(speed),
                    "angle": float(angle),
                    "direction": self._classify_direction(angle),
                }
            )

        return patterns

    def _classify_direction(self, angle: float) -> str:
        """Classify motion direction based on angle."""
        if -45 <= angle <= 45:
            return "right"
        elif 45 < angle <= 135:
            return "down"
        elif angle > 135 or angle <= -135:
            return "left"
        else:
            return "up"

    def _analyze_situations(
        self,
        motion_features: dict[str, Any],
        hud_info: dict[str, Any],
        frame_number: int,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Analyze and classify situations based on HUD info (primary) and motion features (secondary)."""
        situations = []
        timestamp = frame_number / fps

        # Primary: HUD-based situation detection (Phase 1 core functionality)
        hud_situations = self._analyze_hud_situations(hud_info, frame_number, timestamp)
        situations.extend(hud_situations)

        # Secondary: Motion-based situation detection (if available)
        if motion_features and motion_features.get("overall_motion", 0) > 0.3:
            motion_situations = self._analyze_motion_situations(
                motion_features, frame_number, timestamp
            )
            situations.extend(motion_situations)

        return situations

    def _analyze_hud_situations(
        self, hud_info: dict[str, Any], frame_number: int, timestamp: float
    ) -> list[dict[str, Any]]:
        """Analyze situations based on HUD information - core Phase 1 functionality."""
        situations = []

        # Get confidence level from HUD detection
        hud_confidence = hud_info.get("confidence", 0.0)

        if hud_confidence < self.min_situation_confidence:
            return situations

        # Detect game state (pre-snap, during play, post-play)
        game_state = self._detect_game_state(hud_info)
        
        # Add game state as a situation for tracking
        if game_state:
            situations.append({
                "type": f"game_state_{game_state}",
                "confidence": min(hud_confidence * 1.0, 0.95),
                "frame": frame_number,
                "timestamp": timestamp,
                "details": {
                    "game_state": game_state,
                    "play_clock_visible": hud_info.get("play_clock") is not None,
                    "source": "hud_analysis",
                },
            })

        # Detect critical game situations based on HUD data
        down = hud_info.get("down")
        distance = hud_info.get("distance")
        yard_line = hud_info.get("yard_line")
        game_clock = hud_info.get("game_clock")
        score_home = hud_info.get("score_home")
        score_away = hud_info.get("score_away")

        # 3rd Down situations
        if down == 3:
            situation_type = "3rd_down"
            if distance and distance >= 7:
                situation_type = "3rd_and_long"
            elif distance and distance <= 3:
                situation_type = "3rd_and_short"

            situations.append(
                {
                    "type": situation_type,
                    "confidence": min(hud_confidence * 1.2, 0.95),
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "details": {
                        "down": down,
                        "distance": distance,
                        "yard_line": yard_line,
                        "source": "hud_analysis",
                    },
                }
            )

        # 4th Down situations
        if down == 4:
            situations.append(
                {
                    "type": "4th_down",
                    "confidence": min(hud_confidence * 1.1, 0.95),
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "details": {
                        "down": down,
                        "distance": distance,
                        "yard_line": yard_line,
                        "source": "hud_analysis",
                    },
                }
            )

        # Red Zone situations
        if yard_line and "OPP" in str(yard_line):
            try:
                yard_num = int(str(yard_line).split()[-1])
                if yard_num <= 20:
                    situations.append(
                        {
                            "type": "red_zone",
                            "confidence": min(hud_confidence * 1.1, 0.95),
                            "frame": frame_number,
                            "timestamp": timestamp,
                            "details": {
                                "yard_line": yard_line,
                                "yards_to_goal": yard_num,
                                "source": "hud_analysis",
                            },
                        }
                    )
            except (ValueError, IndexError):
                pass

        # Two-minute warning situations
        if game_clock:
            try:
                # Parse clock format like "2:00" or "1:30"
                if ":" in game_clock:
                    minutes, seconds = game_clock.split(":")
                    total_seconds = int(minutes) * 60 + int(seconds)

                    if total_seconds <= 120:  # 2 minutes or less
                        situations.append(
                            {
                                "type": "two_minute_warning",
                                "confidence": min(hud_confidence * 1.0, 0.95),
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "details": {
                                    "game_clock": game_clock,
                                    "time_remaining": total_seconds,
                                    "source": "hud_analysis",
                                },
                            }
                        )
            except (ValueError, IndexError):
                pass

        # Score differential situations
        if score_home is not None and score_away is not None:
            score_diff = abs(score_home - score_away)

            if score_diff <= 3:
                situations.append(
                    {
                        "type": "close_game",
                        "confidence": min(hud_confidence * 1.0, 0.95),
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "details": {
                            "score_home": score_home,
                            "score_away": score_away,
                            "score_difference": score_diff,
                            "source": "hud_analysis",
                        },
                    }
                )

        return situations

    def _detect_game_state(self, hud_info: dict[str, Any]) -> Optional[str]:
        """
        Detect current game state based on HUD elements.
        
        Returns:
            str: One of 'pre_snap', 'during_play', 'post_play', or None
        """
        play_clock = hud_info.get("play_clock")
        
        # Primary indicator: Play clock visibility
        if play_clock is not None:
            # Play clock visible = pre-snap
            try:
                # Additional validation: play clock should be counting down
                play_clock_value = int(str(play_clock).replace(":", ""))
                if 1 <= play_clock_value <= 40:  # Valid play clock range
                    return "pre_snap"
            except (ValueError, TypeError):
                # If we can't parse the play clock, still assume pre-snap if visible
                return "pre_snap"
        else:
            # Play clock not visible - could be during play or post-play
            # We need additional logic to distinguish these states
            
            # Check for other indicators
            down = hud_info.get("down")
            distance = hud_info.get("distance")
            
            # If we have down/distance info, likely during active play
            if down is not None and distance is not None:
                return "during_play"
            else:
                # Limited info available, could be post-play or transition
                return "post_play"
        
        return None

    def _analyze_motion_situations(
        self, motion_features: dict[str, Any], frame_number: int, timestamp: float
    ) -> list[dict[str, Any]]:
        """Analyze situations based on motion patterns - secondary analysis."""
        situations = []

        # Check for high motion events
        if motion_features["overall_motion"] > 0.5:
            # Analyze motion patterns
            for pattern in motion_features.get("motion_patterns", []):
                if pattern["type"] == "linear_motion":
                    # Detect running plays
                    if pattern["direction"] in ["right", "left"]:
                        confidence = min(
                            motion_features["overall_motion"] * 1.2, 0.85
                        )  # Lower confidence for motion-only
                        if confidence > 0.5:  # Lower threshold for motion
                            situations.append(
                                {
                                    "type": "running_play_motion",
                                    "confidence": confidence,
                                    "frame": frame_number,
                                    "timestamp": timestamp,
                                    "details": {
                                        "direction": pattern["direction"],
                                        "speed": pattern["speed"],
                                        "motion_score": motion_features["overall_motion"],
                                        "source": "motion_analysis",
                                    },
                                }
                            )

            # Analyze motion regions
            for region in motion_features.get("motion_regions", []):
                # Detect passing plays
                if region["region"] == "backfield" and region["relative_area"] > 0.1:
                    confidence = min(
                        motion_features["overall_motion"] * 1.1, 0.85
                    )  # Lower confidence for motion-only
                    if confidence > 0.5:  # Lower threshold for motion
                        situations.append(
                            {
                                "type": "passing_play_motion",
                                "confidence": confidence,
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "details": {
                                    "region": region["region"],
                                    "motion_score": motion_features["overall_motion"],
                                    "source": "motion_analysis",
                                },
                            }
                        )

        return situations

    def analyze_sequence(
        self,
        frames: list[np.ndarray],
        start_frame: int,
        fps: float,
        window_size: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Analyze a sequence of frames for temporal patterns.

        Args:
            frames: List of video frames
            start_frame: Starting frame number
            fps: Frames per second of the video
            window_size: Size of the sliding window for temporal analysis

        Returns:
            List[Dict[str, Any]]: Detected situations for each frame
        """
        results = []

        # Process each frame
        for i, frame in enumerate(frames):
            frame_number = start_frame + i
            result = self.detect_situations(frame, frame_number, fps)
            results.append(result)

            # Perform temporal analysis on window
            if len(results) >= window_size:
                window = results[-window_size:]
                temporal_situations = self._analyze_temporal_patterns(window, fps)

                # Add temporal situations to the current frame's results
                results[-1]["situations"].extend(temporal_situations)

        return results

    def _analyze_temporal_patterns(
        self, window: list[dict[str, Any]], fps: float
    ) -> list[dict[str, Any]]:
        """Analyze temporal patterns in a window of frames."""
        patterns = []

        # Count situation types in window
        situation_counts = {}
        for frame in window:
            for situation in frame["situations"]:
                sit_type = situation["type"]
                situation_counts[sit_type] = situation_counts.get(sit_type, 0) + 1

        # Detect sustained situations
        window_size = len(window)
        for sit_type, count in situation_counts.items():
            if count >= window_size * 0.6:  # Present in 60% of frames
                patterns.append(
                    {
                        "type": f"sustained_{sit_type}",
                        "confidence": min(count / window_size * 1.2, 0.95),
                        "frame": window[-1]["frame_number"],
                        "timestamp": window[-1]["timestamp"],
                        "details": {
                            "duration": window_size / fps,
                            "frequency": count / window_size,
                            "source": "temporal_analysis",
                        },
                    }
                )

        return patterns

    def extract_hud_info(self, frame: np.ndarray) -> dict[str, Any]:
        """Extract HUD information from a frame using YOLOv8-based detection."""
        if not self.initialized:
            raise RuntimeError("Situation detector not initialized")

        # Use HUD detector to get game state
        hud_info = self.hud_detector.get_game_state(frame)

        return hud_info

    def detect_mistakes(
        self, situations: list[dict[str, Any]], hud_info: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Detect potential mistakes in gameplay.

        Args:
            situations: List of detected situations
            hud_info: HUD information

        Returns:
            List[Dict[str, Any]]: Detected mistakes
        """
        # Phase 1: Basic mistake detection based on HUD patterns
        mistakes = []

        # Example: Detect potential timeout usage mistakes
        timeouts = hud_info.get("timeouts", {})
        game_clock = hud_info.get("game_clock")

        if game_clock and timeouts:
            # Add basic timeout strategy analysis
            pass

        # Example: Detect potential down and distance issues
        down = hud_info.get("down")
        distance = hud_info.get("distance")

        if down == 4 and distance and distance > 5:
            # Potential risky 4th down attempt
            mistakes.append(
                {
                    "type": "risky_4th_down",
                    "severity": "medium",
                    "description": f"4th down with {distance} yards to go",
                    "details": {"down": down, "distance": distance},
                }
            )

        return mistakes

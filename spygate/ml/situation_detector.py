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

        # Extract key elements for analysis
        down_distance = hud_info.get("down_distance", "")
        game_clock = hud_info.get("game_clock", "")
        play_clock = hud_info.get("play_clock", "")
        yards_to_goal = hud_info.get("yards_to_goal", "")  # Numeric value (e.g., "25", "3", "GL")
        territory_indicator = hud_info.get("territory_indicator", "")  # ▲ = opponent territory, ▼ = own territory
        
        # Determine field position from yards_to_goal + territory_indicator
        yard_line = self._construct_field_position(yards_to_goal, territory_indicator)

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
                    
                    # Goal line situations (hash marks strategy)
                    if yard_num <= 5:
                        situations.append(
                            {
                                "type": "goal_line",
                                "confidence": min(hud_confidence * 1.2, 0.95),
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "details": {
                                    "yard_line": yard_line,
                                    "yards_to_goal": yard_num,
                                    "hash_marks_context": "goal_line_situation",
                                    "strategic_importance": "high",
                                    "source": "hash_marks_analysis",
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

        # Hash marks strategic analysis for field position
        if yard_line:
            qb_position = hud_info.get("qb_position")  # Get QB position for hash mark analysis
            hash_marks_context = self._analyze_hash_marks_position(yard_line, qb_position)
            if hash_marks_context:
                situations.append(
                    {
                        "type": "hash_marks_position",
                        "confidence": min(hud_confidence * 1.0, 0.90),
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "details": {
                            "yard_line": yard_line,
                            "qb_position": qb_position,
                            "hash_marks_zone": hash_marks_context["zone"],
                            "hash_mark_side": hash_marks_context.get("hash_mark_side"),
                            "strategic_implications": hash_marks_context["implications"],
                            "kicking_angle": hash_marks_context.get("kicking_angle"),
                            "source": "hash_marks_analysis",
                        },
                    }
                )

        return situations

    def _detect_game_state(self, hud_info: dict[str, Any]) -> Optional[str]:
        """
        Detect current game state with multiple fallback methods.
        Handles cases where HUD elements or overlays might be missing.
        
        Returns:
            str: One of 'pre_snap', 'during_play', 'post_play', or None
        """
        # Method 1: Play clock visibility (primary indicator)
        play_clock = hud_info.get("play_clock")
        if play_clock is not None:
            try:
                # Play clock visible = pre-snap
                play_clock_value = int(str(play_clock).replace(":", ""))
                if 1 <= play_clock_value <= 40:  # Valid play clock range
                    return "pre_snap"
            except (ValueError, TypeError):
                # If we can't parse but it exists, still likely pre-snap
                return "pre_snap"
        
        # Method 2: Check for explicit game state overlays
        explicit_state = self._detect_explicit_game_state_overlays(hud_info)
        if explicit_state:
            return explicit_state
            
        # Method 3: Motion-based detection (fallback for clips without HUD)
        motion_state = self._detect_game_state_from_motion(hud_info)
        if motion_state:
            return motion_state
            
        # Method 4: Formation-based detection
        formation_state = self._detect_game_state_from_formation(hud_info)
        if formation_state:
            return formation_state
            
        # Method 5: Down/Distance change analysis
        down_distance_state = self._detect_game_state_from_down_distance_changes(hud_info)
        if down_distance_state:
            return down_distance_state
            
        # Method 6: Context clues from available HUD elements
        context_state = self._detect_game_state_from_context(hud_info)
        if context_state:
            return context_state
            
        return None

    def _detect_explicit_game_state_overlays(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Detect explicit game state from text overlays like 'Pre-play'."""
        # Check for explicit text overlays (when available)
        overlays = hud_info.get("text_overlays", [])
        if isinstance(overlays, list):
            overlay_text = " ".join(str(overlay).lower() for overlay in overlays)
        else:
            overlay_text = str(overlays).lower()
            
        if "pre-play" in overlay_text or "pre play" in overlay_text:
            return "pre_snap"
        elif "during play" in overlay_text or "live" in overlay_text:
            return "during_play"
        elif "post-play" in overlay_text or "after play" in overlay_text:
            return "post_play"
            
        return None

    def _detect_game_state_from_motion(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Detect game state based on motion patterns (for clips without HUD)."""
        # This would analyze motion vectors, but for now return None
        # In full implementation, this would:
        # - Analyze player movement patterns
        # - Detect ball movement
        # - Identify formation changes
        return None

    def _detect_game_state_from_formation(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Detect game state from player formations and positioning."""
        # This would analyze player positions, but for now return None
        # In full implementation, this would:
        # - Detect set formations (pre-snap)
        # - Identify players in motion (during play)
        # - Recognize celebration/huddle patterns (post-play)
        return None

    def _detect_game_state_from_down_distance_changes(self, hud_info: dict[str, Any]) -> Optional[str]:
        """
        Detect game state based on down and distance patterns and changes.
        
        Args:
            hud_info: Current HUD information
            
        Returns:
            str: Inferred game state or None
        """
        down_distance = hud_info.get("down_distance", "")
        if not down_distance:
            return None
            
        # Parse current down and distance
        parsed = self._extract_down_distance_with_fallbacks(hud_info)
        if not parsed:
            return None
            
        down = parsed.get("down")
        distance = parsed.get("distance")
        
        # Analyze down/distance patterns for game state clues
        
        # Pattern 1: Fresh first down (likely pre-snap of new drive/play)
        if down == 1 and isinstance(distance, int) and distance >= 10:
            return "pre_snap"  # New drive, fresh down
            
        # Pattern 2: Goal line situations (high detail = pre-snap)
        if distance == "goal":
            return "pre_snap"  # Goal line stands are typically shown pre-snap
            
        # Pattern 3: 4th down situations (critical - usually shown with detail pre-snap)
        if down == 4:
            return "pre_snap"  # 4th down decisions shown pre-snap
            
        # Pattern 4: Short yardage situations (2 yards or less - high detail pre-snap)
        if isinstance(distance, int) and distance <= 2:
            return "pre_snap"  # Short yardage situations get detailed pre-snap display
            
        # Pattern 5: Standard down situations with clean display
        if down in [1, 2, 3] and isinstance(distance, int) and 3 <= distance <= 20:
            # Clean, detailed down/distance usually indicates pre-snap
            return "pre_snap"
            
        # Pattern 6: Unclear or partial down/distance (during play)
        # If we can't parse clearly, might be during action when HUD is minimal
        return None

    def _detect_game_state_from_context(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Use context clues from available HUD elements to infer game state."""
        down = hud_info.get("down")
        distance = hud_info.get("distance")
        game_clock = hud_info.get("game_clock")
        
        # If we have detailed down/distance info, likely pre-snap or post-play
        if down is not None and distance is not None:
            # Check if this looks like a fresh situation (likely pre-snap)
            if isinstance(distance, int) and distance > 0:
                return "pre_snap"  # Fresh down with yards to go
        
        # If no specific indicators but we have game clock, assume during play
        if game_clock and not any([down, distance]):
            return "during_play"
            
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

    def _analyze_hash_marks_position(self, yard_line: str, qb_position: str) -> Optional[dict[str, Any]]:
        """
        Analyze field position relative to hash marks for strategic implications.
        
        Args:
            yard_line: Field position string (e.g., "OPP 25", "OWN 35", "2-PT")
            qb_position: QB position string (e.g., "left", "right", "center")
            
        Returns:
            Dict with hash marks analysis or None if not applicable
        """
        if not yard_line:
            return None
            
        try:
            # Handle special situations (these would come from other field overlays, not yards_to_goal)
            if "2-PT" in yard_line or "XP" in yard_line:
                return {
                    "zone": "conversion_attempt",
                    "hash_mark_side": "center",  # Conversions typically start at center
                    "implications": ["short_yardage", "high_pressure"],
                    "kicking_angle": "center"
                }
            
            # Determine hash mark side based on QB position
            hash_mark_side = "unknown"
            if qb_position:
                if "left" in str(qb_position).lower():
                    hash_mark_side = "left_hash"
                elif "right" in str(qb_position).lower():
                    hash_mark_side = "right_hash"
                elif "center" in str(qb_position).lower():
                    hash_mark_side = "between_hashes"
            
            # Parse yard line for strategic analysis
            if "OPP" in yard_line:
                try:
                    yard_num = int(yard_line.split()[-1])
                    
                    # Determine strategic implications based on hash mark position
                    implications = []
                    kicking_angle = "unknown"
                    
                    if hash_mark_side == "left_hash":
                        implications.extend(["left_hash_tendency", "right_side_field_advantage"])
                        kicking_angle = "angled_right"
                    elif hash_mark_side == "right_hash":
                        implications.extend(["right_hash_tendency", "left_side_field_advantage"])
                        kicking_angle = "angled_left"
                    elif hash_mark_side == "between_hashes":
                        implications.extend(["center_field", "optimal_kicking_angle"])
                        kicking_angle = "straight"
                    
                    # Add distance-based implications
                    if yard_num <= 5:
                        implications.append("goal_line_stand")
                        zone = "goal_line"
                    elif yard_num <= 20:
                        implications.append("red_zone")
                        zone = "red_zone"
                    elif yard_num <= 35:
                        implications.append("scoring_territory")
                        zone = "scoring_territory"
                    else:
                        zone = "opponent_territory"
                    
                    return {
                        "zone": zone,
                        "hash_mark_side": hash_mark_side,
                        "implications": implications,
                        "kicking_angle": kicking_angle,
                        "yards_to_goal": yard_num
                    }
                    
                except (ValueError, IndexError):
                    pass
                    
            elif "OWN" in yard_line:
                # Own territory - different strategic considerations
                try:
                    yard_num = int(yard_line.split()[-1])
                    implications = ["own_territory"]
                    
                    if hash_mark_side == "left_hash":
                        implications.append("left_hash_own_territory")
                    elif hash_mark_side == "right_hash":
                        implications.append("right_hash_own_territory")
                    
                    if yard_num <= 10:
                        implications.append("deep_own_territory")
                        zone = "deep_own_territory"
                    elif yard_num <= 25:
                        implications.append("own_red_zone")
                        zone = "own_red_zone"
                    else:
                        zone = "own_territory"
                        
                    return {
                        "zone": zone,
                        "hash_mark_side": hash_mark_side,
                        "implications": implications,
                        "kicking_angle": "punt_consideration",
                        "yards_to_own_goal": yard_num
                    }
                    
                except (ValueError, IndexError):
                    pass
                    
        except Exception as e:
            logger.warning(f"Error analyzing hash marks position: {e}")
            
        return None

    def _construct_field_position(self, yards_to_goal: str, territory_indicator: str) -> str:
        """Construct field position string based on yards_to_goal and territory_indicator."""
        if not yards_to_goal or not territory_indicator:
            return "unknown"
        
        # Handle numeric yards_to_goal (should be integer)
        try:
            yards = int(str(yards_to_goal))
        except (ValueError, TypeError):
            return "unknown"
        
        # Combine yards_to_goal and territory_indicator 
        if territory_indicator == "▲" or "OPP" in str(territory_indicator).upper():
            return f"OPP {yards}"
        elif territory_indicator == "▼" or "OWN" in str(territory_indicator).upper():
            return f"OWN {yards}"
        else:
            return "unknown"

    def detect_situation_from_partial_clip(
        self, 
        hud_info: dict[str, Any], 
        frame_number: int = 0, 
        timestamp: float = 0.0,
        previous_hud_info: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Detect game situation from potentially incomplete clips.
        Handles cases where HUD elements might be missing or partially visible.
        
        Args:
            hud_info: Detected HUD elements (may be incomplete)
            frame_number: Frame number in the clip
            timestamp: Timestamp in the clip
            previous_hud_info: Previous frame HUD information (if available)
            
        Returns:
            dict: Situation analysis with confidence levels and fallback methods used
        """
        situation_result = {
            "detected_elements": {},
            "missing_elements": [],
            "confidence": 0.0,
            "fallback_methods_used": [],
            "game_state": None,
            "field_position": None,
            "down_distance": None,
            "time_context": None,
            "strategic_context": None,
            "down_distance_changes": None
        }
        
        # Assess what HUD elements are available
        available_elements = self._assess_available_elements(hud_info)
        situation_result["detected_elements"] = available_elements["found"]
        situation_result["missing_elements"] = available_elements["missing"]
        
        # Base confidence on available elements
        base_confidence = len(available_elements["found"]) / len(UI_CLASSES) * 0.8
        
        # Game State Detection (with fallbacks)
        game_state = self._detect_game_state(hud_info)
        if game_state:
            situation_result["game_state"] = game_state
            base_confidence += 0.1
        else:
            # Try alternative methods for clips without clear indicators
            game_state = self._infer_game_state_from_clip_characteristics(hud_info)
            if game_state:
                situation_result["game_state"] = game_state
                situation_result["fallback_methods_used"].append("clip_characteristics")
        
        # Down/Distance Change Analysis (if previous frame available)
        if previous_hud_info:
            dd_changes = self._track_down_distance_changes(hud_info, previous_hud_info)
            situation_result["down_distance_changes"] = dd_changes
            
            # Use change analysis to improve game state detection
            if not game_state and dd_changes.get("game_state_inference"):
                situation_result["game_state"] = dd_changes["game_state_inference"]
                situation_result["fallback_methods_used"].append("down_distance_changes")
                base_confidence += dd_changes.get("confidence", 0.0) * 0.1
        
        # Field Position (with fallbacks)
        field_position = self._extract_field_position_with_fallbacks(hud_info)
        if field_position:
            situation_result["field_position"] = field_position
            base_confidence += 0.1
        
        # Down and Distance (with fallbacks) 
        down_distance = self._extract_down_distance_with_fallbacks(hud_info)
        if down_distance:
            situation_result["down_distance"] = down_distance
            base_confidence += 0.1
            
        # Time Context
        time_context = self._extract_time_context(hud_info)
        if time_context:
            situation_result["time_context"] = time_context
            
        # Strategic Context (even with limited info)
        strategic_context = self._analyze_strategic_context_partial(
            situation_result["field_position"],
            situation_result["down_distance"], 
            situation_result["time_context"]
        )
        if strategic_context:
            situation_result["strategic_context"] = strategic_context
            
        situation_result["confidence"] = min(base_confidence, 0.95)
        return situation_result

    def _assess_available_elements(self, hud_info: dict[str, Any]) -> dict[str, list]:
        """Assess which HUD elements are available vs missing."""
        found = []
        missing = []
        
        for element in UI_CLASSES:
            if element in hud_info and hud_info[element] is not None:
                found.append(element)
            else:
                missing.append(element)
                
        return {"found": found, "missing": missing}

    def _infer_game_state_from_clip_characteristics(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Infer game state from clip characteristics when primary indicators missing."""
        # Check if we have any movement/action indicators
        # This is a simplified version - full implementation would analyze actual video frames
        
        # If we have detailed HUD info, likely pre-snap or post-play
        detailed_elements = ["down_distance", "yards_to_goal", "territory_indicator"]
        if any(elem in hud_info for elem in detailed_elements):
            return "pre_snap"  # Detailed info usually shown pre-snap
            
        # If only basic elements, might be during action
        basic_elements = ["game_clock", "score_bug"]
        if any(elem in hud_info for elem in basic_elements) and len(hud_info) <= 3:
            return "during_play"  # Minimal HUD during action
            
        return None

    def _extract_field_position_with_fallbacks(self, hud_info: dict[str, Any]) -> Optional[str]:
        """Extract field position with multiple fallback methods."""
        # Primary method: yards_to_goal + territory_indicator
        yards_to_goal = hud_info.get("yards_to_goal")
        territory_indicator = hud_info.get("territory_indicator")
        
        if yards_to_goal and territory_indicator:
            return self._construct_field_position(yards_to_goal, territory_indicator)
            
        # Fallback 1: Check for any field position text
        field_pos = hud_info.get("field_position")
        if field_pos:
            return str(field_pos)
            
        # Fallback 2: Infer from other context
        down_distance = hud_info.get("down_distance", "")
        if "Goal" in str(down_distance):
            return "RED_ZONE"  # "1st & Goal" indicates red zone
            
        return None

    def _extract_down_distance_with_fallbacks(self, hud_info: dict[str, Any]) -> Optional[dict]:
        """Extract down and distance with fallback parsing."""
        down_distance = hud_info.get("down_distance")
        if not down_distance:
            return None
            
        # Parse standard format
        import re
        # Try patterns like "1st & 10", "4th & Goal", "2nd & 3"
        pattern = r"(\d+)(?:st|nd|rd|th)?\s*&\s*(.+)"
        match = re.search(pattern, str(down_distance), re.IGNORECASE)
        
        if match:
            down = int(match.group(1))
            distance_text = match.group(2).strip()
            
            # Parse distance
            if distance_text.lower() == "goal":
                return {"down": down, "distance": "goal", "yards_to_go": 0}
            else:
                try:
                    yards = int(distance_text)
                    return {"down": down, "distance": yards, "yards_to_go": yards}
                except ValueError:
                    return {"down": down, "distance": distance_text, "yards_to_go": None}
                    
        return None

    def _extract_time_context(self, hud_info: dict[str, Any]) -> Optional[dict]:
        """Extract time context from available clock information."""
        game_clock = hud_info.get("game_clock")
        play_clock = hud_info.get("play_clock")
        
        time_context = {}
        
        if game_clock:
            time_context["game_clock"] = game_clock
            # Parse quarter/time if available
            if ":" in str(game_clock):
                time_context["time_urgency"] = "normal"
            elif any(urgent in str(game_clock).lower() for urgent in ["2:00", "1:00", ":30"]):
                time_context["time_urgency"] = "high"
                
        if play_clock:
            time_context["play_clock"] = play_clock
            # Assess play clock urgency
            try:
                clock_value = int(str(play_clock).replace(":", ""))
                if clock_value <= 10:
                    time_context["play_urgency"] = "high"
                elif clock_value <= 20:
                    time_context["play_urgency"] = "medium" 
                else:
                    time_context["play_urgency"] = "low"
            except (ValueError, TypeError):
                time_context["play_urgency"] = "unknown"
                
        return time_context if time_context else None

    def _analyze_strategic_context_partial(
        self, 
        field_position: Optional[str], 
        down_distance: Optional[dict], 
        time_context: Optional[dict]
    ) -> Optional[dict]:
        """Analyze strategic context even with partial information."""
        if not any([field_position, down_distance, time_context]):
            return None
            
        context = {"implications": [], "urgency": "normal"}
        
        # Field position implications
        if field_position:
            if "RED_ZONE" in field_position or "Goal" in str(field_position):
                context["implications"].append("red_zone_scoring_opportunity")
                context["urgency"] = "high"
            elif "OWN" in field_position and any(num in field_position for num in ["1", "2", "3", "4", "5"]):
                context["implications"].append("safety_risk")
                context["urgency"] = "high"
                
        # Down implications
        if down_distance:
            down = down_distance.get("down")
            if down == 4:
                context["implications"].append("critical_down")
                context["urgency"] = "high"
            elif down_distance.get("distance") == "goal":
                context["implications"].append("goal_line_stand")
                context["urgency"] = "high"
                
        # Time implications
        if time_context:
            if time_context.get("time_urgency") == "high":
                context["implications"].append("time_pressure")
                context["urgency"] = "high"
            elif time_context.get("play_urgency") == "high":
                context["implications"].append("play_clock_pressure")
                
        return context if context["implications"] else None

    def _track_down_distance_changes(self, current_hud: dict[str, Any], previous_hud: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Track changes in down and distance between frames to detect game state transitions.
        
        Args:
            current_hud: Current frame HUD data
            previous_hud: Previous frame HUD data (if available)
            
        Returns:
            dict: Change analysis with transition indicators
        """
        if not previous_hud:
            return {"change_type": "no_previous_data", "confidence": 0.0}
            
        # Extract current and previous down/distance
        current_dd = self._extract_down_distance_with_fallbacks(current_hud)
        previous_dd = self._extract_down_distance_with_fallbacks(previous_hud)
        
        if not current_dd or not previous_dd:
            return {"change_type": "incomplete_data", "confidence": 0.0}
            
        current_down = current_dd.get("down")
        current_dist = current_dd.get("distance")
        previous_down = previous_dd.get("down")
        previous_dist = previous_dd.get("distance")
        
        # Analyze change patterns
        change_analysis = {
            "change_type": "no_change",
            "confidence": 0.0,
            "game_state_inference": None,
            "details": {}
        }
        
        # Pattern 1: Down advancement (successful play completion)
        if current_down == 1 and previous_down in [2, 3, 4]:
            change_analysis.update({
                "change_type": "first_down_conversion",
                "confidence": 0.9,
                "game_state_inference": "pre_snap",  # New drive setup
                "details": {"previous_down": previous_down, "result": "conversion"}
            })
            
        # Pattern 2: Down increment (incomplete pass, failed rush)
        elif current_down == previous_down + 1:
            change_analysis.update({
                "change_type": "down_increment",
                "confidence": 0.8,
                "game_state_inference": "pre_snap",  # Setting up next play
                "details": {"down_change": f"{previous_down} to {current_down}"}
            })
            
        # Pattern 3: Distance reduction (successful rushing/passing yards)
        elif (isinstance(current_dist, int) and isinstance(previous_dist, int) and 
              current_down == previous_down and current_dist < previous_dist):
            yards_gained = previous_dist - current_dist
            change_analysis.update({
                "change_type": "yards_gained",
                "confidence": 0.7,
                "game_state_inference": "pre_snap",  # Post-play, setting up next
                "details": {"yards_gained": yards_gained, "down": current_down}
            })
            
        # Pattern 4: Distance increase (penalty)
        elif (isinstance(current_dist, int) and isinstance(previous_dist, int) and 
              current_down == previous_down and current_dist > previous_dist):
            penalty_yards = current_dist - previous_dist
            change_analysis.update({
                "change_type": "penalty",
                "confidence": 0.8,
                "game_state_inference": "pre_snap",  # Post-penalty setup
                "details": {"penalty_yards": penalty_yards, "down": current_down}
            })
            
        # Pattern 5: Complete change in down/distance (new drive)
        elif current_down != previous_down or current_dist != previous_dist:
            change_analysis.update({
                "change_type": "significant_change",
                "confidence": 0.6,
                "game_state_inference": "pre_snap",  # Major transition
                "details": {
                    "from": f"{previous_down} & {previous_dist}",
                    "to": f"{current_down} & {current_dist}"
                }
            })
            
        return change_analysis

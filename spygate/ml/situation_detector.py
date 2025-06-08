"""ML-based situation detection for gameplay clips."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from ..video.motion_detector import HardwareAwareMotionDetector
from .hud_detector import HUDDetector

logger = logging.getLogger(__name__)


class SituationDetector:
    """Enhanced situation detector with hardware-aware motion detection."""

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
        # Initialize motion detector with hardware-optimized settings
        self.motion_detector = HardwareAwareMotionDetector(
            use_gpu=self.hardware.has_cuda,
            use_threading=self.optimizer.should_use_threading(),
        )

        # Initialize HUD detector
        self.hud_detector = HUDDetector(model_path=self.model_path)
        self.hud_detector.initialize()

        # Configure ROIs for the football field
        field_rois = self._get_default_field_rois()
        self.motion_detector.set_rois(field_rois)

        self.initialized = True

    def _get_default_field_rois(self) -> List[Dict[str, Any]]:
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

    def detect_situations(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> Dict[str, Any]:
        """
        Detect situations in a video frame using enhanced motion detection.

        Args:
            frame: The video frame as a numpy array
            frame_number: The frame number in the video
            fps: Frames per second of the video

        Returns:
            Dict[str, Any]: Detected situations and their details
        """
        if not self.initialized:
            raise RuntimeError("Situation detector not initialized")

        # Get HUD information
        hud_info = self.extract_hud_info(frame)

        # Detect motion using hardware-aware detector
        motion_result = self.motion_detector.detect_motion(frame, method="hybrid")

        # Extract motion features
        motion_features = self._extract_motion_features(motion_result, frame.shape)

        # Detect situations based on motion and HUD info
        situations = self._analyze_situations(
            motion_features, hud_info, frame_number, fps
        )

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
            "metadata": {
                "motion_score": float(motion_result["score"]),
                "hardware_tier": self.optimizer.get_performance_tier(),
                "analysis_version": "1.0.0",
            },
        }

    def _extract_motion_features(
        self, motion_result: Dict[str, Any], frame_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Extract meaningful features from motion detection results."""
        features = {
            "overall_motion": motion_result["score"],
            "motion_regions": [],
            "motion_patterns": [],
        }

        if not motion_result["motion_detected"]:
            return features

        height, width = frame_shape[:2]

        # Analyze each motion region
        for roi in self._contours_to_roi(motion_result["contours"]):
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

        # Analyze motion patterns if we have history
        if len(self.motion_detector.motion_history) >= 2:
            features["motion_patterns"] = self._analyze_motion_patterns(
                self.motion_detector.motion_history[-2:], frame_shape
            )

        return features

    def _determine_field_region(self, rel_x: float, rel_y: float) -> str:
        """Determine which region of the field a point belongs to."""
        if rel_y < 0.4:
            return "backfield"
        elif rel_y < 0.6:
            return "line_of_scrimmage"
        else:
            return "defensive_secondary"

    def _analyze_motion_patterns(
        self, motion_history: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """Analyze motion patterns from recent history."""
        patterns = []

        # Skip if not enough history
        if len(motion_history) < 2:
            return patterns

        # Get motion centroids from history
        centroids = []
        for entry in motion_history:
            if entry["contours"]:
                moments = cv2.moments(entry["contours"][0])
                if moments["m00"] != 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    centroids.append((cx, cy))

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
        motion_features: Dict[str, Any],
        hud_info: Dict[str, Any],
        frame_number: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """Analyze and classify situations based on motion features and HUD info."""
        situations = []
        timestamp = frame_number / fps

        # Check for high motion events
        if motion_features["overall_motion"] > 0.5:
            # Analyze motion patterns
            for pattern in motion_features["motion_patterns"]:
                if pattern["type"] == "linear_motion":
                    # Detect running plays
                    if pattern["direction"] in ["right", "left"]:
                        confidence = min(motion_features["overall_motion"] * 1.2, 0.95)
                        if confidence > self.min_situation_confidence:
                            situations.append(
                                {
                                    "type": "running_play",
                                    "confidence": confidence,
                                    "frame": frame_number,
                                    "timestamp": timestamp,
                                    "details": {
                                        "direction": pattern["direction"],
                                        "speed": pattern["speed"],
                                        "motion_score": motion_features[
                                            "overall_motion"
                                        ],
                                    },
                                }
                            )

            # Analyze motion regions
            for region in motion_features["motion_regions"]:
                # Detect passing plays
                if region["region"] == "backfield" and region["relative_area"] > 0.1:
                    confidence = min(motion_features["overall_motion"] * 1.1, 0.95)
                    if confidence > self.min_situation_confidence:
                        situations.append(
                            {
                                "type": "passing_play",
                                "confidence": confidence,
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "details": {
                                    "region": region["region"],
                                    "motion_score": motion_features["overall_motion"],
                                },
                            }
                        )

        return situations

    def analyze_sequence(
        self,
        frames: List[np.ndarray],
        start_frame: int,
        fps: float,
        window_size: int = 5,
    ) -> List[Dict[str, Any]]:
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
        self, window: List[Dict[str, Any]], fps: float
    ) -> List[Dict[str, Any]]:
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
                        },
                    }
                )

        return patterns

    def extract_hud_info(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract HUD information from a frame using YOLO11-based detection."""
        if not self.initialized:
            raise RuntimeError("Situation detector not initialized")

        # Use HUD detector to get game state
        hud_info = self.hud_detector.get_game_state(frame)

        return hud_info

    def detect_mistakes(
        self, situations: List[Dict[str, Any]], hud_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect potential mistakes in gameplay.

        Args:
            situations: List of detected situations
            hud_info: HUD information

        Returns:
            List[Dict[str, Any]]: Detected mistakes
        """
        # TODO: Implement mistake detection
        # - Analyze player positioning
        # - Check for missed opportunities
        # - Detect turnovers
        return []

"""
Ball tracking module for detecting and tracking the football in gameplay clips.

This module provides specialized functionality for detecting and tracking the football
using a combination of traditional computer vision techniques and deep learning methods.
The module is hardware-aware and selects the best available methods based on system
capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from ..utils.tracking_hardware import TrackingHardwareManager, TrackingMode
from .object_tracker import ObjectTracker

logger = logging.getLogger(__name__)


class BallTracker:
    """
    Specialized tracker for detecting and tracking the football in gameplay clips.

    This class combines multiple detection and tracking methods to maintain accurate
    ball tracking throughout the gameplay, handling occlusions and rapid movements.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        tracker_type: str = "CSRT",
        min_ball_size: Tuple[int, int] = (10, 10),
        max_ball_size: Tuple[int, int] = (50, 50),
    ):
        """
        Initialize the ball tracker.

        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            tracker_type: Type of tracker to use ('CSRT', 'KCF', etc.)
            min_ball_size: Minimum expected ball size (width, height)
            max_ball_size: Maximum expected ball size (width, height)
        """
        self.confidence_threshold = confidence_threshold
        self.min_ball_size = min_ball_size
        self.max_ball_size = max_ball_size

        # Initialize hardware manager
        self.hardware_manager = TrackingHardwareManager()
        self.tracking_mode = self.hardware_manager.tracking_mode

        # Initialize object tracker
        self.tracker = ObjectTracker(tracker_type)
        self.tracking_active = False
        self.lost_frames = 0
        self.max_lost_frames = 10

        # Initialize detection models based on hardware capabilities
        self.models = {}
        self._initialize_models()

        # Motion-based prediction
        self.prev_positions = []
        self.max_positions_history = 5

        # Initialize Kalman filter for motion prediction
        self.kalman = cv2.KalmanFilter(
            4, 2
        )  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = (
            np.array(
                [[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-2, 0], [0, 0, 0, 1e-2]],
                np.float32,
            )
            * 0.03
        )
        self.kalman_initialized = False

    def _initialize_models(self):
        """Initialize detection models based on hardware capabilities."""
        try:
            # Initialize YOLOv5 if hardware supports it
            if self.tracking_mode in [TrackingMode.ADVANCED, TrackingMode.PROFESSIONAL]:
                import torch.hub

                self.models["yolo"] = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path="models/football_detection.pt",  # Custom trained model for football detection
                )
                if torch.cuda.is_available():
                    self.models["yolo"] = self.models["yolo"].cuda()
                logger.info("Initialized YOLOv5 model for ball detection")
        except Exception as e:
            logger.warning(f"Could not initialize YOLOv5: {e}")

        # Initialize traditional detection methods
        self.models["hough"] = cv2.HoughCircles
        self.models["contour"] = None  # Will use contour detection directly

    def detect_ball(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect the ball in a frame using available detection methods.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Tuple of (x, y, w, h) coordinates if ball detected, None otherwise
        """
        if (
            self.tracking_mode in [TrackingMode.ADVANCED, TrackingMode.PROFESSIONAL]
            and "yolo" in self.models
        ):
            # Try deep learning detection first
            detection = self._detect_ball_yolo(frame)
            if detection is not None:
                return detection

        # Fall back to traditional methods
        detection = self._detect_ball_traditional(frame)
        return detection

    def _detect_ball_yolo(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Detect ball using YOLOv5 model."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get predictions
        results = self.models["yolo"](rgb_frame)

        # Process results
        for *box, conf, cls in results.xyxy[0]:
            if (
                conf > self.confidence_threshold and cls == 0
            ):  # Assuming 0 is football class
                if torch.cuda.is_available():
                    box = [b.cpu().numpy() for b in box]
                else:
                    box = [b.numpy() for b in box]
                x1, y1, x2, y2 = box
                return (x1, y1, x2 - x1, y2 - y1)
        return None

    def _detect_ball_traditional(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Detect ball using traditional computer vision methods."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Try Hough Circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=int(min(self.min_ball_size) // 2),
            maxRadius=int(max(self.max_ball_size) // 2),
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            return (x - r, y - r, 2 * r, 2 * r)

        # Try contour detection
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (
                self.min_ball_size[0] <= w <= self.max_ball_size[0]
                and self.min_ball_size[1] <= h <= self.max_ball_size[1]
            ):
                # Check if contour is approximately circular
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:  # Threshold for circularity
                        return (x, y, w, h)

        return None

    def predict_next_position(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Predict the next ball position based on motion history using Kalman filtering.

        Returns:
            Tuple of (x, y, w, h) coordinates if prediction possible, None otherwise
        """
        if not self.kalman_initialized or len(self.prev_positions) < 2:
            return None

        # Predict next state
        prediction = self.kalman.predict()

        # Get the last known size
        last_pos = self.prev_positions[-1]
        w, h = last_pos[2], last_pos[3]

        # Return predicted position with last known size
        return (float(prediction[0]), float(prediction[1]), w, h)

    def update(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
        """
        Update ball tracking with a new frame.

        Args:
            frame: New video frame

        Returns:
            Tuple of (tracking_successful, bbox), where bbox is (x, y, w, h) if successful
        """
        if self.tracking_active:
            # Try to update existing tracker
            ok, bbox = self.tracker.update(frame)

            if ok:
                self.lost_frames = 0
                self._update_position_history(bbox)

                # Update Kalman filter with new measurement
                if not self.kalman_initialized:
                    self.kalman.statePre = np.array(
                        [[bbox[0]], [bbox[1]], [0], [0]], np.float32
                    )
                    self.kalman_initialized = True
                else:
                    measurement = np.array([[bbox[0]], [bbox[1]]], np.float32)
                    self.kalman.correct(measurement)

                return True, bbox
            else:
                self.lost_frames += 1

        # If tracking is lost or not active
        if self.lost_frames >= self.max_lost_frames:
            self.tracking_active = False
            self.kalman_initialized = False

        # Try to detect the ball
        predicted_bbox = self.predict_next_position() if self.tracking_active else None
        detection = self.detect_ball(frame)

        if detection is not None:
            # Ball detected, initialize or reinitialize tracker
            self.tracker.reset()
            ok = self.tracker.init(frame, detection)
            if ok:
                self.tracking_active = True
                self.lost_frames = 0
                self._update_position_history(detection)

                # Initialize or update Kalman filter
                if not self.kalman_initialized:
                    self.kalman.statePre = np.array(
                        [[detection[0]], [detection[1]], [0], [0]], np.float32
                    )
                    self.kalman_initialized = True
                else:
                    measurement = np.array([[detection[0]], [detection[1]]], np.float32)
                    self.kalman.correct(measurement)

                return True, detection
        elif predicted_bbox is not None:
            # Use predicted position to help reacquire tracking
            self.tracker.reset()
            ok = self.tracker.init(frame, predicted_bbox)
            if ok:
                self.tracking_active = True
                self.lost_frames = 0
                return True, predicted_bbox

        return False, None

    def _update_position_history(self, bbox: Tuple[float, float, float, float]):
        """Update position history for motion prediction."""
        self.prev_positions.append(bbox)
        if len(self.prev_positions) > self.max_positions_history:
            self.prev_positions.pop(0)

    def draw_ball(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw the ball's bounding box and trajectory on the frame.

        Args:
            frame: Input frame to draw on
            bbox: Current ball bounding box (x, y, w, h)
            color: Color to draw the box and trajectory (B, G, R)
            thickness: Line thickness

        Returns:
            Frame with ball visualization
        """
        output = frame.copy()

        # Draw current bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        # Draw trajectory
        if len(self.prev_positions) > 1:
            points = [
                (int(x + w / 2), int(y + h / 2)) for x, y, w, h in self.prev_positions
            ]
            for i in range(1, len(points)):
                cv2.line(output, points[i - 1], points[i], color, thickness)

        return output

    def get_ball_stats(self) -> Dict[str, Any]:
        """
        Get current ball tracking statistics.

        Returns:
            Dictionary containing tracking status, motion statistics, and prediction confidence.
            Keys include:
            - tracking_active: Whether ball is currently being tracked
            - lost_frames: Number of frames since last detection
            - position_history: Number of tracked positions in history
            - speed: Current ball speed (pixels/frame)
            - direction: Movement direction in degrees
            - in_motion: Whether ball is moving
            - acceleration: Current acceleration (pixels/frame²)
            - prediction_confidence: Confidence in motion prediction (0-1)
            - trajectory_stability: How stable the ball's path is (0-1)
        """
        stats = {
            "tracking_active": self.tracking_active,
            "lost_frames": self.lost_frames,
            "position_history": len(self.prev_positions),
        }

        if len(self.prev_positions) >= 2:
            # Calculate current motion statistics
            last_pos = self.prev_positions[-1]
            prev_pos = self.prev_positions[-2]

            dx = last_pos[0] - prev_pos[0]
            dy = last_pos[1] - prev_pos[1]

            speed = np.sqrt(dx * dx + dy * dy)
            angle = np.degrees(np.arctan2(dy, dx))

            # Calculate acceleration if we have enough history
            acceleration = 0.0
            if len(self.prev_positions) >= 3:
                prev_prev_pos = self.prev_positions[-3]
                prev_dx = prev_pos[0] - prev_prev_pos[0]
                prev_dy = prev_pos[1] - prev_prev_pos[1]
                prev_speed = np.sqrt(prev_dx * prev_dx + prev_dy * prev_dy)
                acceleration = speed - prev_speed

            # Calculate trajectory stability
            trajectory_stability = 1.0
            if len(self.prev_positions) >= 3:
                angles = []
                for i in range(len(self.prev_positions) - 2):
                    p1 = self.prev_positions[i]
                    p2 = self.prev_positions[i + 1]
                    p3 = self.prev_positions[i + 2]
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    v1_mag = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
                    v2_mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
                    if v1_mag * v2_mag != 0:
                        cos_angle = dot_product / (v1_mag * v2_mag)
                        cos_angle = min(1.0, max(-1.0, cos_angle))  # Clamp to [-1, 1]
                        angles.append(abs(np.degrees(np.arccos(cos_angle))))

                if angles:
                    # More consistent angles (closer to 180°) indicate a stable trajectory
                    avg_angle_diff = np.mean([abs(180 - angle) for angle in angles])
                    trajectory_stability = max(0.0, 1.0 - (avg_angle_diff / 180.0))

            # Calculate prediction confidence based on tracking status and stability
            prediction_confidence = 0.0
            if self.tracking_active:
                base_confidence = max(
                    0.0, 1.0 - (self.lost_frames / self.max_lost_frames)
                )
                prediction_confidence = base_confidence * trajectory_stability

            stats.update(
                {
                    "speed": float(speed),
                    "direction": float(angle),
                    "in_motion": speed > 1.0,
                    "acceleration": float(acceleration),
                    "prediction_confidence": float(prediction_confidence),
                    "trajectory_stability": float(trajectory_stability),
                }
            )

        return stats

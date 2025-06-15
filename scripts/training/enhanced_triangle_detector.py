"""
Enhanced triangle detection with proper error handling and validation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import ValidationError

from .triangle_detection_config import TriangleDetectionConfig
from .triangle_detection_logger import setup_logger


class TriangleOrientation(str, Enum):
    """Triangle orientation enumeration."""

    UP = "up"  # ▲ = in opponent's territory
    DOWN = "down"  # ▼ = in own territory


class TriangleType(str, Enum):
    """Triangle type enumeration."""

    POSSESSION = "possession"  # Shows which team has the ball
    TERRITORY = "territory"  # Shows field position context


@dataclass
class TriangleDetection:
    """Data class for triangle detection results."""

    type: TriangleType
    orientation: Optional[TriangleOrientation]  # Only used for territory triangles
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class EnhancedTriangleDetector:
    """Enhanced triangle detector with proper error handling and validation."""

    def __init__(self, config_path: Optional[Path] = None, log_file: Optional[Path] = None):
        """
        Initialize the triangle detector.

        Args:
            config_path: Optional path to config file
            log_file: Optional path to log file
        """
        # Set up logging
        self.logger = setup_logger("triangle_detector", log_file)

        # Load config
        try:
            self.config = (
                TriangleDetectionConfig.parse_file(config_path)
                if config_path
                else TriangleDetectionConfig()
            )
        except ValidationError as e:
            self.logger.error(f"Config validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

        self.logger.info("Triangle detector initialized successfully")

    def _validate_image(self, image: np.ndarray) -> None:
        """
        Validate input image.

        Args:
            image: Input image

        Raises:
            ValueError: If image is invalid
        """
        if image is None:
            raise ValueError("Image cannot be None")
        if len(image.shape) != 3:
            raise ValueError("Image must be 3-channel")
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8")

    def _detect_orientation(self, contour: np.ndarray, image_height: int) -> TriangleOrientation:
        """
        Detect triangle orientation.

        Args:
            contour: Triangle contour points
            image_height: Height of the image

        Returns:
            Triangle orientation (UP/DOWN)
        """
        try:
            # Get topmost and bottommost points
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

            # Get the average x-coordinate of the base
            base_points = contour[
                np.abs(contour[:, :, 1] - bottommost[1]) < self.config.BASE_POINT_TOLERANCE
            ]
            base_x = np.mean(base_points[:, :, 0])

            # Triangle points up if the top point is between base points
            x_min = base_x - self.config.ORIENTATION_X_TOLERANCE
            x_max = base_x + self.config.ORIENTATION_X_TOLERANCE

            if x_min <= topmost[0] <= x_max:
                return TriangleOrientation.UP
            return TriangleOrientation.DOWN

        except Exception as e:
            self.logger.error(f"Failed to detect orientation: {e}")
            raise

    def detect_triangles(
        self, image: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None
    ) -> list[TriangleDetection]:
        """
        Detect triangles in image.

        Args:
            image: Input image
            roi: Optional region of interest (x1, y1, x2, y2)

        Returns:
            List of triangle detections

        Raises:
            ValueError: If input validation fails
        """
        try:
            # Validate input
            self._validate_image(image)

            # Extract ROI if provided
            if roi:
                x1, y1, x2, y2 = roi
                image = image[y1:y2, x1:x2]

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, self.config.THRESHOLD, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for contour in contours:
                # Approximate contour
                epsilon = self.config.EPSILON_FACTOR * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if triangle (3 points)
                if len(approx) == 3:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(approx)
                    bbox = (x, y, x + w, y + h)

                    # Determine triangle type based on position
                    if x < image.shape[1] // 2:
                        # Left side = possession triangle
                        detection = TriangleDetection(
                            type=TriangleType.POSSESSION,
                            orientation=None,  # Possession triangles don't have orientation
                            bbox=bbox,
                            confidence=0.95,  # TODO: Calculate actual confidence
                        )
                    else:
                        # Right side = territory triangle
                        orientation = self._detect_orientation(approx, image.shape[0])
                        detection = TriangleDetection(
                            type=TriangleType.TERRITORY,
                            orientation=orientation,
                            bbox=bbox,
                            confidence=0.95,  # TODO: Calculate actual confidence
                        )

                    detections.append(detection)

            self.logger.info(f"Detected {len(detections)} triangles")
            return detections

        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise

    def determine_field_position(
        self,
        possession_detection: Optional[TriangleDetection],
        territory_detection: Optional[TriangleDetection],
    ) -> dict[str, str]:
        """
        Determine complete field position context.

        Args:
            possession_detection: Possession triangle detection
            territory_detection: Territory triangle detection

        Returns:
            Dict containing:
                - possession_team: "home" or "away"
                - in_territory: "own" or "opponent"
                - on_offense: True/False
        """
        try:
            if not possession_detection or not territory_detection:
                raise ValueError("Both possession and territory detections required")

            # Determine possession
            x_pos = possession_detection.bbox[0]
            possession_team = "away" if x_pos < 100 else "home"

            # Determine territory
            in_territory = (
                "opponent" if territory_detection.orientation == TriangleOrientation.UP else "own"
            )

            # Determine if on offense
            on_offense = (
                possession_team == "away"
                if in_territory == "opponent"
                else possession_team == "home"
            )

            return {
                "possession_team": possession_team,
                "in_territory": in_territory,
                "on_offense": on_offense,
            }

        except Exception as e:
            self.logger.error(f"Failed to determine field position: {e}")
            raise

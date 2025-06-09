"""Game detection and interface mapping for multi-game support."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .hardware import HardwareDetector

logger = logging.getLogger(__name__)


class GameVersion(Enum):
    """Supported game versions."""

    MADDEN_25 = "madden_25"
    CFB_25 = "cfb_25"


@dataclass
class GameProfile:
    """Game-specific interface and feature information."""

    version: GameVersion
    hud_layout: dict[str, dict]  # Regions for different HUD elements
    supported_features: list[str]  # List of supported features
    interface_version: str  # UI version identifier


class GameDetectionError(Exception):
    """Base exception for game detection errors."""

    pass


class UnsupportedGameError(GameDetectionError):
    """Raised when an unsupported game is detected."""

    pass


class InvalidFrameError(GameDetectionError):
    """Raised when a frame cannot be processed."""

    pass


class GameDetector:
    """
    Detects and manages game version information and interface mapping.
    Provides functionality to identify the game being played and adapt
    the analysis pipeline accordingly.
    """

    def __init__(self):
        """Initialize the game detector."""
        self.hardware = HardwareDetector()
        self._current_game: Optional[GameVersion] = None
        self._confidence_threshold = 0.8  # Minimum confidence for game detection
        self._frame_buffer_size = 5  # Number of frames to buffer for stable detection
        self._frame_buffer = []  # Buffer of recent detections
        self._game_profiles: dict[GameVersion, GameProfile] = self._initialize_game_profiles()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the GameDetector."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _initialize_game_profiles(self) -> dict[GameVersion, GameProfile]:
        """Initialize game profiles with interface mappings."""
        return {
            GameVersion.MADDEN_25: GameProfile(
                version=GameVersion.MADDEN_25,
                hud_layout={
                    "score_bug": {
                        "region": (50, 50, 300, 100),  # Example coordinates
                        "elements": {
                            "score": (10, 10, 50, 30),
                            "time": (60, 10, 120, 30),
                            "down": (130, 10, 180, 30),
                            "distance": (190, 10, 240, 30),
                        },
                    },
                    "play_art": {
                        "region": (0, 100, 1280, 620),  # Example coordinates
                        "elements": {
                            "offensive_formation": (50, 50, 200, 100),
                            "defensive_formation": (50, 150, 200, 200),
                        },
                    },
                },
                supported_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                ],
                interface_version="m25_1.0",
            ),
            GameVersion.CFB_25: GameProfile(
                version=GameVersion.CFB_25,
                hud_layout={
                    "score_bug": {
                        "region": (40, 40, 280, 90),  # Example coordinates
                        "elements": {
                            "score": (8, 8, 48, 28),
                            "time": (58, 8, 118, 28),
                            "down": (128, 8, 178, 28),
                            "distance": (188, 8, 238, 28),
                        },
                    },
                    "play_art": {
                        "region": (0, 90, 1280, 610),  # Example coordinates
                        "elements": {
                            "offensive_formation": (45, 45, 195, 95),
                            "defensive_formation": (45, 145, 195, 195),
                        },
                    },
                },
                supported_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                ],
                interface_version="cfb25_1.0",
            ),
        }

    def detect_game(self, frame: np.ndarray) -> GameVersion:
        """
        Detect the game version from a video frame.

        Args:
            frame: Video frame as numpy array

        Returns:
            Detected GameVersion

        Raises:
            InvalidFrameError: If the frame cannot be processed
            UnsupportedGameError: If the game cannot be identified
        """
        try:
            if frame is None or frame.size == 0:
                raise InvalidFrameError("Empty or invalid frame provided")

            self.logger.debug("Processing frame for game detection")

            # Convert frame to PIL Image for processing
            try:
                frame_image = Image.fromarray(frame)
            except Exception as e:
                raise InvalidFrameError(f"Failed to convert frame to PIL Image: {e}")

            # Check each game's characteristics
            confidence_scores = {}

            for version in GameVersion:
                try:
                    score = self._calculate_game_confidence(frame_image, version)
                    confidence_scores[version] = score
                    self.logger.debug(f"Confidence score for {version}: {score:.2f}")
                except Exception as e:
                    self.logger.warning(f"Error calculating confidence for {version}: {e}")
                    confidence_scores[version] = 0.0

            # Get the most likely game version
            best_match = max(confidence_scores.items(), key=lambda x: x[1])
            version, confidence = best_match

            # Update detection buffer
            self._frame_buffer.append(version)
            if len(self._frame_buffer) > self._frame_buffer_size:
                self._frame_buffer.pop(0)

            # Get the most common version in the buffer
            from collections import Counter

            buffer_counts = Counter(self._frame_buffer)
            stable_version = buffer_counts.most_common(1)[0][0]
            stable_confidence = buffer_counts[stable_version] / len(self._frame_buffer)

            if stable_confidence >= self._confidence_threshold:
                if self._current_game != stable_version:
                    self.logger.info(f"Game version detected: {stable_version}")
                    self._current_game = stable_version
                return stable_version
            else:
                raise UnsupportedGameError(
                    f"Unable to confidently detect game version. "
                    f"Best match: {version} with confidence {confidence:.2f}"
                )

        except GameDetectionError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in game detection: {e}")
            raise GameDetectionError(f"Game detection failed: {e}")

    def _calculate_game_confidence(self, image: Image.Image, version: GameVersion) -> float:
        """
        Calculate confidence score for a game version based on image analysis.

        Args:
            image: PIL Image to analyze
            version: GameVersion to check against

        Returns:
            Confidence score (0-1)
        """
        profile = self._game_profiles[version]
        return self._analyze_interface_elements(image, profile)

    def _analyze_interface_elements(self, image: Image.Image, profile: GameProfile) -> float:
        """
        Analyze interface elements to determine game version confidence.

        Args:
            image: PIL Image to analyze
            profile: GameProfile to check against

        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        total_checks = 0

        # Check score bug position and layout
        score_bug = profile.hud_layout["score_bug"]
        region = score_bug["region"]
        score_bug_image = image.crop(region)

        # Add points for expected HUD element positions
        for element_name, element_coords in score_bug["elements"].items():
            total_checks += 1
            element_region = image.crop(element_coords)
            # TODO: Implement more sophisticated element detection
            # For now, use basic checks like aspect ratio and color distribution
            if self._check_element_characteristics(element_region, element_name):
                score += 1.0

        # Check play art region characteristics
        play_art = profile.hud_layout["play_art"]
        play_art_image = image.crop(play_art["region"])
        total_checks += 1
        if self._check_play_art_characteristics(play_art_image):
            score += 1.0

        # Normalize score
        return score / total_checks if total_checks > 0 else 0.0

    def _check_element_characteristics(
        self, element_image: Image.Image, element_type: str
    ) -> float:
        """
        Check if an image region matches expected characteristics for a HUD element.
        Uses OCR and pattern matching for more accurate detection.

        Args:
            element_image: Cropped PIL Image of the element
            element_type: Type of HUD element being checked ('score', 'time', 'down', 'distance')

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Convert to numpy array for OpenCV processing
            element_array = np.array(element_image)
            if not element_array.any():
                return 0.0

            # Convert to grayscale
            if len(element_array.shape) == 3:
                element_gray = cv2.cvtColor(element_array, cv2.COLOR_RGB2GRAY)
            else:
                element_gray = element_array

            # Apply adaptive thresholding for better text extraction
            binary = cv2.adaptiveThreshold(
                element_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            # Calculate confidence based on element type
            confidence = 0.0

            if element_type == "score":
                # Expect digits 0-9 with clear contrast
                confidence = self._check_score_element(binary)
            elif element_type == "time":
                # Expect time format (MM:SS)
                confidence = self._check_time_element(binary)
            elif element_type == "down":
                # Expect 1st, 2nd, 3rd, 4th
                confidence = self._check_down_element(binary)
            elif element_type == "distance":
                # Expect number + "yd" or "TO GO"
                confidence = self._check_distance_element(binary)

            self.logger.debug(f"Element {element_type} confidence: {confidence:.2f}")
            return confidence

        except Exception as e:
            self.logger.warning(f"Error checking element characteristics: {e}")
            return 0.0

    def _check_score_element(self, binary_image: np.ndarray) -> float:
        """Check characteristics specific to score elements."""
        # Count connected components (should be 1-2 digits)
        num_labels, labels = cv2.connectedComponents(binary_image)
        if not 2 <= num_labels <= 4:  # 1-2 digits + potential background
            return 0.0

        # Check aspect ratio (should be roughly square for each digit)
        height, width = binary_image.shape
        if not 0.8 <= width / height <= 2.5:
            return 0.0

        # Check pixel density in the region
        density = np.sum(binary_image > 0) / (height * width)
        if not 0.2 <= density <= 0.6:  # Typical range for digits
            return 0.0

        return 1.0

    def _check_time_element(self, binary_image: np.ndarray) -> float:
        """Check characteristics specific to time display elements."""
        height, width = binary_image.shape

        # Check for colon in middle section
        mid_region = binary_image[:, width // 3 : 2 * width // 3]
        has_colon = np.sum(mid_region > 0) > 0

        if not has_colon:
            return 0.0

        # Check for two number groups (minutes and seconds)
        left_region = binary_image[:, : width // 3]
        right_region = binary_image[:, 2 * width // 3 :]

        left_density = np.sum(left_region > 0) / (height * width // 3)
        right_density = np.sum(right_region > 0) / (height * width // 3)

        if not (0.1 <= left_density <= 0.5 and 0.1 <= right_density <= 0.5):
            return 0.0

        return 1.0

    def _check_down_element(self, binary_image: np.ndarray) -> float:
        """Check characteristics specific to down indicators."""
        # Look for typical down text patterns (1st, 2nd, 3rd, 4th)
        height, width = binary_image.shape

        # Should have 2-3 connected components (number + "st"/"nd"/"rd"/"th")
        num_labels, _ = cv2.connectedComponents(binary_image)
        if not 2 <= num_labels <= 4:
            return 0.0

        # Check aspect ratio (should be wider than tall)
        if not 1.5 <= width / height <= 3.0:
            return 0.0

        return 1.0

    def _check_distance_element(self, binary_image: np.ndarray) -> float:
        """Check characteristics specific to distance indicators."""
        # Look for number followed by "yd" or "TO GO"
        height, width = binary_image.shape

        # Should have multiple connected components
        num_labels, _ = cv2.connectedComponents(binary_image)
        if not 3 <= num_labels <= 5:
            return 0.0

        # Check aspect ratio (should be wider than tall)
        if not 2.0 <= width / height <= 4.0:
            return 0.0

        return 1.0

    def _check_play_art_characteristics(self, play_art_image: Image.Image) -> float:
        """
        Check if an image region matches expected characteristics for play art.
        Uses edge detection and line analysis for more accurate detection.

        Args:
            play_art_image: Cropped PIL Image of the play art region

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Convert to numpy array for OpenCV processing
            play_art_array = np.array(play_art_image)
            if not play_art_array.any():
                return 0.0

            # Convert to grayscale
            if len(play_art_array.shape) == 3:
                play_art_gray = cv2.cvtColor(play_art_array, cv2.COLOR_RGB2GRAY)
            else:
                play_art_gray = play_art_array

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(play_art_gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10,
            )

            if lines is None:
                return 0.0

            # Analyze line characteristics
            confidence = self._analyze_play_art_lines(lines)

            # Check for player markers
            player_confidence = self._check_player_markers(play_art_gray)

            # Combine confidences
            final_confidence = (confidence + player_confidence) / 2

            self.logger.debug(f"Play art confidence: {final_confidence:.2f}")
            return final_confidence

        except Exception as e:
            self.logger.warning(f"Error checking play art characteristics: {e}")
            return 0.0

    def _analyze_play_art_lines(self, lines: np.ndarray) -> float:
        """
        Analyze detected lines for play art characteristics.

        Args:
            lines: Array of detected lines from HoughLinesP

        Returns:
            Confidence score between 0 and 1
        """
        if len(lines) < 3:  # Need minimum number of lines for play art
            return 0.0

        # Calculate line angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            angles.append(angle)

        # Count horizontal (0°±15°), vertical (90°±15°), and diagonal (45°±15°) lines
        horizontal_count = sum(1 for a in angles if (a <= 15 or a >= 165))
        vertical_count = sum(1 for a in angles if 75 <= a <= 105)
        diagonal_count = sum(1 for a in angles if (30 <= a <= 60 or 120 <= a <= 150))

        # Play art typically has a mix of line orientations
        total_lines = len(lines)
        if total_lines < 5:
            return 0.3
        elif total_lines < 10:
            return 0.6
        else:
            return 1.0

    def _check_player_markers(self, gray_image: np.ndarray) -> float:
        """
        Check for circular player markers in the play art.

        Args:
            gray_image: Grayscale image to check

        Returns:
            Confidence score between 0 and 1
        """
        # Apply threshold to isolate potential markers
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Count circular contours
        circular_count = 0
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            # Circularity = 4*pi*area/perimeter^2
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Circles have circularity close to 1
            if 0.8 <= circularity <= 1.2:
                circular_count += 1

        # Return confidence based on number of circular markers found
        if circular_count >= 11:  # Full offensive/defensive formation
            return 1.0
        elif circular_count >= 5:  # Partial formation
            return 0.7
        elif circular_count >= 2:  # Some players visible
            return 0.4
        else:
            return 0.0

    def get_interface_mapping(self, game_version: Optional[GameVersion] = None) -> GameProfile:
        """
        Get the interface mapping for a specific game version.

        Args:
            game_version: GameVersion to get mapping for (uses current if None)

        Returns:
            GameProfile with interface mapping information

        Raises:
            ValueError: If no game version is specified and no current version is set
        """
        version = game_version or self._current_game
        if not version:
            raise ValueError("No game version specified and no current version detected")
        return self._game_profiles[version]

    def is_feature_supported(
        self, feature: str, game_version: Optional[GameVersion] = None
    ) -> bool:
        """
        Check if a feature is supported for a specific game version.

        Args:
            feature: Feature name to check
            game_version: GameVersion to check (uses current if None)

        Returns:
            True if the feature is supported
        """
        profile = self.get_interface_mapping(game_version)
        return feature in profile.supported_features

    @property
    def current_game(self) -> Optional[GameVersion]:
        """Get the currently detected game version."""
        return self._current_game

    @property
    def supported_versions(self) -> list[GameVersion]:
        """Get list of supported game versions."""
        return list(GameVersion)

    def _detect_game_version(self, score_bug_roi: np.ndarray) -> str:
        """
        Detect the game version from the score bug ROI.

        Args:
            score_bug_roi: Region of interest containing the score bug

        Returns:
            Game version string (e.g., "madden_25", "cfb_25")
        """
        # Use template matching to identify the game version
        max_similarity = 0
        detected_version = None

        for version, templates in self.score_bug_templates.items():
            for template in templates:
                similarity = cv2.matchTemplate(score_bug_roi, template, cv2.TM_CCOEFF_NORMED).max()

                if similarity > max_similarity:
                    max_similarity = similarity
                    detected_version = version

        return detected_version if max_similarity > self.similarity_threshold else "unknown"

    def _detect_game_state(self, score_bug_roi: np.ndarray) -> dict[str, Any]:
        """
        Detect the game state from the score bug ROI.

        Args:
            score_bug_roi: Region of interest containing the score bug

        Returns:
            Dictionary containing game state information
        """
        # Extract text from score bug using OCR
        text = self._extract_text(score_bug_roi)

        # Parse game state from text
        state = {
            "quarter": self._parse_quarter(text),
            "time": self._parse_time(text),
            "score": self._parse_score(text),
            "down_distance": self._parse_down_distance(text),
        }

        return state

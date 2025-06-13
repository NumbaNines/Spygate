"""
Game situation detector using YOLOv8 and OCR for football game analysis.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from . import Detection
from ...spygate.ml.yolov8_model import EnhancedYOLOv8

logger = logging.getLogger(__name__)


@dataclass
class GameSituation:
    """Represents a detected game situation."""

    down: int
    distance: int
    yard_line: int
    quarter: int
    time_remaining: str
    score_home: int
    score_away: int
    possession: str  # 'home' or 'away'
    confidence: float


class SituationDetector:
    """Detects and analyzes game situations using YOLOv8 and OCR."""

    def __init__(
        self,
        yolo_weights: Optional[Path] = None,
        tesseract_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the situation detector.

        Args:
            yolo_weights: Path to YOLOv8 weights
            tesseract_path: Path to Tesseract executable
            device: Device to run on ('cuda' or 'cpu')
        """
        # Initialize YOLOv8 detector
        model_path = str(yolo_weights) if yolo_weights else "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
        self.detector = EnhancedYOLOv8(model_path=model_path, device=device)

        # Configure Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Cache for temporal consistency
        self._last_situation: Optional[GameSituation] = None
        self._frame_count = 0

    def detect_situation(self, frame: np.ndarray) -> Optional[GameSituation]:
        """Detect the game situation from a video frame.

        Args:
            frame: Input frame as numpy array (BGR)

        Returns:
            Detected game situation or None if detection failed
        """
        # Get HUD element detections
        detections = self.detector.detect(frame)
        if not detections:
            logger.warning("No HUD elements detected")
            return self._last_situation  # Return last known situation

        # Extract text from detected regions
        situation_data = self._extract_situation_data(frame, detections)
        if not situation_data:
            logger.warning("Failed to extract situation data")
            return self._last_situation

        # Create game situation
        try:
            situation = self._create_game_situation(situation_data)

            # Update cache
            self._last_situation = situation
            self._frame_count += 1

            return situation
        except Exception as e:
            logger.error(f"Failed to create game situation: {str(e)}")
            return self._last_situation

    def _extract_situation_data(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> dict[str, str]:
        """Extract text data from detected HUD regions.

        Args:
            frame: Input frame
            detections: List of detected HUD elements

        Returns:
            Dictionary of extracted text data
        """
        data = {}

        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence'] 
            label = detection['class']
            if conf < 0.25:
                continue

            # Extract region
            x1, y1, x2, y2 = map(int, bbox)
            region = frame[y1:y2, x1:x2]

            # Process based on element type
            if label == "down_distance":
                data["down_distance"] = self._process_down_distance(region)
            elif label == "game_clock":
                data["game_clock"] = self._process_game_clock(region)
            elif label == "score_bug":
                score_data = self._process_score_bug(region)
                data.update(score_data)
            elif label == "possession_arrow":
                data["possession"] = self._process_possession(region)

        return data

    def _process_down_distance(self, region: np.ndarray) -> str:
        """Process down and distance text from region."""
        # Enhance region for OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Extract text
        text = pytesseract.image_to_string(
            Image.fromarray(thresh), config="--psm 7"  # Assume single line of text
        )
        return text.strip()

    def _process_game_clock(self, region: np.ndarray) -> str:
        """Process game clock text from region."""
        # Similar to down_distance but with different OCR config
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        text = pytesseract.image_to_string(
            Image.fromarray(thresh), config="--psm 7 -c tessedit_char_whitelist=0123456789:"
        )
        return text.strip()

    def _process_score_bug(self, region: np.ndarray) -> dict[str, str]:
        """Process score bug region for team scores."""
        # This requires more complex processing as it contains multiple elements
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # For now, just get the full text and parse later
        text = pytesseract.image_to_string(
            Image.fromarray(thresh), config="--psm 6"  # Assume uniform block of text
        )

        # TODO: Implement proper parsing of score bug text
        return {"score_home": "0", "score_away": "0"}

    def _process_possession(self, region: np.ndarray) -> str:
        """Process possession arrow region."""
        # This might be better done with template matching or simple pixel analysis
        # For now, return a placeholder
        return "home"

    def _create_game_situation(self, data: dict[str, str]) -> GameSituation:
        """Create a GameSituation object from extracted data.

        Args:
            data: Dictionary of extracted text data

        Returns:
            GameSituation object

        Raises:
            ValueError: If required data is missing or invalid
        """
        # Parse down and distance
        down_distance = data.get("down_distance", "")
        down = self._parse_down(down_distance)
        distance = self._parse_distance(down_distance)

        # Create situation object
        situation = GameSituation(
            down=down,
            distance=distance,
            yard_line=0,  # TODO: Implement yard line detection
            quarter=1,  # TODO: Implement quarter detection
            time_remaining=data.get("game_clock", "00:00"),
            score_home=int(data.get("score_home", 0)),
            score_away=int(data.get("score_away", 0)),
            possession=data.get("possession", "home"),
            confidence=0.8,  # TODO: Calculate actual confidence
        )

        return situation

    def _parse_down(self, text: str) -> int:
        """Parse down number from text."""
        # TODO: Implement proper down parsing
        return 1

    def _parse_distance(self, text: str) -> int:
        """Parse distance from text."""
        # TODO: Implement proper distance parsing
        return 10

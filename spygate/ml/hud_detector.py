"""YOLOv8-based HUD element detection for gameplay clips with OCR processing."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

logger = logging.getLogger(__name__)


class HUDDetector:
    """YOLOv8-based HUD element detector with OCR processing and hardware optimization."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the HUD detector.

        Args:
            model_path: Path to a custom YOLOv8 model. If None, will use a default model.
        """
        self.initialized = False
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = 0.6

        # OCR engine setup
        self.ocr_reader = None
        self.use_easyocr = EASYOCR_AVAILABLE
        self.use_tesseract = TESSERACT_AVAILABLE

        # Use the UI element classes from YOLOv8 model
        self.classes = UI_CLASSES

        # Track the last detected HUD region for optimization
        self.last_hud_region = None
        self.hud_detection_interval = 30  # frames
        self.frame_count = 0

    def initialize(self):
        """Initialize the YOLOv8 model and OCR engines with hardware-aware settings."""
        try:
            # Initialize YOLOv8 model with optimization
            optimization_config = OptimizationConfig(
                enable_dynamic_switching=True,
                enable_adaptive_batch_size=True,
                enable_performance_monitoring=True,
                enable_auto_optimization=True,
            )

            self.model = EnhancedYOLOv8(
                model_path=self.model_path,
                hardware=self.hardware,
                optimization_config=optimization_config,
            )

            # Initialize OCR engines
            self._initialize_ocr()

            # Configure model settings based on hardware tier
            self._configure_model_settings()

            self.initialized = True
            logger.info(
                f"HUD detector initialized on {self.model.device} with {len(self.classes)} UI element classes"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HUD detector: {e}")
            raise

    def _initialize_ocr(self):
        """Initialize OCR engines based on availability and hardware."""
        if self.use_easyocr:
            try:
                # Initialize EasyOCR with appropriate language and GPU settings
                use_gpu = (
                    self.hardware.has_cuda and self.hardware.tier.value >= 3
                )  # Medium tier or above
                self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
                logger.info(f"EasyOCR initialized with GPU: {use_gpu}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.use_easyocr = False

        if not self.use_easyocr and self.use_tesseract:
            try:
                # Test Tesseract availability
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR available as fallback")
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
                self.use_tesseract = False

    def detect_hud_elements(self, frame: np.ndarray) -> dict[str, Any]:
        """Detect HUD elements in a frame using YOLOv8.

        Args:
            frame: Input frame as numpy array

        Returns:
            Dict containing:
            - detections: List of detected elements with their locations and text
            - metadata: Detection info and confidence scores
        """
        if not self.initialized:
            raise RuntimeError("HUD detector not initialized")

        try:
            # Use the optimized YOLOv8 model for detection
            detection_results = self.model.detect_hud_elements(frame)

            # Extract detection data
            detections = []
            if detection_results.boxes.shape[0] > 0:
                for i in range(len(detection_results.boxes)):
                    box = detection_results.boxes[i]
                    score = detection_results.scores[i]
                    class_idx = int(detection_results.classes[i])
                    class_name = detection_results.class_names[class_idx]

                    if score > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box)

                        # Extract text from this HUD element
                        text = self.extract_text(frame, {"bbox": (x1, y1, x2, y2)})

                        detections.append(
                            {
                                "class": class_name,
                                "confidence": float(score),
                                "bbox": (x1, y1, x2, y2),
                                "text": text,
                            }
                        )

            return {
                "detections": detections,
                "metadata": {
                    "hardware_tier": self.hardware.tier.name,
                    "device": self.model.device,
                    "model_version": "YOLOv8",
                    "processing_time": detection_results.processing_time,
                    "frame_processed": self.frame_count,
                },
            }

        except Exception as e:
            logger.error(f"Error during HUD detection: {e}")
            return {"detections": [], "metadata": {"error": str(e)}}

    def _configure_model_settings(self):
        """Configure model settings based on hardware tier."""
        tier_name = self.hardware.tier.name.lower()

        # Adjust settings based on hardware tier
        if tier_name == "ultra_low" or tier_name == "low":
            self.confidence_threshold = 0.7  # Higher confidence to reduce false positives
            self.hud_detection_interval = 45  # Less frequent HUD detection
        elif tier_name == "medium":
            self.confidence_threshold = 0.6
            self.hud_detection_interval = 30
        else:  # high or ultra tier
            self.confidence_threshold = 0.5  # Lower confidence for more detections
            self.hud_detection_interval = 15  # More frequent HUD detection

    def extract_text(self, frame: np.ndarray, detection: dict[str, Any]) -> str:
        """Extract text from a detected HUD element using OCR.

        Args:
            frame: Input frame
            detection: Detection dictionary containing bbox

        Returns:
            Extracted text string
        """
        try:
            # Extract region of interest
            x1, y1, x2, y2 = detection["bbox"]

            # Add padding and ensure valid coordinates
            h, w = frame.shape[:2]
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return ""

            # Preprocess ROI for better OCR accuracy
            processed_roi = self._preprocess_for_ocr(roi)

            # Extract text using available OCR engine
            if self.use_easyocr and self.ocr_reader:
                return self._extract_text_easyocr(processed_roi)
            elif self.use_tesseract:
                return self._extract_text_tesseract(processed_roi)
            else:
                logger.warning("No OCR engine available")
                return ""

        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return ""

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for better OCR accuracy."""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize if too small
        height, width = gray.shape
        if height < 20 or width < 20:
            scale_factor = max(2, 20 // min(height, width))
            gray = cv2.resize(
                gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
            )

        # Apply adaptive thresholding for better contrast
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Noise reduction
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _extract_text_easyocr(self, processed_roi: np.ndarray) -> str:
        """Extract text using EasyOCR."""
        try:
            results = self.ocr_reader.readtext(processed_roi, detail=0, paragraph=False)
            text = " ".join(results).strip()
            return text
        except Exception as e:
            logger.debug(f"EasyOCR extraction failed: {e}")
            return ""

    def _extract_text_tesseract(self, processed_roi: np.ndarray) -> str:
        """Extract text using Tesseract OCR."""
        try:
            # Configure Tesseract for better HUD text recognition
            config = "--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:-& "
            text = pytesseract.image_to_string(processed_roi, config=config).strip()
            return text
        except Exception as e:
            logger.debug(f"Tesseract extraction failed: {e}")
            return ""

    def get_game_state(self, frame: np.ndarray) -> dict[str, Any]:
        """Extract complete game state from HUD elements.

        Args:
            frame: Input frame

        Returns:
            Dict containing game state information
        """
        # Detect HUD elements
        results = self.detect_hud_elements(frame)

        game_state = {
            "down": None,
            "distance": None,
            "yard_line": None,
            "score_home": None,
            "score_away": None,
            "game_clock": None,
            "play_clock": None,
            "possession": None,
            "timeouts": {"home": None, "away": None},
            "penalties": False,
            "confidence": 0.0,
            "raw_detections": results["detections"],
        }

        # Process each detection and extract game state
        total_confidence = 0.0
        detection_count = 0

        for detection in results["detections"]:
            element_type = detection["class"]
            text = detection["text"]
            confidence = detection["confidence"]

            total_confidence += confidence
            detection_count += 1

            # Parse text based on element type
            if element_type == "down_distance" and text:
                down, distance = self._parse_down_distance(text)
                if down:
                    game_state["down"] = down
                if distance:
                    game_state["distance"] = distance

            elif element_type == "score_bug" and text:
                scores = self._parse_score(text)
                if scores:
                    game_state["score_home"] = scores.get("home")
                    game_state["score_away"] = scores.get("away")

            elif element_type == "game_clock" and text:
                clock = self._parse_clock(text)
                if clock:
                    game_state["game_clock"] = clock

            elif element_type == "play_clock" and text:
                play_clock = self._parse_play_clock(text)
                if play_clock:
                    game_state["play_clock"] = play_clock

            elif element_type == "field_position" and text:
                yard_line = self._parse_field_position(text)
                if yard_line:
                    game_state["yard_line"] = yard_line

        # Calculate overall confidence
        if detection_count > 0:
            game_state["confidence"] = total_confidence / detection_count

        return game_state

    def _parse_down_distance(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse down and distance from text."""
        import re

        # Common patterns: "1st & 10", "2nd & 5", "3rd & Long", etc.
        pattern = r"(\d+)[a-zA-Z]*\s*&\s*(\d+|Long|Goal)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            down = int(match.group(1))
            distance_str = match.group(2)

            if distance_str.lower() == "long":
                distance = 10  # Assume 10+ for "long"
            elif distance_str.lower() == "goal":
                distance = 0  # Goal line
            else:
                try:
                    distance = int(distance_str)
                except ValueError:
                    distance = None

            return down, distance

        return None, None

    def _parse_score(self, text: str) -> Optional[dict]:
        """Parse team scores from text."""
        import re

        # Look for numeric patterns
        numbers = re.findall(r"\d+", text)

        if len(numbers) >= 2:
            return {"home": int(numbers[0]), "away": int(numbers[1])}

        return None

    def _parse_clock(self, text: str) -> Optional[str]:
        """Parse game clock from text."""
        import re

        # Look for time patterns: "12:34", "5:02", etc.
        pattern = r"(\d{1,2}):(\d{2})"
        match = re.search(pattern, text)

        if match:
            return f"{match.group(1)}:{match.group(2)}"

        return None

    def _parse_play_clock(self, text: str) -> Optional[int]:
        """Parse play clock from text."""
        import re

        # Look for single or double digit numbers
        numbers = re.findall(r"\d+", text)

        if numbers:
            play_clock = int(numbers[0])
            # Play clock is typically 0-40
            if 0 <= play_clock <= 40:
                return play_clock

        return None

    def _parse_field_position(self, text: str) -> Optional[str]:
        """Parse field position from text."""
        import re

        # Look for yard line patterns: "OWN 25", "OPP 40", etc.
        pattern = r"(OWN|OPP)\s*(\d+)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            side = match.group(1).upper()
            yard_line = int(match.group(2))
            return f"{side} {yard_line}"

        return None

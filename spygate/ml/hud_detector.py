"""YOLO11-based HUD element detection for gameplay clips."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .yolo11_model import UI_CLASSES  # Import the UI classes

logger = logging.getLogger(__name__)


class HUDDetector:
    """YOLO11-based HUD element detector with hardware optimization."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the HUD detector.

        Args:
            model_path: Path to a custom YOLO11 model. If None, will use a default model.
        """
        self.initialized = False
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = 0.6
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use the updated UI element classes
        self.classes = UI_CLASSES

        # Track the last detected HUD region for optimization
        self.last_hud_region = None
        self.hud_detection_interval = 30  # frames
        self.frame_count = 0

    def initialize(self):
        """Initialize the YOLO11 model with hardware-aware settings."""
        try:
            # Load model with hardware optimization
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Use a default model or raise error
                raise ValueError("No valid model path provided")

            # Configure model settings based on hardware tier
            self._configure_model_settings()

            self.initialized = True
            logger.info(
                f"HUD detector initialized on {self.device} with {len(self.classes)} UI element classes"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HUD detector: {e}")
            raise

    def detect_hud_elements(self, frame: np.ndarray) -> dict[str, Any]:
        """Detect HUD elements in a frame using a hierarchical approach.

        First detects the main HUD region, then searches for specific elements within it.

        Args:
            frame: Input frame as numpy array

        Returns:
            Dict containing:
            - hud_region: Coordinates of main HUD
            - elements: Dict of detected elements and their locations
            - metadata: Detection info and confidence scores
        """
        if not self.initialized:
            raise RuntimeError("HUD detector not initialized")

        try:
            # Step 1: Detect or use cached HUD region
            hud_region = self._get_hud_region(frame)
            if hud_region is None:
                return {"detections": [], "metadata": {"error": "No HUD region detected"}}

            # Step 2: Extract HUD region for detailed analysis
            hud_frame = self._extract_region(frame, hud_region)

            # Step 3: Detect elements within HUD region
            elements = self._detect_hud_elements(hud_frame)

            # Step 4: Adjust coordinates to original frame
            adjusted_elements = self._adjust_coordinates(elements, hud_region)

            return {
                "hud_region": hud_region,
                "detections": adjusted_elements,
                "metadata": {
                    "hardware_tier": self.optimizer.get_performance_tier(),
                    "device": self.device,
                    "model_version": "YOLO11",
                    "frame_processed": self.frame_count,
                },
            }

        except Exception as e:
            logger.error(f"Error during HUD detection: {e}")
            return {"detections": [], "metadata": {"error": str(e)}}

    def _get_hud_region(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Get the HUD region, either from cache or by detection.

        Args:
            frame: Input frame

        Returns:
            Tuple of (x1, y1, x2, y2) for HUD region or None if not found
        """
        # Only detect HUD region periodically to save processing
        if (self.frame_count % self.hud_detection_interval == 0) or (self.last_hud_region is None):
            # Run detection focusing only on HUD class
            results = self.model(frame, verbose=False)[0]

            for box in results.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                if class_id == self.classes["hud"] and confidence > self.confidence_threshold:
                    self.last_hud_region = tuple(map(int, box.xyxy[0].tolist()))
                    break

        self.frame_count += 1
        return self.last_hud_region

    def _extract_region(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        """Extract a region from the frame.

        Args:
            frame: Input frame
            region: (x1, y1, x2, y2) coordinates

        Returns:
            Extracted region as numpy array
        """
        x1, y1, x2, y2 = region
        return frame[y1:y2, x1:x2]

    def _detect_hud_elements(self, hud_frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect specific elements within the HUD region.

        Args:
            hud_frame: Cropped frame containing only HUD region

        Returns:
            List of detected elements with their details
        """
        results = self.model(hud_frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Skip HUD class itself and low confidence detections
            if class_id != self.classes["hud"] and confidence > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    {
                        "class": self.classes[class_id],
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return detections

    def _adjust_coordinates(
        self, elements: list[dict[str, Any]], hud_region: tuple[int, int, int, int]
    ) -> list[dict[str, Any]]:
        """Adjust coordinates of detected elements to the original frame.

        Args:
            elements: List of detected elements
            hud_region: Original HUD region coordinates

        Returns:
            List of elements with adjusted coordinates
        """
        hud_x1, hud_y1, _, _ = hud_region
        adjusted = []

        for elem in elements:
            x1, y1, x2, y2 = elem["bbox"]
            adjusted.append(
                {
                    "class": elem["class"],
                    "confidence": elem["confidence"],
                    "bbox": (x1 + hud_x1, y1 + hud_y1, x2 + hud_x1, y2 + hud_y1),
                }
            )

        return adjusted

    def _configure_model_settings(self):
        """Configure model settings based on hardware tier."""
        tier = self.optimizer.get_performance_tier()

        # Adjust settings based on hardware tier
        if tier == "low":
            self.confidence_threshold = 0.7  # Higher confidence to reduce false positives
            self.hud_detection_interval = 45  # Less frequent HUD detection
        elif tier == "medium":
            self.confidence_threshold = 0.6
            self.hud_detection_interval = 30
        else:  # high tier
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
            roi = frame[y1:y2, x1:x2]

            # TODO: Implement OCR using EasyOCR or Tesseract
            # For now, return empty string
            return ""

        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
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
        }

        # Process each detection
        for detection in results["detections"]:
            # Extract text based on element type
            text = self.extract_text(frame, detection)

            # Update game state based on element type
            # TODO: Implement parsing logic for each element type

        return game_state

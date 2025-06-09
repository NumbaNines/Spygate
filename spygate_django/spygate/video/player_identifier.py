"""
Player identification module using jersey number recognition and facial recognition.

This module provides functionality to identify players in video frames using
multiple identification methods, including jersey number OCR and facial recognition.
The module is hardware-aware and selects the best available methods based on
system capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import easyocr
import face_recognition
import numpy as np
import torch
from PIL import Image

from ..utils.tracking_hardware import TrackingHardwareManager, TrackingMode

logger = logging.getLogger(__name__)


class PlayerIdentifier:
    """
    Identifies players in video frames using jersey numbers and facial recognition.

    This class provides multiple identification methods and automatically selects
    the best ones based on hardware capabilities and performance requirements.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the player identifier.

        Args:
            confidence_threshold: Minimum confidence score for identifications (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.hardware_manager = TrackingHardwareManager()
        self.tracking_mode = self.hardware_manager.tracking_mode

        # Initialize OCR and face recognition based on hardware capabilities
        self._initialize_identification_models()

        # Cache for face encodings to avoid recomputing
        self.face_encodings_cache = {}

    def _initialize_identification_models(self):
        """Initialize identification models based on hardware capabilities."""
        # Initialize EasyOCR for jersey number recognition
        try:
            gpu = torch.cuda.is_available() and self.tracking_mode in [
                TrackingMode.ADVANCED,
                TrackingMode.PROFESSIONAL,
            ]
            self.ocr_reader = easyocr.Reader(["en"], gpu=gpu)
            logger.info(f"Initialized EasyOCR with GPU={gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.ocr_reader = None

        # Initialize face recognition if hardware supports it
        if self.tracking_mode in [TrackingMode.ADVANCED, TrackingMode.PROFESSIONAL]:
            logger.info("Initializing face recognition")
            self.use_face_recognition = True
        else:
            logger.info("Face recognition disabled due to hardware limitations")
            self.use_face_recognition = False

    def identify_player(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        reference_faces: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, Union[str, float]]:
        """
        Identify a player in a frame given their bounding box.

        Args:
            frame: Input video frame (BGR format)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            reference_faces: Dictionary mapping player names to their face encodings

        Returns:
            Dictionary containing identification results:
            {
                'jersey_number': str,  # Detected jersey number or None
                'jersey_confidence': float,  # Confidence in jersey number detection
                'face_match': str,  # Matched player name or None
                'face_confidence': float,  # Confidence in face match
            }
        """
        result = {
            "jersey_number": None,
            "jersey_confidence": 0.0,
            "face_match": None,
            "face_confidence": 0.0,
        }

        # Extract player region
        x1, y1, x2, y2 = map(int, bbox)
        player_region = frame[y1:y2, x1:x2]

        # Try jersey number recognition
        if self.ocr_reader is not None:
            jersey_result = self._recognize_jersey_number(player_region)
            if jersey_result is not None:
                result["jersey_number"] = jersey_result["number"]
                result["jersey_confidence"] = jersey_result["confidence"]

        # Try face recognition if enabled and reference faces provided
        if self.use_face_recognition and reference_faces:
            face_result = self._match_face(player_region, reference_faces)
            if face_result is not None:
                result["face_match"] = face_result["name"]
                result["face_confidence"] = face_result["confidence"]

        return result

    def _recognize_jersey_number(
        self, player_region: np.ndarray
    ) -> Optional[dict[str, Union[str, float]]]:
        """
        Recognize jersey number in player region using OCR.

        Args:
            player_region: Cropped image containing the player

        Returns:
            Dictionary with number and confidence if found, None otherwise
        """
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Run OCR
            results = self.ocr_reader.readtext(thresh)

            # Filter and process results
            for bbox, text, conf in results:
                # Clean and validate text
                text = "".join(filter(str.isdigit, text))
                if text and conf > self.confidence_threshold:
                    return {"number": text, "confidence": float(conf)}

        except Exception as e:
            logger.error(f"Jersey number recognition failed: {e}")

        return None

    def _match_face(
        self, player_region: np.ndarray, reference_faces: dict[str, np.ndarray]
    ) -> Optional[dict[str, Union[str, float]]]:
        """
        Match a face in the player region against reference faces.

        Args:
            player_region: Cropped image containing the player
            reference_faces: Dictionary mapping player names to their face encodings

        Returns:
            Dictionary with matched name and confidence if found, None otherwise
        """
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)

            # Detect and encode face
            face_locations = face_recognition.face_locations(rgb_region)
            if not face_locations:
                return None

            face_encoding = face_recognition.face_encodings(rgb_region, face_locations)[0]

            # Find best match
            best_match = None
            best_distance = float("inf")

            for name, ref_encoding in reference_faces.items():
                distance = face_recognition.face_distance([ref_encoding], face_encoding)[0]
                if distance < best_distance and distance < 0.6:  # 0.6 is a good threshold
                    best_distance = distance
                    best_match = name

            if best_match:
                # Convert distance to confidence (0-1 range)
                confidence = 1 - min(best_distance, 1.0)
                if confidence > self.confidence_threshold:
                    return {"name": best_match, "confidence": confidence}

        except Exception as e:
            logger.error(f"Face matching failed: {e}")

        return None

    def add_reference_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Add a reference face for future matching.

        Args:
            name: Name of the player
            face_image: Image containing the player's face (BGR format)

        Returns:
            bool: True if face was successfully added, False otherwise
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Detect and encode face
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                logger.warning(f"No face detected for player {name}")
                return False

            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
            self.face_encodings_cache[name] = face_encoding
            logger.info(f"Added reference face for player {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add reference face for {name}: {e}")
            return False

    def get_identification_info(self) -> dict:
        """Get information about available identification methods."""
        return {
            "tracking_mode": self.tracking_mode,
            "ocr_available": self.ocr_reader is not None,
            "face_recognition_enabled": self.use_face_recognition,
            "gpu_available": torch.cuda.is_available(),
            "confidence_threshold": self.confidence_threshold,
            "cached_faces": list(self.face_encodings_cache.keys()),
        }

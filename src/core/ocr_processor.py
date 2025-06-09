"""
OCR processor for extracting text from HUD elements.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles text extraction from HUD elements using OCR."""

    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        """Initialize the OCR processor.

        Args:
            tesseract_cmd: Optional path to tesseract executable
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Configure tesseract parameters for better accuracy on game text
        self.custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789QTHND&:"

        # Regions of interest for different HUD elements
        self.roi_configs = {
            "down_distance": {
                "preprocessing": "binary",
                "whitelist": "0123456789THND&",
                "expected_format": r"\d{1}[THND]{2}\s*&\s*\d{1,2}",
            },
            "game_clock": {
                "preprocessing": "binary",
                "whitelist": "0123456789:",
                "expected_format": r"\d{2}:\d{2}",
            },
            "score": {
                "preprocessing": "binary",
                "whitelist": "0123456789",
                "expected_format": r"\d{1,2}",
            },
        }

    def preprocess_image(self, img: np.ndarray, method: str) -> np.ndarray:
        """Preprocess image for better OCR results.

        Args:
            img: Input image
            method: Preprocessing method ('binary', 'adaptive', etc.)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if method == "binary":
            # Simple binary thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh
        elif method == "adaptive":
            # Adaptive thresholding for varying lighting conditions
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            return gray

    def extract_text(
        self, img: np.ndarray, roi_type: str, bbox: Optional[list[int]] = None
    ) -> tuple[str, float]:
        """Extract text from a specific region of interest.

        Args:
            img: Input image
            roi_type: Type of ROI ('down_distance', 'game_clock', 'score')
            bbox: Optional bounding box [x1, y1, x2, y2]

        Returns:
            Tuple of (extracted text, confidence score)
        """
        try:
            # Crop to ROI if bbox provided
            if bbox:
                x1, y1, x2, y2 = bbox
                roi = img[y1:y2, x1:x2]
            else:
                roi = img

            # Get ROI config
            config = self.roi_configs.get(
                roi_type, {"preprocessing": "binary", "whitelist": None, "expected_format": None}
            )

            # Preprocess
            processed = self.preprocess_image(roi, config["preprocessing"])

            # Update tesseract config with whitelist if specified
            ocr_config = self.custom_config
            if config["whitelist"]:
                ocr_config = f"{ocr_config} -c tessedit_char_whitelist={config['whitelist']}"

            # Extract text
            text = pytesseract.image_to_string(processed, config=ocr_config).strip()

            # Get confidence score
            data = pytesseract.image_to_data(
                processed, config=ocr_config, output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence for all detected text
            confidences = [float(conf) for conf in data["conf"] if conf != "-1"]
            confidence = np.mean(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            logger.error(f"OCR extraction failed for {roi_type}: {str(e)}")
            return "", 0.0

    def process_hud(
        self, frame: np.ndarray, detections: list[dict[str, Any]]
    ) -> dict[str, tuple[str, float]]:
        """Process all HUD elements in a frame.

        Args:
            frame: Input frame
            detections: List of detections with class and bbox

        Returns:
            Dictionary mapping HUD element types to (text, confidence) tuples
        """
        results = {}

        for det in detections:
            class_name = det["class"]
            bbox = det["bbox"]

            # Map detection classes to ROI types
            if class_name == "hud":
                # Process each HUD element type
                for roi_type in ["down_distance", "game_clock", "score"]:
                    text, conf = self.extract_text(frame, roi_type, bbox)
                    results[roi_type] = (text, conf)

            elif class_name == "playcall":
                text, conf = self.extract_text(frame, "playcall", bbox)
                results["playcall"] = (text, conf)

        return results

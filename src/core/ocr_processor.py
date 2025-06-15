"""
OCR processor for extracting text from HUD elements.

⚠️  DEPRECATED: This OCR processor is deprecated.
Use src/spygate/ml/enhanced_ocr.py instead, which includes:
- Optimal preprocessing parameters (0.925 score from 20K parameter sweep)
- Multi-engine OCR with fallback
- Game-specific optimizations
- Temporal smoothing and validation

This file is kept for legacy compatibility only.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Issue deprecation warning
warnings.warn(
    "OCRProcessor is deprecated. Use src.spygate.ml.enhanced_ocr.EnhancedOCR instead "
    "for optimal performance with scientifically-determined preprocessing parameters.",
    DeprecationWarning,
    stacklevel=2,
)


class OCRProcessor:
    """
    ⚠️  DEPRECATED: Handles text extraction from HUD elements using OCR.

    Use src.spygate.ml.enhanced_ocr.EnhancedOCR instead for:
    - 92.5% accuracy with optimal preprocessing parameters
    - Multi-engine OCR with custom Madden model + EasyOCR + Tesseract fallback
    - Game-specific text patterns and validation
    - Temporal smoothing and historical tracking
    """

    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        """Initialize the OCR processor.

        Args:
            tesseract_cmd: Optional path to tesseract executable
        """
        warnings.warn(
            "OCRProcessor is deprecated. Use EnhancedOCR from src.spygate.ml.enhanced_ocr instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Configure tesseract parameters for better accuracy on game text
        self.custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789QTHND&:"

        # Regions of interest for different HUD elements
        self.roi_configs = {
            "down_distance": {
                "preprocessing": "optimal",  # Use optimal preprocessing
                "whitelist": "0123456789THND&",
                "expected_format": r"\d{1}[THND]{2}\s*&\s*\d{1,2}",
            },
            "game_clock": {
                "preprocessing": "optimal",  # Use optimal preprocessing
                "whitelist": "0123456789:",
                "expected_format": r"\d{2}:\d{2}",
            },
            "score": {
                "preprocessing": "optimal",  # Use optimal preprocessing
                "whitelist": "0123456789",
                "expected_format": r"\d{1,2}",
            },
        }

    def preprocess_image(self, img: np.ndarray, method: str) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        ⚠️  DEPRECATED: Use EnhancedOCR.preprocess_image() instead for optimal results.
        This method now uses the scientifically-determined optimal parameters.

        Args:
            img: Input image
            method: Preprocessing method ('optimal' recommended, legacy: 'binary', 'adaptive')

        Returns:
            Preprocessed image
        """
        if method == "optimal":
            # Use the OPTIMAL parameters found through 20K parameter sweep (Score: 0.925)
            try:
                # Store original dimensions
                original_height, original_width = img.shape[:2]

                # Stage 1: Convert to grayscale (ALWAYS FIRST)
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 2:
                    gray = img.copy()
                else:
                    gray = img.copy()

                # Stage 2: Scale with LANCZOS4 (ALWAYS SECOND) - OPTIMAL: 3.5x
                scale_factor = 3.5
                new_height, new_width = int(gray.shape[0] * scale_factor), int(
                    gray.shape[1] * scale_factor
                )
                scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

                # Stage 3: CLAHE (ALWAYS THIRD) - OPTIMAL: clip=1.0, grid=(4,4)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
                clahe_applied = clahe.apply(scaled)

                # Stage 4: Gamma correction (CONDITIONAL) - OPTIMAL: off
                # Skipping gamma correction as optimal setting is 'off'
                gamma_corrected = clahe_applied

                # Stage 5: Gaussian blur (CONDITIONAL) - OPTIMAL: (3,3)
                blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

                # Stage 6: Thresholding (ALWAYS APPLIED) - OPTIMAL: adaptive_mean, block=13, C=3
                thresholded = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 3
                )

                # Stage 7: Morphological closing (ALWAYS APPLIED) - OPTIMAL: (3,3) kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

                # Stage 8: Sharpening (CONDITIONAL) - OPTIMAL: off
                # Skipping sharpening as optimal setting is False
                final = morphed

                # Resize back to original dimensions
                resized = cv2.resize(
                    final, (original_width, original_height), interpolation=cv2.INTER_AREA
                )

                return resized

            except Exception as e:
                logger.error(f"Error in optimal preprocessing: {e}")
                # Fallback to original image
                return img

        elif method == "binary":
            # Legacy: Simple binary thresholding
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh

        elif method == "adaptive":
            # Legacy: Adaptive thresholding for varying lighting conditions
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            return img

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

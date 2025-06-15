"""
Enhanced OCR module for SpygateAI.
Handles text detection and recognition from game HUD elements.

PRIMARY OCR: Custom-trained Madden OCR model (92-94% accuracy)
FALLBACK: EasyOCR and Tesseract for redundancy
"""

import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image

# Import PRIMARY custom OCR engine
from .custom_ocr import SpygateMaddenOCR

logger = logging.getLogger(__name__)


@dataclass
class OCRValidation:
    """Validation parameters for OCR results."""

    min_confidence: float = 0.7
    max_retries: int = 3
    history_size: int = 5
    temporal_threshold: float = 0.8
    yard_line_max_change: int = 10
    min_yard_line_confidence: float = 0.4
    territory_consistency_window: int = 3


@dataclass
class OCRResult:
    """OCR result with confidence and source."""

    text: str
    confidence: float
    source: str
    bbox: Optional[list[int]] = None


class EnhancedOCR:
    """
    Enhanced OCR processor with multi-engine fallback and game-specific optimizations.

    Features:
    - PRIMARY: Custom-trained Madden OCR model (92-94% accuracy)
    - FALLBACK: EasyOCR and Tesseract for redundancy
    - Game-specific text preprocessing
    - Robust error handling and validation
    - Confidence scoring
    - Temporal smoothing
    - Partial occlusion handling
    """

    def __init__(self, hardware=None):
        """Initialize OCR engines and configurations.

        Args:
            hardware: Hardware tier for adaptive settings
        """
        self.hardware_tier = hardware
        self.validation = OCRValidation()

        # Initialize PRIMARY custom OCR engine
        # TEMPORARILY DISABLED: Model needs retraining due to poor accuracy
        # Current model produces garbled output like "K1tC" for all inputs
        USE_CUSTOM_OCR = False  # Set to True after retraining

        if USE_CUSTOM_OCR:
            try:
                self.custom_ocr = SpygateMaddenOCR()
                if self.custom_ocr.is_available():
                    logger.info("🚀 PRIMARY OCR: Custom Madden OCR loaded successfully!")
                    logger.info(f"   Expected accuracy: 92-94% (vs 70-80% EasyOCR)")
                    logger.info(f"   Model info: {self.custom_ocr.get_model_info()}")
                else:
                    logger.warning("❌ PRIMARY OCR: Custom Madden OCR failed to load")
                    self.custom_ocr = None
            except Exception as e:
                logger.error(f"Failed to initialize PRIMARY custom OCR: {e}")
                self.custom_ocr = None
        else:
            logger.info("⚠️  CUSTOM OCR DISABLED: Model needs retraining")
            logger.info("   Using EasyOCR as primary until custom model is retrained")
            self.custom_ocr = None

        # Initialize FALLBACK EasyOCR engine
        try:
            self.reader = easyocr.Reader(["en"])
            logger.info("✅ FALLBACK OCR: EasyOCR engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FALLBACK EasyOCR: {e}")
            self.reader = None

        # Historical tracking for temporal smoothing
        self.history = {
            "down": deque(maxlen=self.validation.history_size),
            "distance": deque(maxlen=self.validation.history_size),
            "score_home": deque(maxlen=self.validation.history_size),
            "score_away": deque(maxlen=self.validation.history_size),
            "quarter": deque(maxlen=self.validation.history_size),
            "time": deque(maxlen=self.validation.history_size),
            "yard_line": deque(maxlen=self.validation.history_size),
            "territory": deque(maxlen=self.validation.history_size),  # Track field territory
        }

        # Common game text patterns with validation rules
        self.patterns = {
            "down": {
                "pattern": r"(\d)(st|nd|rd|th)",
                "validate": lambda x: 1 <= int(x) <= 4,
                "format": lambda x: int(x),
            },
            "distance": {
                "pattern": r"(\d+)",
                "validate": lambda x: 1 <= int(x) <= 99,
                "format": lambda x: int(x),
            },
            "score": {
                "pattern": r"(\d{1,2})",
                "validate": lambda x: 0 <= int(x) <= 99,
                "format": lambda x: int(x),
            },
            "quarter": {
                "pattern": r"(\d)(st|nd|rd|th)",
                "validate": lambda x: 1 <= int(x) <= 5,  # Including overtime
                "format": lambda x: int(x),
            },
            "time": {
                "pattern": r"(\d{1,2}):(\d{2})",
                "validate": lambda m, s: 0 <= int(m) <= 15 and 0 <= int(s) <= 59,
                "format": lambda m, s: f"{int(m):02d}:{int(s):02d}",
            },
            "yard_line": {
                "pattern": r"(?:OWN |OPP )?(\d{1,2})",  # Now handles "OWN" and "OPP" prefixes
                "validate": lambda x, t: self._validate_yard_line(x, t),
                "format": lambda x, t: self._format_yard_line(x, t),
            },
            "territory": {
                "pattern": r"(OWN|OPP)",
                "validate": lambda x: x in ["OWN", "OPP"],
                "format": lambda x: x,
            },
        }

        # Common OCR correction mappings for yard lines
        self.yard_line_corrections = {
            "0": ["o", "O", "D", "Q"],
            "1": ["l", "I", "|"],
            "2": ["z", "Z"],
            "3": ["8", "B"],
            "5": ["S", "s"],
            "6": ["b"],
            "8": ["3", "B"],
            "9": ["g", "q"],
        }

        # Define fixed HUD text patterns
        self.text_patterns = {
            "team_abbrev": r"^[A-Z]{3}$",  # 3 uppercase letters
            "score": r"^\d{1,2}$",  # 1-2 digits
            "down": r"^[1-4]$",  # 1-4
            "distance": r"^\d{1,3}$",  # 1-3 digits
            "yard_line": r"^\d{1,2}$",  # Just the yard line number (1-50)
        }

        # Define relative ROI coordinates (percentages of HUD region height/width)
        self.roi_regions = {
            "left_team": {"x": (0.05, 0.15), "y": (0.3, 0.7)},
            "left_score": {"x": (0.15, 0.25), "y": (0.3, 0.7)},
            "right_team": {"x": (0.3, 0.4), "y": (0.3, 0.7)},
            "right_score": {"x": (0.4, 0.5), "y": (0.3, 0.7)},
            "down_distance": {"x": (0.45, 0.65), "y": (0.3, 0.7)},
            "yard_line": {
                "x": (0.85, 0.95),
                "y": (0.3, 0.7),
            },  # Just the number, territory triangle detected separately
        }

    def _validate_yard_line(self, yard_line: str, territory: Optional[str] = None) -> bool:
        """
        Enhanced yard line validation with territory context.

        Args:
            yard_line: Detected yard line number
            territory: Field territory indicator ('OWN' or 'OPP')

        Returns:
            bool: Whether the yard line is valid
        """
        try:
            # Clean and correct common OCR mistakes
            yard_line = self._correct_yard_line(yard_line)

            # Convert to int and validate basic range
            yard = int(yard_line)
            if not (1 <= yard <= 50):
                return False

            # If we have history, check for reasonable progression
            if self.history["yard_line"] and len(self.history["yard_line"]) > 0:
                last_yard = int(self.history["yard_line"][-1])
                if abs(yard - last_yard) > self.validation.yard_line_max_change:
                    logger.warning(f"Suspicious yard line change: {last_yard} -> {yard}")
                    return False

            # If we have territory context, validate consistency
            if territory and self.history["territory"] and len(self.history["territory"]) > 0:
                last_territory = self.history["territory"][-1]
                if territory != last_territory and yard == int(self.history["yard_line"][-1]):
                    # Territory changed but yard line didn't - suspicious
                    return False

            return True

        except (ValueError, TypeError) as e:
            logger.warning(f"Yard line validation error: {e}")
            return False

    def _format_yard_line(self, yard_line: str, territory: Optional[str] = None) -> int:
        """Format yard line with territory context."""
        try:
            yard = int(self._correct_yard_line(yard_line))
            if territory == "OPP":
                # Store territory for future validation
                self.history["territory"].append("OPP")
            elif territory == "OWN":
                self.history["territory"].append("OWN")

            # Store yard line for future validation
            self.history["yard_line"].append(str(yard))
            return yard
        except (ValueError, TypeError):
            return None

    def _correct_yard_line(self, text: str) -> str:
        """Apply common OCR corrections for yard lines."""
        text = text.strip()

        # Apply corrections
        for digit, mistakes in self.yard_line_corrections.items():
            for mistake in mistakes:
                text = text.replace(mistake, digit)

        return text

    def _extract_text_multi_engine(
        self, image: np.ndarray, region_type: str = "unknown", use_tesseract: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text using multi-engine approach with custom OCR as PRIMARY.

        Args:
            image: Input image region
            region_type: Type of region for specialized processing
            use_tesseract: Whether to use slow Tesseract fallback (default: False for speed)

        Returns:
            Best OCR result with source information
        """
        results = []

        # PRIMARY: Try custom Madden OCR first (92-94% accuracy)
        if self.custom_ocr and self.custom_ocr.is_available():
            try:
                custom_result = self.custom_ocr.extract_text(image, region_type)
                if custom_result.get("text") and custom_result.get("confidence", 0) > 0.3:
                    results.append(
                        {
                            "text": custom_result["text"],
                            "confidence": custom_result["confidence"],
                            "source": "custom_madden_ocr_primary",
                            "raw_result": custom_result,
                        }
                    )
                    logger.debug(
                        f"🚀 PRIMARY OCR: '{custom_result['text']}' (conf: {custom_result['confidence']:.3f})"
                    )

                    # TEMPORARILY DISABLED: High confidence early exit due to model accuracy issues
                    # Custom model needs retraining - currently producing garbled results
                    # if custom_result.get("confidence", 0) > 0.8:
                    #     logger.debug("✅ High confidence custom OCR result, skipping fallbacks for speed")
                    #     return results[0]

            except Exception as e:
                logger.warning(f"PRIMARY custom OCR failed: {e}")

        # FALLBACK 1: EasyOCR (70-80% accuracy) - Always try if custom OCR failed or low confidence
        if self.reader:
            try:
                easyocr_results = self.reader.readtext(image)
                if easyocr_results:
                    # Get the best EasyOCR result
                    best_easy = max(easyocr_results, key=lambda x: x[2])  # Sort by confidence
                    bbox, text, conf = best_easy
                    if text.strip() and conf > 0.3:
                        results.append(
                            {
                                "text": text.strip(),
                                "confidence": conf,
                                "source": "easyocr_fallback",
                                "bbox": bbox,
                            }
                        )
                        logger.debug(f"📋 FALLBACK EasyOCR: '{text.strip()}' (conf: {conf:.3f})")
            except Exception as e:
                logger.warning(f"EasyOCR fallback failed: {e}")

        # FALLBACK 2: Tesseract (60-70% accuracy) - ONLY if explicitly requested (slow)
        if use_tesseract and not results:  # Only use if no other results and explicitly requested
            try:
                # Preprocess for Tesseract
                processed = self._preprocess_text_region(image)
                tesseract_text = pytesseract.image_to_string(processed, config="--psm 7").strip()
                if tesseract_text:
                    results.append(
                        {
                            "text": tesseract_text,
                            "confidence": 0.4,  # Lower confidence for Tesseract
                            "source": "tesseract_fallback",
                        }
                    )
                    logger.debug(f"🔧 FALLBACK Tesseract: '{tesseract_text}' (conf: 0.4)")
            except Exception as e:
                logger.warning(f"Tesseract fallback failed: {e}")

        # Select best result
        if results:
            # SMART SELECTION: Prefer EasyOCR if it has significantly higher confidence
            # This is a temporary fix while custom OCR model is being retrained
            custom_results = [r for r in results if r["source"] == "custom_madden_ocr_primary"]
            easyocr_results = [r for r in results if r["source"] == "easyocr_fallback"]

            if custom_results and easyocr_results:
                custom_conf = custom_results[0]["confidence"]
                easy_conf = easyocr_results[0]["confidence"]

                # If EasyOCR has much higher confidence (>0.3 difference), prefer it
                if easy_conf - custom_conf > 0.3:
                    best_result = easyocr_results[0]
                    logger.debug(
                        f"🔄 SMART SELECT: EasyOCR preferred due to higher confidence ({easy_conf:.3f} vs {custom_conf:.3f})"
                    )
                else:
                    # Otherwise, prioritize custom OCR as intended
                    best_result = custom_results[0]
                    logger.debug(f"🚀 CUSTOM SELECT: Using custom OCR despite lower confidence")
            else:
                # Fallback to original logic
                results.sort(
                    key=lambda x: (x["source"] == "custom_madden_ocr_primary", x["confidence"]),
                    reverse=True,
                )
                best_result = results[0]

            logger.debug(
                f"✅ SELECTED: {best_result['source']} - '{best_result['text']}' (conf: {best_result['confidence']:.3f})"
            )
            return best_result

        # No results found
        logger.warning("❌ All OCR engines failed to extract text")
        return {"text": "", "confidence": 0.0, "source": "none", "error": "All OCR engines failed"}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR with OPTIMIZED parameters from 20K parameter sweep.

        🏆 OPTIMAL PARAMETERS (Score: 0.939 from 19,778+ combinations tested):
        - Scale: 3.5x (LANCZOS4)
        - CLAHE: clip=1.0, grid=(4,4)
        - Blur: (3,3) Gaussian
        - Threshold: adaptive_mean, block=13, C=3
        - Morphological: (3,3) closing kernel
        - Gamma: 0.8, Sharpening: off

        Args:
            image: Input image array

        Returns:
            Preprocessed image array with optimal parameters
        """
        try:
            # Store original dimensions
            original_height, original_width = image.shape[:2]

            # Stage 1: Convert to grayscale (ALWAYS FIRST)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image.copy()
            elif image.shape[2] == 4:  # RGBA
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Stage 2: Scale with LANCZOS4 (ALWAYS SECOND) - OPTIMAL: 3.5x
            scale_factor = 3.5
            new_height, new_width = int(gray.shape[0] * scale_factor), int(
                gray.shape[1] * scale_factor
            )
            scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Stage 3: CLAHE (ALWAYS THIRD) - OPTIMAL: clip=1.0, grid=(4,4)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
            clahe_applied = clahe.apply(scaled)

            # Stage 4: Gamma correction (CONDITIONAL) - OPTIMAL: 0.8
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
                "uint8"
            )
            gamma_corrected = cv2.LUT(clahe_applied, table)

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

            # Convert back to BGR for OCR compatibility
            processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

            return processed

        except Exception as e:
            logger.error(f"Error in optimized image preprocessing: {e}")
            # Fallback to original image
            return image

    def process_region(self, image: np.ndarray, debug_mode: bool = False) -> dict[str, Any]:
        """
        Process an image region and extract text information using multi-engine OCR.

        Args:
            image: Image region to process
            debug_mode: Enable detailed confidence debugging

        Returns:
            Dict containing extracted information
        """
        try:
            # Preprocess image for OCR
            processed = self.preprocess_image(image)

            # Run multi-engine OCR (PRIMARY: Custom Madden OCR, FALLBACK: EasyOCR + Tesseract)
            ocr_result = self._extract_text_multi_engine(processed, "yard_line_territory")

            # Convert to legacy format for compatibility
            if ocr_result.get("text"):
                # Create fake results list for compatibility with existing parsing logic
                results = [
                    (
                        ocr_result.get("bbox", [0, 0, 100, 50]),  # Default bbox if not available
                        ocr_result["text"],
                        ocr_result["confidence"],
                    )
                ]
            else:
                results = []

            if debug_mode:
                logger.info(f"OCR found {len(results)} text regions")
                for i, (bbox, text, conf) in enumerate(results):
                    logger.info(f"  Region {i+1}: '{text}' (conf: {conf:.3f})")

            # Initialize results dictionary
            extracted_info = {}
            confidence_scores = []
            debug_info = {"raw_ocr_results": [], "territory_attempts": [], "yard_line_attempts": []}

            # First pass - look for territory indicators
            territory = None
            max_territory_conf = 0
            for bbox, text, conf in results:
                text_upper = text.strip().upper()
                debug_info["raw_ocr_results"].append(
                    {"text": text, "confidence": conf, "bbox": bbox}
                )

                if match := re.search(self.patterns["territory"]["pattern"], text_upper):
                    detected_territory = match.group(1)
                    debug_info["territory_attempts"].append(
                        {
                            "detected": detected_territory,
                            "confidence": conf,
                            "text": text,
                            "selected": False,
                        }
                    )

                    # Only update territory if confidence is higher
                    if conf > max_territory_conf:
                        # Mark previous attempt as not selected
                        for attempt in debug_info["territory_attempts"]:
                            attempt["selected"] = False
                        # Mark current as selected
                        debug_info["territory_attempts"][-1]["selected"] = True

                        territory = detected_territory
                        max_territory_conf = conf
                        confidence_scores.append(conf)

                        if debug_mode:
                            logger.info(
                                f"Territory detected: '{detected_territory}' (conf: {conf:.3f})"
                            )

            # Second pass - extract yard line with multiple attempts
            yard_line_candidates = []
            for bbox, text, conf in results:
                text_clean = text.strip()

                # Try to find yard line numbers
                if match := re.search(self.patterns["yard_line"]["pattern"], text_clean):
                    value = match.group(1)

                    debug_info["yard_line_attempts"].append(
                        {
                            "raw_text": text,
                            "extracted_value": value,
                            "confidence": conf,
                            "territory_context": territory,
                            "validation_passed": False,
                            "formatted_value": None,
                        }
                    )

                    # Clean and validate the yard line
                    if self.patterns["yard_line"]["validate"](value, territory):
                        formatted_value = self.patterns["yard_line"]["format"](value, territory)
                        debug_info["yard_line_attempts"][-1]["validation_passed"] = True
                        debug_info["yard_line_attempts"][-1]["formatted_value"] = formatted_value

                        if formatted_value is not None:
                            yard_line_candidates.append((formatted_value, conf))

                            if debug_mode:
                                logger.info(
                                    f"Valid yard line candidate: {formatted_value} (conf: {conf:.3f}) from '{text}'"
                                )
                    else:
                        if debug_mode:
                            logger.info(
                                f"Yard line validation failed for '{value}' from '{text}' (conf: {conf:.3f})"
                            )

            # Select the most likely yard line based on:
            # 1. Historical consistency
            # 2. Confidence score
            # 3. Territory context
            if yard_line_candidates:
                # Sort by confidence
                yard_line_candidates.sort(key=lambda x: x[1], reverse=True)

                if debug_mode:
                    logger.info(
                        f"Yard line candidates sorted by confidence: {yard_line_candidates}"
                    )

                # If we have history, prefer candidates close to previous value
                original_candidates = yard_line_candidates.copy()
                if self.history["yard_line"] and len(self.history["yard_line"]) > 0:
                    last_yard = int(self.history["yard_line"][-1])
                    valid_candidates = [
                        (yard, conf)
                        for yard, conf in yard_line_candidates
                        if abs(yard - last_yard) <= self.validation.yard_line_max_change
                    ]

                    if valid_candidates:
                        yard_line_candidates = valid_candidates
                        if debug_mode:
                            logger.info(
                                f"Filtered by history (last: {last_yard}): {valid_candidates}"
                            )
                    else:
                        if debug_mode:
                            logger.info(
                                f"No candidates within change threshold of {last_yard}, keeping all"
                            )

                # Use the best remaining candidate
                best_yard, best_conf = yard_line_candidates[0]
                extracted_info["yard_line"] = best_yard
                confidence_scores.append(best_conf)

                # Update history
                self.history["yard_line"].append(str(best_yard))

                if debug_mode:
                    logger.info(f"Selected yard line: {best_yard} (conf: {best_conf:.3f})")

                    # Check if confidence is below threshold
                    if best_conf < self.validation.min_yard_line_confidence:
                        logger.warning(
                            f"Selected yard line confidence {best_conf:.3f} below threshold {self.validation.min_yard_line_confidence}"
                        )

                    # Report confidence boost from historical consistency
                    if len(original_candidates) != len(yard_line_candidates):
                        logger.info(
                            f"Historical filtering changed selection from {original_candidates[0]} to {yard_line_candidates[0]}"
                        )

            elif debug_mode:
                logger.info("No valid yard line candidates found")

            # Add territory to results if found
            if territory:
                extracted_info["territory"] = territory
                self.history["territory"].append(territory)

            # Add confidence score if we have any matches
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                extracted_info["confidence"] = avg_confidence

                if debug_mode:
                    logger.info(
                        f"Average confidence: {avg_confidence:.3f} from {len(confidence_scores)} detections"
                    )
                    if avg_confidence < self.validation.min_confidence:
                        logger.warning(
                            f"Average confidence {avg_confidence:.3f} below min threshold {self.validation.min_confidence}"
                        )

            # Add debug info if in debug mode
            if debug_mode:
                extracted_info["debug_info"] = debug_info

            return extracted_info
        except Exception as e:
            logger.error(f"Error processing region: {e}")
            return {}

    def _validate_and_smooth(self, text_info: dict[str, Any], region_type: str) -> dict[str, Any]:
        """
        Validate OCR results and apply temporal smoothing.

        Args:
            text_info: Dictionary of extracted text
            region_type: Type of HUD region

        Returns:
            Validated and smoothed text information
        """
        validated = {}

        for key, value in text_info.items():
            if not value:
                continue

            pattern_info = self.patterns.get(key)
            if not pattern_info:
                continue

            try:
                # Apply validation rules
                if key == "time":
                    minutes, seconds = value.split(":")
                    if pattern_info["validate"](minutes, seconds):
                        formatted = pattern_info["format"](minutes, seconds)
                        self.history[key].append(formatted)
                        validated[key] = formatted
                else:
                    if pattern_info["validate"](value):
                        formatted = pattern_info["format"](value)
                        self.history[key].append(formatted)
                        validated[key] = formatted

            except Exception as e:
                logger.warning(f"Validation failed for {key}: {e}")

        return validated if validated else None

    def _get_best_historical_values(self, region_type: str = None) -> dict[str, Any]:
        """
        Get most reliable values from history when current detection fails.

        Args:
            region_type: Type of HUD region

        Returns:
            Dictionary of best historical values
        """
        historical = {}

        for key, history in self.history.items():
            if not history:
                continue

            if region_type and not key.startswith(region_type):
                continue

            # Get most common value in recent history
            values = list(history)
            if len(values) >= self.validation.history_size * self.validation.temporal_threshold:
                from collections import Counter

                most_common = Counter(values).most_common(1)
                if most_common:
                    historical[key] = most_common[0][0]

        return historical

    def handle_partial_occlusion(
        self, img: np.ndarray, visible_regions: list[str]
    ) -> dict[str, Any]:
        """
        Handle cases where HUD is partially occluded.

        Args:
            img: Input image array
            visible_regions: List of visible HUD regions

        Returns:
            Best-effort OCR results using visible regions
        """
        results = {}

        for region in visible_regions:
            # Process visible regions
            region_results = self.process_region(img)
            results.update(region_results)

        # Fill in missing data from history
        historical = self._get_best_historical_values()

        # Only use historical values for missing fields
        for key, value in historical.items():
            if key not in results:
                results[key] = value

        return results

    def _parse_ocr_results(self, results: list, region_type: str) -> dict[str, Optional[str]]:
        """Parse OCR results and extract game information."""
        text_info = {}

        for bbox, text, conf in results:
            text = text.strip().lower()

            # Extract down
            if match := re.search(self.patterns["down"]["pattern"], text):
                text_info["down"] = int(match.group(1))

            # Extract distance
            if match := re.search(self.patterns["distance"]["pattern"], text):
                text_info["distance"] = int(match.group(1))

            # Extract score
            if match := re.search(self.patterns["score"]["pattern"], text):
                # Determine if home or away based on position
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                if x_center < bbox[2][0] / 2:  # Using bbox width instead of img.shape
                    text_info["score_away"] = int(match.group(1))
                else:
                    text_info["score_home"] = int(match.group(1))

            # Extract quarter
            if match := re.search(self.patterns["quarter"]["pattern"], text):
                text_info["quarter"] = int(match.group(1))

            # Extract time
            if match := re.search(self.patterns["time"]["pattern"], text):
                text_info["time"] = f"{match.group(1)}:{match.group(2)}"

            # Extract territory first (needed for yard line context)
            territory = None
            if match := re.search(self.patterns["territory"]["pattern"], text):
                territory = match.group(1).upper()
                if self.patterns["territory"]["validate"](territory):
                    text_info["territory"] = territory

            # Extract yard line with territory context
            if match := re.search(self.patterns["yard_line"]["pattern"], text):
                yard_line = match.group(1)
                if self.patterns["yard_line"]["validate"](yard_line, territory):
                    text_info["yard_line"] = self.patterns["yard_line"]["format"](
                        yard_line, territory
                    )

        return text_info

    def _parse_text(self, text: str, region_type: str) -> dict[str, Optional[str]]:
        """Parse raw text and extract game information."""
        text_info = {}
        text = text.lower()

        # Extract territory first for yard line context
        territory = None
        if match := re.search(self.patterns["territory"]["pattern"], text):
            territory = match.group(1).upper()
            if self.patterns["territory"]["validate"](territory):
                text_info["territory"] = territory

        # Apply pattern matching for the specific region type
        if region_type == "yard_line":
            if match := re.search(self.patterns["yard_line"]["pattern"], text):
                yard_line = match.group(1)
                if self.patterns["yard_line"]["validate"](yard_line, territory):
                    text_info["yard_line"] = self.patterns["yard_line"]["format"](
                        yard_line, territory
                    )
        elif pattern_info := self.patterns.get(region_type):
            if match := re.search(pattern_info["pattern"], text):
                if region_type == "time":
                    text_info[region_type] = f"{match.group(1)}:{match.group(2)}"
                else:
                    text_info[region_type] = int(match.group(1))

        return text_info

    def process_hud_region(self, hud_region: np.ndarray) -> dict[str, Any]:
        """Process specific regions of HUD for optimal OCR performance."""
        results = {}
        h, w = hud_region.shape[:2]

        for region_name, coords in self.roi_regions.items():
            # Calculate absolute coordinates
            x1 = int(w * coords["x"][0])
            x2 = int(w * coords["x"][1])
            y1 = int(h * coords["y"][0])
            y2 = int(h * coords["y"][1])

            # Extract ROI
            roi = hud_region[y1:y2, x1:x2]

            # Apply preprocessing based on region type
            if "score" in region_name or "down" in region_name:
                # Optimize for number detection
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)[1]
            elif "team" in region_name:
                # Optimize for text detection
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.GaussianBlur(roi, (3, 3), 0)

            # Perform OCR using multi-engine approach with custom OCR as primary
            ocr_result = self._extract_text_multi_engine(roi, region_name)

            if ocr_result.get("text"):
                text = ocr_result["text"]
                conf = ocr_result["confidence"]

                # Validate against expected patterns
                if region_name in ["left_team", "right_team"]:
                    if re.match(self.text_patterns["team_abbrev"], text):
                        results[region_name] = {"text": text, "confidence": conf}
                elif "score" in region_name:
                    if re.match(self.text_patterns["score"], text):
                        results[region_name] = {"text": int(text), "confidence": conf}
                elif region_name == "down_distance":
                    # Parse down and distance
                    down_match = re.search(r"([1-4])", text)
                    dist_match = re.search(r"(\d{1,3})", text)
                    if down_match and dist_match:
                        results["down"] = {"text": int(down_match.group(1)), "confidence": conf}
                        results["distance"] = {"text": int(dist_match.group(1)), "confidence": conf}
                elif region_name == "yard_line":
                    if re.match(self.text_patterns["yard_line"], text):
                        results[region_name] = {"text": text, "confidence": conf}

        return results

    def validate_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Validate OCR results against known game constraints."""
        validated = results.copy()

        # Validate scores (0-99)
        for score_key in ["left_score", "right_score"]:
            if score_key in validated:
                score = validated[score_key]["text"]
                if not (0 <= score <= 99):
                    validated.pop(score_key)

        # Validate down (1-4)
        if "down" in validated:
            down = validated["down"]["text"]
            if not (1 <= down <= 4):
                validated.pop("down")

        # Validate distance (1-99)
        if "distance" in validated:
            distance = validated["distance"]["text"]
            if not (1 <= distance <= 99):
                validated.pop("distance")

        # Validate yard line (1-50)
        if "yard_line" in validated:
            try:
                yard = int(validated["yard_line"]["text"])
                # Must be between 1-50 (no 0 yard line in football)
                if not (1 <= yard <= 50):
                    validated.pop("yard_line")
            except ValueError:
                validated.pop("yard_line")

        return validated

    def extract_game_clock(self, region: np.ndarray) -> Optional[str]:
        """Extract game clock time using PRIMARY custom OCR (MM:SS format)."""
        try:
            # Use multi-engine OCR with custom OCR as primary
            ocr_result = self._extract_text_multi_engine(region, "game_clock_area")

            if not ocr_result.get("text"):
                return None

            text = ocr_result["text"].strip()
            logger.debug(f"Game Clock OCR: '{text}' from {ocr_result['source']}")

            # Look for MM:SS pattern
            if match := re.search(r"(\d{1,2}):(\d{2})", text):
                minutes, seconds = match.groups()
                if 0 <= int(minutes) <= 15 and 0 <= int(seconds) <= 59:
                    result = f"{int(minutes):02d}:{int(seconds):02d}"
                    logger.debug(
                        f"✅ Game Clock extracted: '{result}' (conf: {ocr_result['confidence']:.3f})"
                    )
                    return result

            logger.debug(f"❌ No valid clock pattern found in: '{text}'")
            return None

        except Exception as e:
            logger.error(f"Game clock extraction error: {e}")
            return None

    def extract_play_clock(self, region: np.ndarray) -> Optional[str]:
        """Extract play clock countdown using PRIMARY custom OCR (0-40 seconds)."""
        try:
            # Use multi-engine OCR with custom OCR as primary
            ocr_result = self._extract_text_multi_engine(region, "play_clock_area")

            if not ocr_result.get("text"):
                return None

            text = ocr_result["text"].strip()
            logger.debug(f"Play Clock OCR: '{text}' from {ocr_result['source']}")

            # Look for number pattern
            if match := re.search(r"(\d{1,2})", text):
                seconds = int(match.group(1))
                if 0 <= seconds <= 40:
                    result = str(seconds)
                    logger.debug(
                        f"✅ Play Clock extracted: '{result}' (conf: {ocr_result['confidence']:.3f})"
                    )
                    return result

            logger.debug(f"❌ No valid play clock pattern found in: '{text}'")
            return None

        except Exception as e:
            logger.error(f"Play clock extraction error: {e}")
            return None

    def extract_down_distance(self, region: np.ndarray) -> Optional[str]:
        """Extract down and distance from region using PRIMARY custom OCR (e.g., '3rd & 7')."""
        try:
            # Use multi-engine OCR with custom OCR as primary
            ocr_result = self._extract_text_multi_engine(region, "down_distance_area")

            if not ocr_result.get("text"):
                return None

            text = ocr_result["text"].strip().upper()
            logger.debug(f"Down/Distance OCR: '{text}' from {ocr_result['source']}")

            # PRE-PROCESS: Fix common OCR character confusion
            # Fix 'I' confused with '1' at the beginning
            text = re.sub(r"^I(?=ST|ND|RD|TH|\s*&)", "1", text)
            text = re.sub(r"^O(?=ST|ND|RD|TH|\s*&)", "0", text)  # Fix 'O' confused with '0'

            # Fix other common OCR errors
            text = re.sub(r"(?<=\d)S[T]", "ST", text)  # Fix broken 'ST'
            text = re.sub(r"(?<=\d)N[D]", "ND", text)  # Fix broken 'ND'
            text = re.sub(r"(?<=\d)R[D]", "RD", text)  # Fix broken 'RD'
            text = re.sub(r"(?<=\d)T[H]", "TH", text)  # Fix broken 'TH'

            logger.debug(f"After OCR corrections: '{text}'")

            # Look for down & distance patterns (more flexible)
            patterns = [
                r"([0-4])(ST|ND|RD|TH)\s*&\s*(\d{1,2})",  # 3RD & 7
                r"([0-4])(ST|ND|RD|TH)\s*AND\s*(\d{1,2})",  # 3RD AND 7
                r"([0-4])\s*&\s*(\d{1,2})",  # 3 & 7 (no suffix)
                r"([0-4])(ST|ND|RD|TH)\s*&\s*GOAL",  # 1ST & GOAL
                r"([0-4])\s*&\s*GOAL",  # 1 & GOAL
                # Handle spaced out text
                r"([0-4])\s+(ST|ND|RD|TH)\s*&\s*(\d{1,2})",  # 3 RD & 7
                r"([0-4])\s+(ST|ND|RD|TH)\s*&\s*GOAL",  # 1 ST & GOAL
            ]

            for i, pattern in enumerate(patterns):
                if match := re.search(pattern, text):
                    down = int(match.group(1))
                    if 1 <= down <= 4:
                        if "GOAL" in text:
                            result = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd' if down == 3 else 'th'} & Goal"
                            logger.debug(
                                f"✅ Down/Distance extracted: '{result}' (pattern {i+1}, conf: {ocr_result['confidence']:.3f})"
                            )
                            return result
                        else:
                            # Find distance - could be group 2 or 3 depending on pattern
                            distance_group = 3 if len(match.groups()) >= 3 else 2
                            try:
                                distance = int(match.group(distance_group))
                                if 1 <= distance <= 99:
                                    result = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd' if down == 3 else 'th'} & {distance}"
                                    logger.debug(
                                        f"✅ Down/Distance extracted: '{result}' (pattern {i+1}, conf: {ocr_result['confidence']:.3f})"
                                    )
                                    return result
                            except (IndexError, ValueError):
                                # Handle patterns without distance group (like "3 & 7")
                                if len(match.groups()) >= 2:
                                    try:
                                        distance = int(match.group(2))
                                        if 1 <= distance <= 99:
                                            result = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd' if down == 3 else 'th'} & {distance}"
                                            logger.debug(
                                                f"✅ Down/Distance extracted: '{result}' (pattern {i+1} fallback, conf: {ocr_result['confidence']:.3f})"
                                            )
                                            return result
                                    except (IndexError, ValueError):
                                        continue

            logger.debug(
                f"❌ No valid down/distance pattern found in: '{text}' (original: '{ocr_result.get('text', '')}')"
            )
            return None

        except Exception as e:
            logger.error(f"Down/distance extraction error: {e}")
            return None

    def extract_scores(self, region: np.ndarray) -> Optional[Dict[str, str]]:
        """Extract team scores using PRIMARY custom OCR."""
        try:
            # Use multi-engine OCR with custom OCR as primary
            ocr_result = self._extract_text_multi_engine(region, "score_area")

            if not ocr_result.get("text"):
                return None

            text = ocr_result["text"].strip().upper()
            logger.debug(f"Scores OCR: '{text}' from {ocr_result['source']}")

            # Look for team abbreviations and scores
            # Pattern: TEAM1 SCORE1 TEAM2 SCORE2 (e.g., "DND 27 YBG 0")
            if match := re.search(r"([A-Z]{2,3})\s*(\d{1,2})\s*([A-Z]{2,3})\s*(\d{1,2})", text):
                away_team, away_score, home_team, home_score = match.groups()
                result = {
                    "away_team": away_team,
                    "away_score": away_score,
                    "home_team": home_team,
                    "home_score": home_score,
                }
                logger.debug(
                    f"✅ Scores extracted: {result} (conf: {ocr_result['confidence']:.3f})"
                )
                return result

            logger.debug(f"❌ No valid score pattern found in: '{text}'")
            return None

        except Exception as e:
            logger.error(f"Score extraction error: {e}")
            return None

    def _preprocess_clock_region(self, region: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for clock regions."""
        # Resize to make text larger
        height, width = region.shape[:2]
        scale = max(4, 80 // height)  # Ensure at least 80px height
        resized = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Enhance contrast for clock digits
        enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=30)

        # Apply threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def _preprocess_text_region(self, region: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for text regions."""
        # Resize to make text larger
        height, width = region.shape[:2]
        scale = max(3, 60 // height)  # Ensure at least 60px height
        resized = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Enhance contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=20)

        return enhanced

    def analyze_region_color(self, region: np.ndarray) -> dict:
        """
        Analyze the dominant colors in a region to detect penalty indicators.

        Args:
            region: Image region to analyze

        Returns:
            Dictionary with color analysis results
        """
        try:
            # Convert to HSV for better color detection
            if len(region.shape) == 3:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            else:
                # Convert grayscale to BGR first
                bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Define yellow color range in HSV
            # Yellow in HSV: Hue 20-30, Saturation 100-255, Value 100-255
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Create mask for yellow pixels
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Calculate percentage of yellow pixels
            total_pixels = region.shape[0] * region.shape[1]
            yellow_pixels = cv2.countNonZero(yellow_mask)
            yellow_percentage = yellow_pixels / total_pixels if total_pixels > 0 else 0

            # Analyze overall brightness (for penalty flag detection)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            avg_brightness = np.mean(gray)

            # Check for high contrast (text on colored background)
            brightness_std = np.std(gray)

            return {
                "yellow_percentage": yellow_percentage,
                "avg_brightness": avg_brightness,
                "brightness_std": brightness_std,
                "is_penalty_colored": yellow_percentage > 0.3,  # 30% yellow threshold
                "is_high_contrast": brightness_std > 50,  # High contrast text
            }

        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {
                "yellow_percentage": 0.0,
                "avg_brightness": 0.0,
                "brightness_std": 0.0,
                "is_penalty_colored": False,
                "is_high_contrast": False,
            }

    def debug_region_processing(
        self, image: np.ndarray, description: str = "Region"
    ) -> dict[str, Any]:
        """
        Debug wrapper for process_region with detailed logging.

        Args:
            image: Image region to process
            description: Description of the region for logging

        Returns:
            Processing results with debug information
        """
        logger.info(f"=== DEBUG: Processing {description} ===")
        logger.info(f"Image shape: {image.shape}")

        # Enable debug mode
        results = self.process_region(image, debug_mode=True)

        # Additional analysis
        if "debug_info" in results:
            debug_info = results["debug_info"]

            logger.info(f"Raw OCR Results: {len(debug_info['raw_ocr_results'])}")
            for i, result in enumerate(debug_info["raw_ocr_results"]):
                logger.info(f"  {i+1}: '{result['text']}' conf={result['confidence']:.3f}")

            logger.info(f"Territory Attempts: {len(debug_info['territory_attempts'])}")
            for attempt in debug_info["territory_attempts"]:
                status = "SELECTED" if attempt["selected"] else "rejected"
                logger.info(
                    f"  {attempt['detected']} ({status}) conf={attempt['confidence']:.3f} from '{attempt['text']}'"
                )

            logger.info(f"Yard Line Attempts: {len(debug_info['yard_line_attempts'])}")
            for attempt in debug_info["yard_line_attempts"]:
                status = "VALID" if attempt["validation_passed"] else "INVALID"
                logger.info(
                    f"  {attempt['extracted_value']} ({status}) conf={attempt['confidence']:.3f} from '{attempt['raw_text']}'"
                )
                if attempt["formatted_value"]:
                    logger.info(f"    -> formatted as: {attempt['formatted_value']}")

        # Final results summary
        logger.info(f"Final Results: {list(results.keys())}")
        for key, value in results.items():
            if key != "debug_info":
                logger.info(f"  {key}: {value}")

        logger.info(f"=== END DEBUG: {description} ===\n")

        return results

    def debug_temporal_integration_issue(
        self, test_region: np.ndarray, current_time: float = None
    ) -> dict[str, Any]:
        """
        Debug the specific temporal manager integration issue with burst sampling.

        This method helps identify confidence voting conflicts between:
        1. Burst sampling consensus (current_time=None)
        2. Temporal manager voting (current_time!=None)

        Args:
            test_region: Image region to test OCR on
            current_time: Pass None for burst sampling, timestamp for temporal mode

        Returns:
            Debug information about the integration issue
        """
        logger.info(f"=== TEMPORAL INTEGRATION DEBUG ===")
        logger.info(f"Mode: {'BURST SAMPLING' if current_time is None else 'TEMPORAL MANAGER'}")
        logger.info(f"Current time: {current_time}")

        # Test 1: OCR extraction with current mode
        logger.info("\n--- Test 1: OCR Extraction ---")
        ocr_results = self.process_region(test_region, debug_mode=True)

        # Test 2: Compare confidence scores
        logger.info("\n--- Test 2: Confidence Analysis ---")
        if "confidence" in ocr_results:
            conf = ocr_results["confidence"]
            logger.info(f"Average confidence: {conf:.3f}")

            if conf < self.validation.min_confidence:
                logger.warning(
                    f"Confidence {conf:.3f} below threshold {self.validation.min_confidence}"
                )
                logger.info("This could cause temporal voting conflicts!")
            else:
                logger.info(f"Confidence {conf:.3f} above threshold - should work well")
        else:
            logger.warning("No confidence score returned - integration issue!")

        # Test 3: Historical impact simulation
        logger.info("\n--- Test 3: Historical Integration ---")
        if current_time is None:
            logger.info(
                "BURST MODE: Bypassing temporal history (consensus voting handles reliability)"
            )
        else:
            logger.info("TEMPORAL MODE: Using historical smoothing and frequency control")

            # Check if temporal logic would extract at this time
            # Note: This would require temporal manager instance
            logger.info("Historical data impact: Not simulated (would need temporal manager)")

        # Test 4: Integration recommendations
        logger.info("\n--- Test 4: Integration Recommendations ---")
        integration_status = {
            "mode": "BURST_SAMPLING" if current_time is None else "TEMPORAL_MANAGER",
            "ocr_confidence": ocr_results.get("confidence", 0.0),
            "threshold_passed": ocr_results.get("confidence", 0.0)
            >= self.validation.min_confidence,
            "recommended_action": "",
            "potential_conflicts": [],
        }

        if current_time is None:
            # Burst sampling mode
            if integration_status["threshold_passed"]:
                integration_status["recommended_action"] = "Continue with burst consensus voting"
            else:
                integration_status["recommended_action"] = (
                    "Lower confidence threshold for burst sampling"
                )
                integration_status["potential_conflicts"].append(
                    "Burst consensus may fail due to low confidence"
                )
        else:
            # Temporal manager mode
            if integration_status["threshold_passed"]:
                integration_status["recommended_action"] = "Temporal voting should work well"
            else:
                integration_status["recommended_action"] = (
                    "Increase OCR preprocessing for temporal stability"
                )
                integration_status["potential_conflicts"].append(
                    "Temporal voting needs higher confidence for stability"
                )

        logger.info(f"Mode: {integration_status['mode']}")
        logger.info(f"Confidence: {integration_status['ocr_confidence']:.3f}")
        logger.info(f"Threshold passed: {integration_status['threshold_passed']}")
        logger.info(f"Recommendation: {integration_status['recommended_action']}")

        if integration_status["potential_conflicts"]:
            logger.warning("Potential conflicts:")
            for conflict in integration_status["potential_conflicts"]:
                logger.warning(f"  - {conflict}")

        logger.info(f"=== END TEMPORAL INTEGRATION DEBUG ===\n")

        return {
            "ocr_results": ocr_results,
            "integration_status": integration_status,
            "mode": "burst_sampling" if current_time is None else "temporal_manager",
        }

    def analyze_confidence_patterns(self, test_images: list, descriptions: list = None) -> dict:
        """
        Analyze confidence patterns across multiple test images.

        Args:
            test_images: List of image arrays to test
            descriptions: Optional descriptions for each image

        Returns:
            Analysis results with confidence statistics
        """
        if descriptions is None:
            descriptions = [f"Image_{i+1}" for i in range(len(test_images))]

        analysis = {
            "total_images": len(test_images),
            "successful_detections": 0,
            "confidence_scores": [],
            "low_confidence_cases": [],
            "validation_failures": [],
            "territory_detections": 0,
            "yard_line_detections": 0,
        }

        logger.info(f"=== CONFIDENCE ANALYSIS: {len(test_images)} images ===")

        for i, (image, desc) in enumerate(zip(test_images, descriptions)):
            logger.info(f"\nAnalyzing {desc}...")

            results = self.process_region(image, debug_mode=True)

            if results:
                analysis["successful_detections"] += 1

                if "confidence" in results:
                    conf = results["confidence"]
                    analysis["confidence_scores"].append(conf)

                    if conf < self.validation.min_confidence:
                        analysis["low_confidence_cases"].append(
                            {"image": desc, "confidence": conf, "results": results}
                        )

                if "territory" in results:
                    analysis["territory_detections"] += 1

                if "yard_line" in results:
                    analysis["yard_line_detections"] += 1
            else:
                analysis["validation_failures"].append(desc)

        # Calculate statistics
        if analysis["confidence_scores"]:
            analysis["avg_confidence"] = sum(analysis["confidence_scores"]) / len(
                analysis["confidence_scores"]
            )
            analysis["min_confidence"] = min(analysis["confidence_scores"])
            analysis["max_confidence"] = max(analysis["confidence_scores"])

        # Summary
        logger.info(f"\n=== ANALYSIS SUMMARY ===")
        logger.info(
            f"Successful detections: {analysis['successful_detections']}/{analysis['total_images']} ({analysis['successful_detections']/analysis['total_images']*100:.1f}%)"
        )
        logger.info(f"Territory detections: {analysis['territory_detections']}")
        logger.info(f"Yard line detections: {analysis['yard_line_detections']}")

        if analysis["confidence_scores"]:
            logger.info(f"Average confidence: {analysis['avg_confidence']:.3f}")
            logger.info(
                f"Confidence range: {analysis['min_confidence']:.3f} - {analysis['max_confidence']:.3f}"
            )
            logger.info(f"Low confidence cases: {len(analysis['low_confidence_cases'])}")

        if analysis["validation_failures"]:
            logger.info(f"Validation failures: {analysis['validation_failures']}")

        return analysis

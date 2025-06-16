#!/usr/bin/env python3
"""
Enhanced OCR System for SpygateAI
Significantly improves text extraction accuracy through:
- Advanced preprocessing pipelines
- Multi-engine OCR (EasyOCR + Tesseract)
- Sport-specific text validation
- Intelligent confidence scoring
- Game-specific HUD text patterns
- Production-grade error handling and recovery
"""

import copy
import logging
import re
import sys
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# OCR Engine availability flags
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

# Configure logging
logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Base exception for OCR-related errors."""

    pass


class OCREngineError(OCRError):
    """Raised when OCR engine initialization or processing fails."""

    pass


class ValidationError(OCRError):
    """Raised when text validation fails critically."""

    pass


class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails."""

    pass


class EngineStatus(Enum):
    """Status of OCR engines."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass
class OCRResult:
    """Structured OCR result with comprehensive metadata."""

    text: str = ""
    confidence: float = 0.0
    engine: str = "none"
    preprocessing: str = "none"
    validation_score: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False
    raw_results: list[dict] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Check if OCR was successful."""
        return bool(self.text) and self.confidence > 0.0 and not self.error

    @property
    def final_confidence(self) -> float:
        """Get weighted confidence score including validation."""
        if self.validation_score > 0:
            return (self.confidence * 0.7) + (self.validation_score * 0.3)
        return self.confidence


class EnhancedOCRSystem:
    """Complete enhanced OCR system building on optimized preprocessing."""

    def __init__(self, optimized_paddle_ocr):
        self.primary_ocr = optimized_paddle_ocr
        self.temporal_history = deque(maxlen=5)
        self.previous_state = None

        # Try to initialize secondary OCR engines
        self.secondary_engines = self._init_secondary_engines()

        print("🚀 Enhanced OCR System initialized")
        print(f"   Primary: Optimized PaddleOCR (0.939 baseline)")
        print(f"   Secondary engines: {len(self.secondary_engines)}")

    def _init_secondary_engines(self):
        """Initialize secondary OCR engines for ensemble."""
        engines = []

        # Try Tesseract
        try:
            import pytesseract

            engines.append(
                {
                    "name": "tesseract",
                    "config": "--psm 8 -c tessedit_char_whitelist=0123456789stndrdthGoal&: ",
                    "weight": 0.25,
                }
            )
        except ImportError:
            pass

        # Try EasyOCR
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=True)
            engines.append({"name": "easyocr", "reader": reader, "weight": 0.15})
        except ImportError:
            pass

        return engines

    def extract_enhanced(self, processed_image, frame_number):
        """Enhanced extraction with ensemble + temporal + validation."""

        # 1. Primary OCR (optimized preprocessing)
        primary_result = self._extract_primary(processed_image)

        # 2. Secondary OCR engines (if available)
        secondary_results = self._extract_secondary(processed_image)

        # 3. Ensemble voting
        ensemble_result = self._ensemble_vote(primary_result, secondary_results)

        # 4. Temporal filtering
        filtered_result = self._temporal_filter(ensemble_result, frame_number)

        # 5. Game logic validation
        validated_result = self._game_logic_validate(filtered_result)

        # 6. Calculate final confidence
        final_result = self._calculate_final_confidence(validated_result)

        return final_result

    def _extract_primary(self, processed_image):
        """Extract using optimized PaddleOCR (0.939 baseline)."""
        try:
            paddle_result = self.primary_ocr.ocr(processed_image, cls=True)
            if paddle_result and paddle_result[0]:
                text = paddle_result[0][0][1][0]
                confidence = paddle_result[0][0][1][1]

                parsed = self._parse_down_distance(text)
                if parsed:
                    parsed.update(
                        {
                            "engine": "paddle_optimized",
                            "confidence": confidence,
                            "text": text,
                            "weight": 0.6,
                        }
                    )
                    return parsed
        except Exception as e:
            print(f"⚠️ Primary OCR error: {e}")

        return None

    def _extract_secondary(self, processed_image):
        """Extract using secondary OCR engines."""
        results = []

        for engine in self.secondary_engines:
            try:
                if engine["name"] == "tesseract":
                    import pytesseract

                    text = pytesseract.image_to_string(
                        processed_image, config=engine["config"]
                    ).strip()

                    if text:
                        parsed = self._parse_down_distance(text)
                        if parsed:
                            parsed.update(
                                {
                                    "engine": "tesseract",
                                    "confidence": 0.8,
                                    "text": text,
                                    "weight": engine["weight"],
                                }
                            )
                            results.append(parsed)

                elif engine["name"] == "easyocr":
                    easy_results = engine["reader"].readtext(processed_image)
                    if easy_results:
                        best_result = max(easy_results, key=lambda x: x[2])
                        parsed = self._parse_down_distance(best_result[1])
                        if parsed:
                            parsed.update(
                                {
                                    "engine": "easyocr",
                                    "confidence": best_result[2],
                                    "text": best_result[1],
                                    "weight": engine["weight"],
                                }
                            )
                            results.append(parsed)

            except Exception as e:
                print(f"⚠️ {engine['name']} error: {e}")

        return results

    def _ensemble_vote(self, primary_result, secondary_results):
        """Weighted ensemble voting across OCR engines."""

        if not primary_result and not secondary_results:
            return None

        all_results = []
        if primary_result:
            all_results.append(primary_result)
        all_results.extend(secondary_results)

        if len(all_results) == 1:
            return all_results[0]

        # Vote on down and distance separately
        down_votes = {}
        distance_votes = {}

        for result in all_results:
            down = result.get("down")
            distance = result.get("distance")
            weight = result.get("weight", 1.0)

            if down:
                down_votes[down] = down_votes.get(down, 0) + weight
            if distance is not None:
                distance_votes[distance] = distance_votes.get(distance, 0) + weight

        # Get consensus
        consensus_down = max(down_votes.items(), key=lambda x: x[1])[0] if down_votes else None
        consensus_distance = (
            max(distance_votes.items(), key=lambda x: x[1])[0] if distance_votes else None
        )

        # Calculate ensemble confidence
        total_weight = sum(r.get("weight", 1.0) for r in all_results)
        down_confidence = down_votes.get(consensus_down, 0) / total_weight if consensus_down else 0
        distance_confidence = (
            distance_votes.get(consensus_distance, 0) / total_weight if consensus_distance else 0
        )

        ensemble_confidence = (down_confidence + distance_confidence) / 2

        return {
            "engine": "ensemble",
            "down": consensus_down,
            "distance": consensus_distance,
            "confidence": ensemble_confidence,
            "contributing_engines": [r["engine"] for r in all_results],
            "ensemble_voting": True,
        }

    def _temporal_filter(self, ocr_result, frame_number):
        """Apply temporal consistency filtering."""

        if not ocr_result:
            return ocr_result

        current_down = ocr_result.get("down")
        current_distance = ocr_result.get("distance")

        # Store in temporal history
        self.temporal_history.append(
            {
                "frame": frame_number,
                "down": current_down,
                "distance": current_distance,
                "confidence": ocr_result.get("confidence", 0),
            }
        )

        # Need at least 3 frames for filtering
        if len(self.temporal_history) < 3:
            return ocr_result

        # Check consistency with recent frames
        recent_downs = [r["down"] for r in self.temporal_history if r["down"] is not None]
        recent_distances = [
            r["distance"] for r in self.temporal_history if r["distance"] is not None
        ]

        if not recent_downs:
            return ocr_result

        # Count occurrences
        down_counter = Counter(recent_downs)
        distance_counter = Counter(recent_distances)

        # Get most common values
        most_common_down = down_counter.most_common(1)[0]
        most_common_distance = distance_counter.most_common(1)[0] if recent_distances else (None, 0)

        # Calculate consistency scores
        down_consistency = most_common_down[1] / len(recent_downs)
        distance_consistency = (
            most_common_distance[1] / len(recent_distances) if recent_distances else 0
        )

        # Apply temporal correction if needed
        temporal_corrected = False
        if current_down != most_common_down[0] and down_consistency >= 0.6:
            print(
                f"🔄 Temporal correction: {current_down} → {most_common_down[0]} (consistency: {down_consistency:.2f})"
            )
            ocr_result["down"] = most_common_down[0]
            temporal_corrected = True

        if current_distance != most_common_distance[0] and distance_consistency >= 0.6:
            print(
                f"🔄 Temporal correction: {current_distance} → {most_common_distance[0]} (consistency: {distance_consistency:.2f})"
            )
            ocr_result["distance"] = most_common_distance[0]
            temporal_corrected = True

        # Add temporal metadata
        ocr_result["temporal_corrected"] = temporal_corrected
        ocr_result["temporal_consistency"] = (down_consistency + distance_consistency) / 2

        return ocr_result

    def _game_logic_validate(self, ocr_result):
        """Apply Madden game logic validation."""

        if not ocr_result:
            return ocr_result

        down = ocr_result.get("down")
        distance = ocr_result.get("distance")

        validation_score = 1.0
        validation_notes = []

        # Rule 1: Valid ranges
        if down and not (1 <= down <= 4):
            validation_score *= 0.1
            validation_notes.append(f"Invalid down: {down}")

        if distance is not None and not (0 <= distance <= 99):
            validation_score *= 0.1
            validation_notes.append(f"Invalid distance: {distance}")

        # Rule 2: Down progression logic
        if self.previous_state and down:
            prev_down = self.previous_state.get("down")
            if prev_down:
                valid_transitions = [prev_down, prev_down + 1, 1]
                if down not in valid_transitions:
                    validation_score *= 0.3
                    validation_notes.append(f"Suspicious transition: {prev_down} → {down}")

        # Rule 3: Common pattern boosts
        if down and distance is not None:
            common_patterns = {
                (1, 10): 1.2,  # Very common
                (2, 7): 1.1,  # After 3-yard gain
                (3, 1): 1.1,  # Short yardage
                (4, 1): 1.05,  # 4th down conversion
            }

            pattern = (down, distance)
            if pattern in common_patterns:
                validation_score *= common_patterns[pattern]
                validation_notes.append(f"Common pattern: {pattern}")

        # Apply validation
        original_confidence = ocr_result.get("confidence", 0)
        validated_confidence = original_confidence * validation_score

        ocr_result["validation_score"] = validation_score
        ocr_result["validated_confidence"] = validated_confidence
        ocr_result["validation_notes"] = validation_notes

        if validation_notes:
            print(f"🎮 Game logic: {validation_notes}")

        # Update state
        self.previous_state = {"down": down, "distance": distance}

        return ocr_result

    def _calculate_final_confidence(self, result):
        """Calculate final confidence score."""

        if not result:
            return result

        base_confidence = result.get("confidence", 0)
        temporal_consistency = result.get("temporal_consistency", 0)
        validation_score = result.get("validation_score", 1.0)

        # Weighted final confidence
        final_confidence = (
            base_confidence * 0.5
            + temporal_consistency * 0.3  # Base OCR
            + validation_score * 0.2  # Temporal consistency  # Game logic
        )

        result["final_confidence"] = min(1.0, final_confidence)
        result["enhancement_applied"] = True

        return result

    def _parse_down_distance(self, text):
        """Parse down and distance from OCR text."""
        import re

        # Pattern for "1st & 10", "3rd & Goal", etc.
        pattern = re.compile(r"(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+|Goal|goal)", re.IGNORECASE)
        match = pattern.search(text)

        if match:
            try:
                down = int(match.group(1))
                distance_str = match.group(2).lower()
                distance = 0 if distance_str == "goal" else int(distance_str)

                if 1 <= down <= 4 and 0 <= distance <= 30:
                    return {"down": down, "distance": distance}
            except ValueError:
                pass

        return None

    def enhance_image_for_ocr(
        self, image: np.ndarray, enhancement_type: str = "adaptive"
    ) -> list[np.ndarray]:
        """Enhance image with multiple preprocessing strategies for OCR."""
        try:
            # Validate and sanitize input image
            if image is None or image.size == 0:
                logger.error("Critical error in image enhancement: Input image is empty or None")
                raise ImageProcessingError("Input image is empty or None")

            # Handle corrupted or invalid image data
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                logger.warning("Image contains NaN or infinite values, cleaning...")
                image = np.nan_to_num(image, nan=127, posinf=255, neginf=0)

            # Ensure proper data type and range
            if image.dtype != np.uint8:
                logger.warning(f"Converting image from {image.dtype} to uint8")
                image = np.clip(image, 0, 255).astype(np.uint8)

            # Validate image shape
            if len(image.shape) not in [2, 3] or image.shape[0] == 0 or image.shape[1] == 0:
                raise ImageProcessingError(f"Invalid image shape: {image.shape}")

            # Handle corrupted or invalid data types
            try:
                # Convert to proper format if needed
                if image.dtype not in [np.uint8, np.float32, np.float64]:
                    # Try to convert to uint8
                    if np.issubdtype(image.dtype, np.integer):
                        image = np.clip(image, 0, 255).astype(np.uint8)
                    else:
                        # For float types, normalize to 0-255 range
                        image_min, image_max = np.nanmin(image), np.nanmax(image)
                        if (
                            np.isfinite(image_min)
                            and np.isfinite(image_max)
                            and image_max > image_min
                        ):
                            image = ((image - image_min) / (image_max - image_min) * 255).astype(
                                np.uint8
                            )
                        else:
                            raise ImageProcessingError(
                                "Image contains invalid values (NaN/Inf) that cannot be normalized"
                            )

                # Handle special values
                if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                    # Replace NaN and Inf with valid values
                    image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)

                # Ensure values are in valid range
                image = np.clip(image, 0, 255).astype(np.uint8)

            except Exception as e:
                logger.error(f"Failed to sanitize image data: {e}")
                raise ImageProcessingError(f"Cannot process corrupted image data: {e}")

            # Ensure image has valid dimensions
            if len(image.shape) not in [2, 3]:
                logger.error(f"Invalid image dimensions: {image.shape}")
                raise ImageProcessingError(f"Unsupported image dimensions: {image.shape}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    logger.warning(f"Color conversion failed, using alternative method: {e}")
                    # Alternative grayscale conversion for edge cases
                    if image.shape[2] >= 3:
                        gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
                    else:
                        gray = image[:, :, 0]  # Use first channel
            else:
                gray = image

            # Quick quality check
            if gray.std() < 5:  # Very low contrast
                logger.warning("Very low contrast image detected")

            # Return simple grayscale for fallback cases or specific request
            if enhancement_type == "simple":
                return [gray]

            enhanced_images = []

            # Strategy 1: Original grayscale
            enhanced_images.append(gray)

            # Strategy 2: Contrast enhancement
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast_enhanced = clahe.apply(gray)
                enhanced_images.append(contrast_enhanced)
            except Exception as e:
                logger.warning(f"Contrast enhancement failed: {e}")

            # Strategy 3: Adaptive thresholding
            try:
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                enhanced_images.append(adaptive_thresh)
            except Exception as e:
                logger.warning(f"Adaptive thresholding failed: {e}")

            # Strategy 4: Otsu's thresholding
            try:
                _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                enhanced_images.append(otsu_thresh)
            except Exception as e:
                logger.warning(f"Otsu thresholding failed: {e}")

            # Strategy 5: Morphological operations
            try:
                kernel = np.ones((2, 2), np.uint8)
                morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                enhanced_images.append(morphed)
            except Exception as e:
                logger.warning(f"Morphological operations failed: {e}")

            return enhanced_images if enhanced_images else [gray]

        except ImageProcessingError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"Critical error in image enhancement: {e}")
            # Last resort - try to return basic grayscale conversion
            try:
                if image is not None and image.size > 0:
                    return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
                else:
                    raise ImageProcessingError("Cannot process image: Input is empty or None")
            except:
                raise ImageProcessingError(f"Cannot process image: {e}")

    def extract_text_multi_engine(
        self, image: np.ndarray, text_type: Optional[str] = None
    ) -> list[OCRResult]:
        """
        Extract text using multiple OCR engines and preprocessing strategies with comprehensive error handling.

        Args:
            image: Input image
            text_type: Expected text type for validation (optional)

        Returns:
            List of OCR results sorted by confidence
        """
        results = []
        self.performance_stats["total_extractions"] += 1

        try:
            # Get multiple preprocessed versions
            enhanced_images = self.enhance_image_for_ocr(image, "all")

            for i, processed_img in enumerate(enhanced_images):
                preprocessing_method = [
                    "original",
                    "binary",
                    "adaptive",
                    "otsu",
                    "enhanced",
                    "denoised",
                    "morphology",
                ][i]

                # Try EasyOCR
                if self.engine_status["easyocr"] == EngineStatus.AVAILABLE:
                    for attempt in range(self.max_retries):
                        easyocr_result = self._extract_with_easyocr(
                            processed_img, preprocessing_method
                        )
                        if easyocr_result and easyocr_result.is_successful:
                            results.append(easyocr_result)
                            break
                        elif attempt == self.max_retries - 1:
                            self.performance_stats["engine_failures"]["easyocr"] += 1

                # Try Tesseract
                if self.engine_status["tesseract"] == EngineStatus.AVAILABLE:
                    for attempt in range(self.max_retries):
                        tesseract_result = self._extract_with_tesseract(
                            processed_img, preprocessing_method, text_type
                        )
                        if tesseract_result and tesseract_result.is_successful:
                            results.append(tesseract_result)
                            break
                        elif attempt == self.max_retries - 1:
                            self.performance_stats["engine_failures"]["tesseract"] += 1

            # Sort by final confidence
            results = sorted(results, key=lambda x: x.final_confidence, reverse=True)

            # Apply text validation if type specified
            if text_type and text_type in self.text_patterns:
                validated_results = []
                for result in results:
                    try:
                        validation_score = self._validate_text(result.text, text_type)
                        result.validation_score = validation_score
                        validated_results.append(result)
                    except ValidationError as e:
                        logger.warning(f"Validation failed for '{result.text}': {e}")
                        result.validation_score = 0.0
                        validated_results.append(result)

                results = sorted(validated_results, key=lambda x: x.final_confidence, reverse=True)

            if results:
                self.performance_stats["successful_extractions"] += 1

            return results

        except Exception as e:
            logger.error(f"Critical error in multi-engine extraction: {e}")
            if self.debug:
                traceback.print_exc()
            return [OCRResult(error=f"Multi-engine extraction failed: {e}")]

    def _extract_with_easyocr(
        self, image: np.ndarray, preprocessing_method: str
    ) -> Optional[OCRResult]:
        """Extract text using EasyOCR with comprehensive error handling."""
        if self.engine_status["easyocr"] != EngineStatus.AVAILABLE:
            return None

        try:
            import time

            start_time = time.time()

            ocr_results = self.easyocr_reader.readtext(image, detail=1, paragraph=False)
            processing_time = time.time() - start_time

            if not ocr_results:
                return OCRResult(
                    engine="EasyOCR",
                    preprocessing=preprocessing_method,
                    processing_time=processing_time,
                    error="No text detected",
                )

            # Combine all text and calculate average confidence
            texts = []
            confidences = []

            for bbox, text, confidence in ocr_results:
                if confidence > 0.1:  # Very low threshold for inclusion
                    texts.append(text.strip())
                    confidences.append(confidence)

            if not texts:
                return OCRResult(
                    engine="EasyOCR",
                    preprocessing=preprocessing_method,
                    processing_time=processing_time,
                    error="No high-confidence text found",
                )

            combined_text = " ".join(texts)
            avg_confidence = np.mean(confidences)

            # Apply text corrections
            corrected_text = self._apply_corrections(combined_text)

            return OCRResult(
                text=corrected_text,
                confidence=avg_confidence,
                engine="EasyOCR",
                preprocessing=preprocessing_method,
                processing_time=processing_time,
                raw_results=[
                    {
                        "bbox": bbox.tolist() if hasattr(bbox, "tolist") else bbox,
                        "text": text,
                        "confidence": conf,
                    }
                    for bbox, text, conf in ocr_results
                ],
            )

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            if self.debug:
                traceback.print_exc()

            # Mark engine as failed if consistent failures
            self.performance_stats["engine_failures"]["easyocr"] += 1
            if self.performance_stats["engine_failures"]["easyocr"] > 10:
                self.engine_status["easyocr"] = EngineStatus.FALLBACK
                logger.warning("EasyOCR marked as fallback due to repeated failures")

            return OCRResult(
                engine="EasyOCR", preprocessing=preprocessing_method, error=f"EasyOCR failed: {e}"
            )

    def _extract_with_tesseract(
        self, image: np.ndarray, preprocessing_method: str, text_type: Optional[str] = None
    ) -> Optional[OCRResult]:
        """Extract text using Tesseract with optimized configurations and error handling."""
        if self.engine_status["tesseract"] != EngineStatus.AVAILABLE:
            return None

        try:
            import time

            start_time = time.time()

            # Configure Tesseract based on text type
            configs = self._get_tesseract_configs(text_type)

            best_result = None
            best_confidence = 0

            for config_name, config in configs.items():
                try:
                    # Extract text
                    text = pytesseract.image_to_string(image, config=config).strip()

                    if not text:
                        continue

                    # Get confidence data
                    data = pytesseract.image_to_data(
                        image, config=config, output_type=pytesseract.Output.DICT
                    )
                    confidences = [
                        float(conf) for conf in data["conf"] if conf != "-1" and float(conf) > 0
                    ]

                    if not confidences:
                        continue

                    avg_confidence = np.mean(confidences)

                    # Apply corrections
                    corrected_text = self._apply_corrections(text)

                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        processing_time = time.time() - start_time
                        best_result = OCRResult(
                            text=corrected_text,
                            confidence=avg_confidence / 100.0,  # Normalize to 0-1
                            engine="Tesseract",
                            preprocessing=preprocessing_method,
                            processing_time=processing_time,
                            raw_results=[
                                {
                                    "config": config_name,
                                    "raw_text": text,
                                    "confidences": confidences,
                                }
                            ],
                        )

                except Exception as e:
                    logger.warning(f"Tesseract config {config_name} failed: {e}")
                    continue

            if best_result:
                return best_result
            else:
                return OCRResult(
                    engine="Tesseract",
                    preprocessing=preprocessing_method,
                    processing_time=time.time() - start_time,
                    error="No valid text extracted with any configuration",
                )

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            if self.debug:
                traceback.print_exc()

            # Mark engine as failed if consistent failures
            self.performance_stats["engine_failures"]["tesseract"] += 1
            if self.performance_stats["engine_failures"]["tesseract"] > 10:
                self.engine_status["tesseract"] = EngineStatus.FALLBACK
                logger.warning("Tesseract marked as fallback due to repeated failures")

            return OCRResult(
                engine="Tesseract",
                preprocessing=preprocessing_method,
                error=f"Tesseract failed: {e}",
            )

    def _get_tesseract_configs(self, text_type: Optional[str] = None) -> dict[str, str]:
        """Get optimized Tesseract configurations for different text types."""

        base_configs = {
            "default": "--oem 3 --psm 8",
            "single_line": "--oem 3 --psm 7",
            "single_word": "--oem 3 --psm 8",
            "digits_only": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
        }

        if text_type == "down_distance":
            base_configs.update(
                {
                    "down_distance": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789THNDRS&",
                    "down_distance_simple": "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789&",
                }
            )
        elif text_type == "score":
            base_configs.update(
                {
                    "score": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-:",
                    "score_simple": "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
                }
            )
        elif text_type == "time":
            base_configs.update(
                {
                    "time": "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:.",
                }
            )
        elif text_type == "team_names":
            base_configs.update(
                {
                    "team_names": "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                }
            )

        return base_configs

    def _apply_corrections(self, text: str) -> str:
        """Apply common OCR error corrections with validation."""
        try:
            corrected = text

            # Apply character-level corrections
            for wrong, right in self.ocr_corrections.items():
                corrected = corrected.replace(wrong, right)

            # Clean up whitespace
            corrected = re.sub(r"\s+", " ", corrected).strip()

            return corrected

        except Exception as e:
            logger.warning(f"Text correction failed: {e}")
            return text  # Return original if correction fails

    def _validate_text(self, text: str, text_type: str) -> float:
        """
        Validate extracted text against expected patterns.

        Returns:
            Validation score between 0.0 and 1.0
        """
        try:
            if text_type not in self.text_patterns:
                return 0.5  # Neutral score if no patterns defined

            patterns = self.text_patterns[text_type]

            # Check if text matches any pattern
            for pattern in patterns:
                if pattern.match(text.strip()):
                    return 1.0  # Perfect match

            # Partial scoring based on similarity
            cleaned_text = re.sub(r"[^\w\d&:-]", "", text.upper())

            if text_type == "down_distance":
                # Check for down (1-4) and distance (0-99)
                if re.search(r"[1-4]", cleaned_text) and re.search(r"\d{1,2}", cleaned_text):
                    return 0.7
            elif text_type == "score":
                # Check for two numbers
                if len(re.findall(r"\d+", cleaned_text)) >= 2:
                    return 0.7
            elif text_type == "time":
                # Check for time format
                if re.search(r"\d{1,2}[\:\.]?\d{2}", cleaned_text):
                    return 0.7

            return 0.2  # Low score but not zero

        except Exception as e:
            logger.warning(f"Text validation failed for '{text}' as {text_type}: {e}")
            return 0.0

    def extract_text_from_region(
        self, frame: np.ndarray, bbox: list[int], text_type: Optional[str] = None, padding: int = 5
    ) -> OCRResult:
        """
        Extract text from a specific region with enhanced accuracy and comprehensive error handling.

        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            text_type: Expected text type for validation
            padding: Padding around the region

        Returns:
            Best extraction result with metadata
        """
        try:
            # Validate inputs
            if frame is None or frame.size == 0:
                return OCRResult(error="Input frame is empty or None")

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return OCRResult(error=f"Invalid bbox format: {bbox}")

            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Validate and clamp coordinates
            if x1 >= x2 or y1 >= y2:
                return OCRResult(error=f"Invalid bbox coordinates: {bbox}")

            # Apply padding and ensure valid coordinates
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            # Final validation
            if x1 >= x2 or y1 >= y2:
                return OCRResult(error=f"Invalid bbox after padding: [{x1}, {y1}, {x2}, {y2}]")

            # Extract region
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return OCRResult(error="Empty region after extraction")

            # Extract text using multiple engines
            results = self.extract_text_multi_engine(roi, text_type)

            if not results:
                return OCRResult(error="No OCR results from any engine")

            # Filter successful results
            successful_results = [r for r in results if r.is_successful]

            if not successful_results:
                # Use fallback if no successful results
                if self.fallback_enabled:
                    self.performance_stats["fallback_uses"] += 1
                    fallback_result = self._fallback_text_extraction(roi, text_type)
                    fallback_result.fallback_used = True
                    return fallback_result
                else:
                    # Return best failed result with error info
                    best_failed = max(results, key=lambda x: x.confidence)
                    best_failed.error = (
                        f"All engines failed. Best confidence: {best_failed.confidence:.3f}"
                    )
                    return best_failed

            # Return best successful result
            best_result = successful_results[0]
            best_result.raw_results = [r.__dict__ for r in results[:3]]  # Keep top 3 for comparison

            if self.debug:
                print(f"🔍 Text extraction for {text_type or 'unknown'}:")
                print(
                    f"   • Best: '{best_result.text}' (conf: {best_result.final_confidence:.3f}, {best_result.engine})"
                )
                for i, result in enumerate(successful_results[1:4], 1):
                    print(
                        f"   • Alt {i}: '{result.text}' (conf: {result.final_confidence:.3f}, {result.engine})"
                    )

            return best_result

        except Exception as e:
            logger.error(f"Critical error in text extraction from region: {e}")
            if self.debug:
                traceback.print_exc()
            return OCRResult(error=f"Critical extraction failure: {e}")

    def _fallback_text_extraction(
        self, roi: np.ndarray, text_type: Optional[str] = None
    ) -> OCRResult:
        """
        Fallback text extraction using simple methods when primary engines fail.
        """
        try:
            logger.info("Using fallback text extraction")

            # Simple threshold + basic pattern matching
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Template matching for common patterns
            if text_type == "down_distance":
                # Look for digit patterns
                digit_patterns = ["1ST", "2ND", "3RD", "4TH"]
                for pattern in digit_patterns:
                    if self._simple_template_match(binary, pattern):
                        return OCRResult(
                            text=pattern + " & 10",  # Default assumption
                            confidence=0.3,
                            engine="fallback",
                            preprocessing="template_match",
                            fallback_used=True,
                        )

            # Generic fallback - return empty result
            return OCRResult(
                text="",
                confidence=0.0,
                engine="fallback",
                preprocessing="simple_threshold",
                fallback_used=True,
                error="Fallback extraction found no text",
            )

        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return OCRResult(
                error=f"Fallback extraction failed: {e}", engine="fallback", fallback_used=True
            )

    def _simple_template_match(self, image: np.ndarray, pattern: str) -> bool:
        """Simple template matching for fallback extraction."""
        try:
            # This is a placeholder for actual template matching
            # In a real implementation, you'd have pre-created templates
            return False
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
            return False

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()

        if stats["total_extractions"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_extractions"]
        else:
            stats["success_rate"] = 0.0

        stats["engine_status"] = {k: v.value for k, v in self.engine_status.items()}

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "engine_failures": {"easyocr": 0, "tesseract": 0},
            "fallback_uses": 0,
            "average_confidence": 0.0,
        }
        logger.info("Performance statistics reset")

    def health_check(self) -> dict[str, Any]:
        """Perform a comprehensive health check of the OCR system."""
        health = {"status": "healthy", "engines": {}, "issues": [], "recommendations": []}

        # Check engine status
        available_engines = 0
        for engine, status in self.engine_status.items():
            health["engines"][engine] = {
                "status": status.value,
                "failures": self.performance_stats["engine_failures"].get(engine, 0),
            }

            if status == EngineStatus.AVAILABLE:
                available_engines += 1
            elif status == EngineStatus.FAILED:
                health["issues"].append(f"{engine} engine failed")

        # Overall health assessment
        if available_engines == 0:
            health["status"] = "critical"
            health["issues"].append("No OCR engines available")
            health["recommendations"].append("Install EasyOCR or Tesseract")
        elif available_engines == 1:
            health["status"] = "warning"
            health["recommendations"].append("Install additional OCR engine for redundancy")

        # Performance assessment
        stats = self.get_performance_stats()
        if stats["total_extractions"] > 0:
            if stats["success_rate"] < 0.5:
                health["status"] = "warning" if health["status"] == "healthy" else health["status"]
                health["issues"].append(f"Low success rate: {stats['success_rate']:.1%}")

            if stats["fallback_uses"] > stats["total_extractions"] * 0.3:
                health["issues"].append("High fallback usage - primary engines may be failing")

        health["performance"] = stats

        return health


if __name__ == "__main__":
    # Basic test
    print("🧪 Enhanced OCR System Test")

    try:
        ocr = EnhancedOCRSystem(debug=True)
        print(f"✅ Initialization successful")

        # Health check
        health = ocr.health_check()
        print(f"🏥 Health Status: {health['status']}")

        if health["issues"]:
            print("⚠️ Issues:")
            for issue in health["issues"]:
                print(f"   • {issue}")

        if health["recommendations"]:
            print("💡 Recommendations:")
            for rec in health["recommendations"]:
                print(f"   • {rec}")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()

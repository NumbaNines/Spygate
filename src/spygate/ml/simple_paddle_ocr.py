"""
Simple PaddleOCR Wrapper for SpygateAI
Provides a clean interface without complex dependencies
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SimplePaddleOCRWrapper:
    """
    Simple wrapper that provides the interface expected by enhanced_game_analyzer.
    Uses PaddleOCR with optimal preprocessing parameters.
    """

    def __init__(self):
        """Initialize the PaddleOCR system."""
        try:
            # Try to import PaddleOCR
            try:
                from paddleocr import PaddleOCR

                # Start with CPU mode to avoid CUDA issues
                logger.info("🔄 Initializing PaddleOCR with CPU mode...")
                self.paddle = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,  # Use CPU to avoid CUDA issues
                    show_log=False,
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    rec_batch_num=6,
                    max_text_length=25,
                    use_space_char=True,
                    drop_score=0.3,
                )
                self.ocr_engine = "paddleocr"
                logger.info("✅ PaddleOCR initialized successfully with CPU")
            except ImportError:
                # Fallback to EasyOCR if PaddleOCR not available
                try:
                    import easyocr

                    # Try GPU first, fallback to CPU if issues
                    try:
                        self.paddle = easyocr.Reader(["en"], gpu=True)
                        logger.info("✅ EasyOCR initialized with GPU")
                    except Exception as gpu_error:
                        logger.warning(f"⚠️ EasyOCR GPU failed: {gpu_error}")
                        self.paddle = easyocr.Reader(["en"], gpu=False)
                        logger.info("✅ EasyOCR initialized with CPU")

                    self.ocr_engine = "easyocr"
                    logger.warning("⚠️ PaddleOCR not available, using EasyOCR fallback")
                except ImportError:
                    # Final fallback to Tesseract
                    import pytesseract

                    self.paddle = None
                    self.ocr_engine = "tesseract"
                    logger.warning(
                        "⚠️ Neither PaddleOCR nor EasyOCR available, using Tesseract fallback"
                    )

            # Frame counter for temporal filtering
            self.frame_counter = 0

        except Exception as e:
            logger.error(f"Failed to initialize OCR system: {e}")
            raise

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the optimal preprocessing pipeline from the 20K parameter sweep.

        Optimal parameters (Score: 0.939):
        - Scale: 3.5x (LANCZOS4)
        - CLAHE: clip=1.0, grid=(4,4)
        - Blur: (3,3) Gaussian
        - Threshold: adaptive_mean, block=13, C=3
        - Morphological: (3,3) closing kernel
        - Gamma: 0.8
        - Sharpening: off
        """
        try:
            # Store original dimensions
            original_height, original_width = image.shape[:2]

            # Stage 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Stage 2: Scale with LANCZOS4 (3.5x)
            scale_factor = 3.5
            new_height = int(gray.shape[0] * scale_factor)
            new_width = int(gray.shape[1] * scale_factor)
            scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Stage 3: CLAHE (clip=1.0, grid=(4,4))
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
            clahe_applied = clahe.apply(scaled)

            # Stage 4: Gamma correction (0.8)
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
                "uint8"
            )
            gamma_corrected = cv2.LUT(clahe_applied, table)

            # Stage 5: Gaussian blur (3,3)
            blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

            # Stage 6: Adaptive thresholding (mean, block=13, C=3)
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 3
            )

            # Stage 7: Morphological closing (3,3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

            # Stage 8: No sharpening (optimal setting)

            # For PaddleOCR, convert back to BGR
            if self.ocr_engine == "paddleocr":
                processed = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
            else:
                processed = morphed

            return processed

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return image

    def _extract_text_from_results(self, results) -> str:
        """Extract text from OCR results based on engine type."""
        try:
            if not results:
                return ""

            if self.ocr_engine == "paddleocr":
                # PaddleOCR format: [[[bbox], (text, confidence)], ...]
                texts = []
                for line in results:
                    if line and len(line) > 1:
                        text, confidence = line[1]
                        if confidence > 0.3:  # Confidence threshold
                            texts.append(text)
                return " ".join(texts)

            elif self.ocr_engine == "easyocr":
                # EasyOCR format: [(bbox, text, confidence), ...]
                texts = []
                for result in results:
                    if len(result) >= 3:
                        bbox, text, confidence = result
                        if confidence > 0.3:  # Confidence threshold
                            texts.append(text)
                return " ".join(texts)

            else:
                # Tesseract - results is already a string
                return str(results).strip()

        except Exception as e:
            logger.error(f"Error extracting text from results: {e}")
            return ""

    def extract_text(self, region: np.ndarray) -> str:
        """
        Extract text from a region.

        Args:
            region: Image region to extract text from

        Returns:
            Extracted text string
        """
        try:
            # Preprocess the region
            processed = self.preprocess_for_ocr(region)

            # Extract with appropriate OCR engine
            self.frame_counter += 1

            if self.ocr_engine == "paddleocr":
                results = self.paddle.ocr(processed, cls=True)
                if results and len(results) > 0:
                    return self._extract_text_from_results(results[0])
                else:
                    return ""

            elif self.ocr_engine == "easyocr":
                results = self.paddle.readtext(processed)
                return self._extract_text_from_results(results)

            else:  # tesseract
                import pytesseract

                results = pytesseract.image_to_string(
                    processed,
                    config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ&:",
                )
                return self._extract_text_from_results(results)

            return ""

        except Exception as e:
            logger.error(f"Error in extract_text: {e}")
            return ""

    def extract_down_distance(self, region: np.ndarray) -> Optional[str]:
        """
        Extract down and distance information from a region.

        Args:
            region: Image region containing down/distance text

        Returns:
            Down and distance text (e.g., "1ST & 10")
        """
        return self.extract_text(region)

    def extract_game_clock(self, region: np.ndarray) -> Optional[str]:
        """
        Extract game clock information from a region.

        Args:
            region: Image region containing game clock text

        Returns:
            Game clock text (e.g., "12:34")
        """
        return self.extract_text(region)

    def extract_play_clock(self, region: np.ndarray) -> Optional[str]:
        """
        Extract play clock information from a region.

        Args:
            region: Image region containing play clock text

        Returns:
            Play clock text (e.g., "25")
        """
        return self.extract_text(region)

    def extract_scores(self, region: np.ndarray) -> Optional[dict[str, str]]:
        """
        Extract score information from a region.

        Args:
            region: Image region containing score text

        Returns:
            Dictionary with score information
        """
        text = self.extract_text(region)
        return {"raw_text": text} if text else None

    def process_region(self, region: np.ndarray, debug_mode: bool = False) -> dict[str, Any]:
        """
        Process a region and extract all relevant information.

        Args:
            region: Image region to process
            debug_mode: Whether to include debug information

        Returns:
            Dictionary with extracted information
        """
        text = self.extract_text(region)
        return {
            "text": text,
            "confidence": 0.8 if text else 0.0,
            "engine": self.ocr_engine,
            "frame_count": self.frame_counter,
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for OCR.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        return self.preprocess_for_ocr(image)

#!/usr/bin/env python3
"""
Madden OCR Deployment - Integration with SpygateAI
Replaces existing OCR with custom trained model for maximum accuracy
Optimized for Madden constraints: play clock ≤40, quarter time ≤5:00, distance ≤34 yards
"""

import logging
import os
import re
from typing import Dict, Optional

import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaddenOCRPredictor:
    """Production OCR predictor using trained Madden model"""

    def __init__(
        self,
        model_path: str = "madden_ocr_model.h5",
        mappings_path: str = "madden_ocr_mappings.pkl",
    ):
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}

        # Madden constraints for validation
        self.constraints = {
            "play_clock_max": 40,
            "quarter_time_max": 300,
            "down_distance_max": 34,
            "down_max": 4,
            "score_max": 99,
        }

        # Try to load model
        self._load_model(model_path, mappings_path)

    def _load_model(self, model_path: str, mappings_path: str):
        """Load trained model and character mappings"""
        try:
            if os.path.exists(model_path) and os.path.exists(mappings_path):
                import pickle

                import tensorflow as tf
                from tensorflow import keras

                # Load character mappings
                with open(mappings_path, "rb") as f:
                    mappings = pickle.load(f)

                self.char_to_idx = mappings["char_to_idx"]
                self.idx_to_char = mappings["idx_to_char"]

                # Load model
                self.model = keras.models.load_model(model_path)
                logger.info("✅ Madden OCR model loaded successfully")
            else:
                logger.warning("Model files not found. Using fallback OCR only.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict_text(self, image: np.ndarray, context: str = "") -> Dict[str, any]:
        """Predict text from image with validation"""
        if self.model is None:
            return {"text": "", "confidence": 0.0, "valid": False, "error": "Model not loaded"}

        try:
            # Preprocess and predict
            processed = self._preprocess_image(image)
            processed = np.expand_dims(processed, axis=0)

            prediction = self.model.predict(processed, verbose=0)
            predicted_text = self._decode_prediction(prediction[0])

            # Apply corrections and validation
            corrected_text = self._apply_corrections(predicted_text, context)
            is_valid = self._validate_text(corrected_text, context)

            return {
                "text": corrected_text,
                "raw_text": predicted_text,
                "confidence": 0.8,  # Simplified confidence
                "valid": is_valid,
                "context": context,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"text": "", "confidence": 0.0, "valid": False, "error": str(e)}

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to standard size
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_CUBIC)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)

    def _decode_prediction(self, prediction: np.ndarray) -> str:
        """Decode prediction to text"""
        # Simplified decoding - in real implementation would use CTC decoding
        indices = np.argmax(prediction, axis=1)
        chars = []
        for idx in indices:
            if idx > 0 and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return "".join(chars).strip()

    def _apply_corrections(self, text: str, context: str) -> str:
        """Apply Madden-specific corrections"""
        if not text:
            return text

        corrected = text.upper().strip()

        # General corrections for Madden font
        corrections = {
            "O": "0",
            "I": "1",
            "S": "5",
            "G": "6",
            "B": "8",
            "q": "9",
            "G0AL": "GOAL",
            "GDAL": "GOAL",
            "FL4G": "FLAG",
            "FLOG": "FLAG",
            "1S1": "1ST",
            "15T": "1ST",
            "2N0": "2ND",
            "2NO": "2ND",
            "3R0": "3RD",
            "3RO": "3RD",
            "4T8": "4TH",
            "4TI": "4TH",
        }

        for wrong, right in corrections.items():
            corrected = corrected.replace(wrong, right)

        # Context-specific corrections
        if context == "down_distance_area":
            corrected = self._correct_down_distance(corrected)
        elif context == "play_clock_area":
            corrected = self._correct_play_clock(corrected)
        elif context == "game_clock_area":
            corrected = self._correct_game_clock(corrected)

        return corrected

    def _correct_down_distance(self, text: str) -> str:
        """Correct down & distance specific text"""
        if "GOAL" in text:
            return "GOAL"

        # Fix ordinal + distance patterns
        patterns = [
            (r"(\d+)(S1|S7|ST)\s*&\s*(\d+)", r"\1ST & \3"),
            (r"(\d+)(N0|NO|ND)\s*&\s*(\d+)", r"\1ND & \3"),
            (r"(\d+)(R0|RO|RD)\s*&\s*(\d+)", r"\1RD & \3"),
            (r"(\d+)(T8|TI|TH)\s*&\s*(\d+)", r"\1TH & \3"),
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _correct_play_clock(self, text: str) -> str:
        """Correct play clock text - must be 1-40"""
        digits = re.sub(r"[^\d]", "", text)
        if digits and digits.isdigit():
            clock_value = int(digits)
            if 1 <= clock_value <= self.constraints["play_clock_max"]:
                return str(clock_value)
        return text

    def _correct_game_clock(self, text: str) -> str:
        """Correct game clock text - must be 0:00-4:59"""
        corrections = {"O:": "0:", "I:": "1:", ":O0": ":00", ":0O": ":00"}
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        return text

    def _validate_text(self, text: str, context: str) -> bool:
        """Validate text against Madden constraints"""
        if not text or not text.strip():
            return False

        text = text.strip().upper()

        if context == "down_distance_area":
            return self._validate_down_distance(text)
        elif context == "play_clock_area":
            return self._validate_play_clock(text)
        elif context == "game_clock_area":
            return self._validate_game_clock(text)

        return True

    def _validate_down_distance(self, text: str) -> bool:
        """Validate down & distance - must be 1-4 down, 1-34 distance, or GOAL"""
        if text == "GOAL":
            return True

        patterns = [
            r"^([1-4])(ST|ND|RD|TH)\s*&\s*([1-9]|[12][0-9]|3[0-4])$",
            r"^([1-4])\s*&\s*([1-9]|[12][0-9]|3[0-4])$",
        ]

        return any(re.match(pattern, text) for pattern in patterns)

    def _validate_play_clock(self, text: str) -> bool:
        """Validate play clock - must be 1-40"""
        if text.isdigit():
            clock_value = int(text)
            return 1 <= clock_value <= self.constraints["play_clock_max"]
        return False

    def _validate_game_clock(self, text: str) -> bool:
        """Validate game clock - must be 0:00-4:59"""
        patterns = [r"^([0-4]):[0-5][0-9]$", r"^([0-4]):[0-5][0-9]\.[0-9]$"]
        return any(re.match(pattern, text) for pattern in patterns)


class SpygateOCRIntegration:
    """Integration layer for SpygateAI system"""

    def __init__(self):
        self.madden_ocr = MaddenOCRPredictor()
        self.fallback_available = self._init_fallback()

    def _init_fallback(self) -> bool:
        """Initialize fallback OCR engines"""
        try:
            import easyocr
            import pytesseract

            self.easyocr_reader = easyocr.Reader(["en"])
            logger.info("✅ Fallback OCR engines loaded")
            return True
        except ImportError as e:
            logger.warning(f"Fallback OCR not available: {e}")
            return False

    def extract_text(self, image: np.ndarray, context: str = "") -> Dict[str, any]:
        """Extract text using custom model with fallback"""
        # Try custom Madden model first
        if self.madden_ocr.model is not None:
            result = self.madden_ocr.predict_text(image, context)

            # If valid result, return it
            if result["valid"] and result["confidence"] > 0.5:
                logger.debug(f"✅ Madden OCR: '{result['text']}'")
                return result

        # Fallback to traditional OCR
        if self.fallback_available:
            fallback_result = self._fallback_ocr(image, context)
            if fallback_result["valid"]:
                logger.debug(f"🔄 Fallback OCR: '{fallback_result['text']}'")
                return fallback_result

        # Return best available result
        if self.madden_ocr.model is not None:
            return self.madden_ocr.predict_text(image, context)

        return {"text": "", "confidence": 0.0, "valid": False, "error": "No OCR available"}

    def _fallback_ocr(self, image: np.ndarray, context: str) -> Dict[str, any]:
        """Fallback OCR using traditional engines"""
        try:
            # Convert image for OCR
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Try EasyOCR
            easyocr_results = self.easyocr_reader.readtext(image_rgb, detail=0)
            easyocr_text = " ".join(easyocr_results) if easyocr_results else ""

            # Try Tesseract
            import pytesseract
            from PIL import Image

            pil_image = Image.fromarray(image_rgb)
            tesseract_text = pytesseract.image_to_string(pil_image, config="--psm 8").strip()

            # Choose best result
            for engine, text in [("easyocr", easyocr_text), ("tesseract", tesseract_text)]:
                if text and len(text.strip()) > 0:
                    corrected = self.madden_ocr._apply_corrections(text, context)
                    valid = self.madden_ocr._validate_text(corrected, context)

                    if valid:
                        return {
                            "text": corrected,
                            "confidence": 0.6,
                            "valid": True,
                            "engine": engine,
                        }

            # Return best non-validated result
            best_text = easyocr_text if easyocr_text else tesseract_text
            return {"text": best_text, "confidence": 0.3, "valid": False, "engine": "fallback"}

        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return {"text": "", "confidence": 0.0, "valid": False, "error": str(e)}


# Global instance for easy integration
spygate_ocr = SpygateOCRIntegration()


def extract_madden_text(image: np.ndarray, context: str = "") -> str:
    """Simple function for easy integration with existing SpygateAI code"""
    result = spygate_ocr.extract_text(image, context)
    return result["text"]


def extract_madden_text_detailed(image: np.ndarray, context: str = "") -> Dict[str, any]:
    """Detailed function returning full OCR results"""
    return spygate_ocr.extract_text(image, context)


def main():
    """Test the deployment system"""
    print("🎯 Madden OCR Deployment - Ultimate Accuracy System")
    print("=" * 55)

    # Test system status
    if spygate_ocr.madden_ocr.model is not None:
        print("✅ Custom Madden OCR model loaded successfully")
        print("   - Optimized for Madden HUD font and patterns")
        print("   - Built-in validation for game constraints")
        print("   - Play clock: 1-40 seconds")
        print("   - Game clock: 0:00-4:59")
        print("   - Down & distance: 1-4 down, 1-34 yards, GOAL")
    else:
        print("❌ Custom model not available")
        print("   - Run training pipeline first")
        print("   - Using fallback OCR only")

    if spygate_ocr.fallback_available:
        print("✅ Fallback OCR engines available (EasyOCR + Tesseract)")
    else:
        print("❌ No fallback OCR available")

    print("\n🔧 Integration Functions:")
    print("   extract_madden_text(image, context) - Simple text extraction")
    print("   extract_madden_text_detailed(image, context) - Full results")
    print("\n📋 Supported contexts:")
    print("   - 'down_distance_area' - Down & distance text")
    print("   - 'play_clock_area' - Play clock (1-40)")
    print("   - 'game_clock_area' - Game clock (0:00-4:59)")
    print("\n🎯 System ready for SpygateAI integration!")


if __name__ == "__main__":
    main()

"""
Custom Madden OCR Integration
Uses trained Madden-specific OCR models for enhanced HUD text extraction.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


class CustomMaddenOCR:
    """
    Custom OCR engine specifically trained on Madden HUD elements.
    Provides superior accuracy for down/distance, clock, and score extraction.
    """

    def __init__(self, models_dir: str = "trained_madden_ocr"):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Trained models for each region type
        self.models = {}
        self.processors = {}

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "region_stats": {},
        }

        self.load_trained_models()

    def load_trained_models(self):
        """Load all trained Madden OCR models"""
        if not self.models_dir.exists():
            logger.warning(f"No trained models found at {self.models_dir}")
            return

        # Check for training summary
        summary_path = self.models_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            logger.info(f"Loading {len(summary['trained_models'])} trained models")

            for region_type, model_path in summary["trained_models"].items():
                self.load_region_model(region_type, model_path)
        else:
            # Fallback: scan for model directories
            for model_dir in self.models_dir.glob("madden_ocr_*"):
                region_type = model_dir.name.replace("madden_ocr_", "")
                self.load_region_model(region_type, str(model_dir))

        logger.info(f"✅ Loaded {len(self.models)} custom Madden OCR models")

    def load_region_model(self, region_type: str, model_path: str):
        """Load specific region model"""
        try:
            model_dir = Path(model_path)

            # Load processor and model
            processor = TrOCRProcessor.from_pretrained(model_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
            model.to(self.device)
            model.eval()

            self.processors[region_type] = processor
            self.models[region_type] = model

            logger.info(f"Loaded {region_type} OCR model from {model_dir}")

        except Exception as e:
            logger.error(f"Failed to load {region_type} model: {e}")

    def extract_text(self, image: np.ndarray, region_type: str = "down_distance") -> Dict[str, Any]:
        """
        Extract text using custom trained model for specific region.

        Args:
            image: OpenCV image (BGR format)
            region_type: Type of region ('down_distance', 'game_clock', 'play_clock', 'scores')

        Returns:
            Dictionary with extracted text and confidence
        """
        if region_type not in self.models:
            return self.fallback_extraction(image, region_type)

        try:
            self.extraction_stats["total_extractions"] += 1

            # Preprocess image
            processed_image = self.preprocess_for_model(image)

            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)

            # Extract text using trained model
            result = self.extract_with_model(pil_image, region_type)

            # Post-process and validate
            validated_result = self.validate_extraction(result, region_type)

            if validated_result["confidence"] > 0.5:
                self.extraction_stats["successful_extractions"] += 1

            # Update region stats
            if region_type not in self.extraction_stats["region_stats"]:
                self.extraction_stats["region_stats"][region_type] = {
                    "total": 0,
                    "successful": 0,
                    "avg_confidence": 0.0,
                }

            stats = self.extraction_stats["region_stats"][region_type]
            stats["total"] += 1
            if validated_result["confidence"] > 0.5:
                stats["successful"] += 1

            return validated_result

        except Exception as e:
            logger.error(f"Custom OCR extraction failed for {region_type}: {e}")
            return self.fallback_extraction(image, region_type)

    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal model performance"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance contrast
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        # Resize for consistency with training
        height, width = image.shape[:2]
        if height < 32 or width < 64:
            scale_factor = max(32 / height, 64 / width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return image

    def extract_with_model(self, pil_image: Image.Image, region_type: str) -> Dict[str, Any]:
        """Extract text using the trained model"""
        processor = self.processors[region_type]
        model = self.models[region_type]

        # Process image
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_length=50, num_beams=4, early_stopping=True
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate confidence (basic implementation)
        confidence = self.calculate_confidence(generated_text, region_type)

        return {
            "text": generated_text.strip(),
            "confidence": confidence,
            "method": f"custom_madden_{region_type}",
            "preprocessing": True,
        }

    def calculate_confidence(self, text: str, region_type: str) -> float:
        """Calculate confidence score based on text patterns and region type"""
        if not text or len(text.strip()) == 0:
            return 0.0

        confidence = 0.5  # Base confidence

        if region_type == "down_distance":
            # Check for down & distance patterns
            if any(pattern in text.upper() for pattern in ["ST &", "ND &", "RD &", "TH &"]):
                confidence += 0.3
            if any(char.isdigit() for char in text):
                confidence += 0.2

        elif region_type == "game_clock":
            # Check for clock patterns (MM:SS)
            if ":" in text and len(text.split(":")) == 2:
                confidence += 0.4
                try:
                    parts = text.split(":")
                    if parts[0].isdigit() and parts[1].isdigit():
                        confidence += 0.2
                except:
                    pass

        elif region_type == "play_clock":
            # Check for simple number
            if text.isdigit() and 1 <= int(text) <= 40:
                confidence += 0.4

        elif region_type == "scores":
            # Check for team score patterns
            if any(char.isdigit() for char in text) and len(text) > 3:
                confidence += 0.3

        return min(confidence, 1.0)

    def validate_extraction(self, result: Dict[str, Any], region_type: str) -> Dict[str, Any]:
        """Validate and correct extraction results"""
        text = result["text"]
        confidence = result["confidence"]

        # Apply region-specific corrections
        if region_type == "down_distance":
            text = self.correct_down_distance(text)
        elif region_type == "game_clock":
            text = self.correct_game_clock(text)
        elif region_type == "play_clock":
            text = self.correct_play_clock(text)
        elif region_type == "scores":
            text = self.correct_scores(text)

        # Adjust confidence based on corrections
        if text != result["text"]:
            confidence = max(0.6, confidence)  # Boost confidence after correction

        return {
            "text": text,
            "confidence": confidence,
            "method": result["method"],
            "preprocessing": result["preprocessing"],
            "corrected": text != result["text"],
        }

    def correct_down_distance(self, text: str) -> str:
        """Apply Madden-specific down & distance corrections"""
        text = text.upper().strip()

        # Common OCR mistakes for down & distance
        corrections = {
            "1ST": ["1ST", "IST", "1 ST", "1S7", "151"],
            "2ND": ["2ND", "2 ND", "2N0", "ZND"],
            "3RD": ["3RD", "3 RD", "3R0", "SRD"],
            "4TH": ["4TH", "4 TH", "47H", "ATH"],
            "&": ["&", "S", "8", "B"],
        }

        # Apply corrections
        for correct, variants in corrections.items():
            for variant in variants:
                text = text.replace(variant, correct)

        # Ensure proper format
        if "ST" in text and "&" in text:
            return text
        elif any(down in text for down in ["1ST", "2ND", "3RD", "4TH"]):
            if "&" not in text:
                # Try to add missing &
                for down in ["1ST", "2ND", "3RD", "4TH"]:
                    if down in text:
                        rest = text.replace(down, "").strip()
                        if rest.isdigit():
                            return f"{down} & {rest}"

        return text

    def correct_game_clock(self, text: str) -> str:
        """Apply game clock corrections"""
        text = text.strip()

        # Ensure proper MM:SS format
        if ":" not in text:
            # Try to insert colon
            if len(text) >= 3:
                # Assume format like "1234" -> "12:34"
                return f"{text[:-2]}:{text[-2:]}"

        return text

    def correct_play_clock(self, text: str) -> str:
        """Apply play clock corrections"""
        text = text.strip()

        # Extract digits only
        digits = "".join(c for c in text if c.isdigit())

        if digits and 1 <= int(digits) <= 40:
            return digits

        return text

    def correct_scores(self, text: str) -> str:
        """Apply team scores corrections"""
        # Basic cleanup for team scores
        return text.strip()

    def fallback_extraction(self, image: np.ndarray, region_type: str) -> Dict[str, Any]:
        """Fallback to generic OCR if custom model unavailable"""
        try:
            # Use EasyOCR as fallback
            import easyocr

            reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = reader.readtext(image_rgb)

            if results:
                # Get text with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1]
                confidence = best_result[2] * 0.8  # Reduce confidence for fallback

                return {
                    "text": text,
                    "confidence": confidence,
                    "method": "fallback_easyocr",
                    "preprocessing": False,
                }
        except:
            pass

        return {"text": "", "confidence": 0.0, "method": "fallback_failed", "preprocessing": False}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get OCR performance statistics"""
        stats = self.extraction_stats.copy()

        if stats["total_extractions"] > 0:
            stats["overall_success_rate"] = (
                stats["successful_extractions"] / stats["total_extractions"]
            )
        else:
            stats["overall_success_rate"] = 0.0

        # Calculate per-region success rates
        for region, region_stats in stats["region_stats"].items():
            if region_stats["total"] > 0:
                region_stats["success_rate"] = region_stats["successful"] / region_stats["total"]
            else:
                region_stats["success_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "region_stats": {},
        }

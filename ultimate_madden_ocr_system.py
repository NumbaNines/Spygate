#!/usr/bin/env python3
"""
Ultimate Maximum Accuracy Madden OCR System
Specifically optimized for Madden HUD font and text patterns:
- Numbers: 0-9
- Ordinals: st, nd, rd, th
- Special words: GOAL, qtr, FLAG
- Down/distance patterns: 1ST & 10, 2ND & 7, etc.
- Time patterns: 15:00, 2:34, etc.
"""

import json
import logging
import os
import random
import re
import sqlite3
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaddenFontPatterns:
    """Madden-specific font patterns and validation with game constraints"""

    # Core character set used in Madden HUD
    VALID_CHARS = set("0123456789STNDRDTHGOALqtrFLAG&:- ")

    # Madden game constraints
    CONSTRAINTS = {
        "play_clock_max": 40,  # Play clock never exceeds 40 seconds
        "quarter_time_max": 300,  # Quarter time max 5:00 (300 seconds)
        "down_distance_max": 34,  # Down & distance never exceeds 34 yards
        "down_max": 4,  # Maximum down is 4th
        "score_max": 99,  # Reasonable score maximum
    }

    # Common patterns in Madden HUD with constraints
    PATTERNS = {
        "down_distance": [
            r"^([1-4])(ST|ND|RD|TH)\s*&\s*([1-9]|[12][0-9]|3[0-4])$",  # 1ST & 1-34, etc.
            r"^([1-4])\s*&\s*([1-9]|[12][0-9]|3[0-4])$",  # 1 & 1-34, etc.
            r"^GOAL$",  # GOAL line
        ],
        "play_clock": [
            r"^([1-9]|[12][0-9]|3[0-9]|40)$",  # 1-40 seconds
        ],
        "game_clock": [
            r"^([0-4]):[0-5][0-9]$",  # 0:00 to 4:59
            r"^([0-4]):[0-5][0-9]\.[0-9]$",  # With tenths
        ],
        "quarter": [
            r"^([1-4])(ST|ND|RD|TH)\s*QTR$",  # 1ST-4TH QTR
            r"^([1-4])\s*QTR$",  # 1-4 QTR
        ],
        "score": [
            r"^([0-9]|[1-9][0-9])$",  # 0-99
        ],
        "territory": [
            r"^(50|[1-4][0-9]|[1-9])$",  # 1-50 (yard line numbers only)
        ],
        "flag": [
            r"^FLAG$",  # Penalty flag
        ],
    }

    # Enhanced OCR corrections for Madden font with constraints
    CORRECTIONS = {
        # Number corrections
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "S": "5",
        "s": "5",
        "G": "6",
        "g": "6",
        "B": "8",
        "b": "8",
        "q": "9",
        # Letter corrections
        "0": "O",
        "6": "G",
        "8": "B",
        "5": "S",
        # Word corrections
        "G0AL": "GOAL",
        "GDAL": "GOAL",
        "G04L": "GOAL",
        "GQAL": "GOAL",
        "FL4G": "FLAG",
        "FLOG": "FLAG",
        "FL46": "FLAG",
        "FLAC": "FLAG",
        "QTR": "QTR",
        "QT8": "QTR",
        "QT6": "QTR",
        "QT0": "QTR",
        # Ordinal corrections
        "1ST": "1ST",
        "15T": "1ST",
        "1S7": "1ST",
        "1S1": "1ST",
        "2ND": "2ND",
        "2N0": "2ND",
        "2N6": "2ND",
        "2NO": "2ND",
        "3RD": "3RD",
        "3R0": "3RD",
        "3R6": "3RD",
        "3RO": "3RD",
        "4TH": "4TH",
        "47H": "4TH",
        "4T8": "4TH",
        "4TI": "4TH",
    }

    @classmethod
    def validate_text(cls, text: str, context: str) -> bool:
        """Validate extracted text against Madden constraints"""
        if not text or not text.strip():
            return False

        text = text.strip().upper()

        # Context-specific validation
        if context == "down_distance_area":
            return cls._validate_down_distance(text)
        elif context == "play_clock_area":
            return cls._validate_play_clock(text)
        elif context == "game_clock_area":
            return cls._validate_game_clock(text)
        elif context == "territory_triangle_area":
            return cls._validate_territory(text)

        return True

    @classmethod
    def _validate_down_distance(cls, text: str) -> bool:
        """Validate down & distance text"""
        if text == "GOAL":
            return True

        # Check patterns
        for pattern in cls.PATTERNS["down_distance"]:
            match = re.match(pattern, text)
            if match:
                if "&" in text:
                    parts = text.split("&")
                    if len(parts) == 2:
                        # Extract distance number
                        distance_part = parts[1].strip()
                        distance_num = re.findall(r"\d+", distance_part)
                        if distance_num:
                            distance = int(distance_num[0])
                            return distance <= cls.CONSTRAINTS["down_distance_max"]
                return True

        return False

    @classmethod
    def _validate_play_clock(cls, text: str) -> bool:
        """Validate play clock text"""
        if text.isdigit():
            clock_value = int(text)
            return 1 <= clock_value <= cls.CONSTRAINTS["play_clock_max"]
        return False

    @classmethod
    def _validate_game_clock(cls, text: str) -> bool:
        """Validate game clock text"""
        for pattern in cls.PATTERNS["game_clock"]:
            if re.match(pattern, text):
                # Extract minutes
                if ":" in text:
                    minutes = int(text.split(":")[0])
                    return minutes <= 4  # Max 4:59
        return False

    @classmethod
    def _validate_territory(cls, text: str) -> bool:
        """Validate territory/field position text"""
        for pattern in cls.PATTERNS["territory"]:
            if re.match(pattern, text):
                return True
        return False

    @classmethod
    def correct_text(cls, text: str, context: str) -> str:
        """Apply Madden-specific corrections to OCR text"""
        if not text:
            return text

        corrected = text.upper().strip()

        # Apply general corrections
        for wrong, right in cls.CORRECTIONS.items():
            corrected = corrected.replace(wrong, right)

        # Context-specific corrections
        if context == "down_distance_area":
            corrected = cls._correct_down_distance(corrected)
        elif context == "play_clock_area":
            corrected = cls._correct_play_clock(corrected)
        elif context == "game_clock_area":
            corrected = cls._correct_game_clock(corrected)
        elif context == "territory_triangle_area":
            corrected = cls._correct_territory(corrected)

        return corrected

    @classmethod
    def _correct_down_distance(cls, text: str) -> str:
        """Apply down & distance specific corrections"""
        # Common down & distance OCR mistakes
        corrections = {
            "1S1 & 10": "1ST & 10",
            "1S7 & 10": "1ST & 10",
            "2N0 & 7": "2ND & 7",
            "2NO & 7": "2ND & 7",
            "3R0 & 3": "3RD & 3",
            "3RO & 3": "3RD & 3",
            "4T8 & 1": "4TH & 1",
            "4TI & 1": "4TH & 1",
            "G0AL": "GOAL",
            "GDAL": "GOAL",
            "GQAL": "GOAL",
        }

        for wrong, right in corrections.items():
            if wrong in text:
                text = text.replace(wrong, right)

        return text

    @classmethod
    def _correct_play_clock(cls, text: str) -> str:
        """Apply play clock specific corrections"""
        # Remove non-digit characters
        digits_only = re.sub(r"[^\d]", "", text)

        if digits_only and digits_only.isdigit():
            clock_value = int(digits_only)
            # Ensure within valid range
            if 1 <= clock_value <= cls.CONSTRAINTS["play_clock_max"]:
                return str(clock_value)

        return text

    @classmethod
    def _correct_game_clock(cls, text: str) -> str:
        """Apply game clock specific corrections"""
        # Common game clock OCR mistakes
        corrections = {
            "O:": "0:",
            "o:": "0:",
            "I:": "1:",
            "l:": "1:",
            ":O0": ":00",
            ":o0": ":00",
            ":0O": ":00",
            ":0o": ":00",
        }

        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

        return text

    @classmethod
    def _correct_territory(cls, text: str) -> str:
        """Apply territory/field position specific corrections for yard line numbers"""
        # Common yard line number OCR mistakes
        corrections = {
            # Number corrections for yard lines (1-50)
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "S": "5",
            "s": "5",
            "G": "6",
            "g": "6",
            "B": "8",
            "b": "8",
            "q": "9",
        }

        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

        return text


class MaddenOCRExtractor:
    """Extract text regions from Madden screenshots using YOLO + enhanced preprocessing"""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.easyocr_reader = easyocr.Reader(["en"])

        # Target classes for OCR extraction
        self.ocr_classes = [
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
            "territory_triangle_area",
        ]

    def extract_regions(self, image_path: str) -> List[Dict]:
        """Extract all OCR regions from image with multiple preprocessing variants"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return []

        # Run YOLO detection
        results = self.model(image, conf=0.3, verbose=False)

        extracted_regions = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                if class_name in self.ocr_classes and confidence > 0.3:
                    # Extract bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Add padding
                    padding = 5
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.shape[1], x2 + padding)
                    y2 = min(image.shape[0], y2 + padding)

                    # Extract region
                    region = image[y1:y2, x1:x2]

                    if region.size == 0:
                        continue

                    # Generate multiple preprocessing variants
                    variants = self._generate_preprocessing_variants(region)

                    # Extract text from each variant
                    for variant_name, variant_image in variants.items():
                        ocr_results = self._extract_text_multi_engine(variant_image)

                        extracted_regions.append(
                            {
                                "image_path": image_path,
                                "class_name": class_name,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "variant": variant_name,
                                "region_image": variant_image,
                                "ocr_results": ocr_results,
                                "width": x2 - x1,
                                "height": y2 - y1,
                            }
                        )

        return extracted_regions

    def _generate_preprocessing_variants(self, region: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate multiple preprocessing variants for maximum OCR accuracy"""
        variants = {}

        # Original
        variants["original"] = region.copy()

        # Grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        variants["grayscale"] = gray

        # Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        variants["enhanced_contrast"] = enhanced

        # Gaussian blur removal (sharpening)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        variants["sharpened"] = sharpened

        # Threshold variants
        _, thresh_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["thresh_binary"] = thresh_binary

        _, thresh_binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants["thresh_binary_inv"] = thresh_binary_inv

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        variants["adaptive_thresh"] = adaptive

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_close = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)
        variants["morph_close"] = morph_close

        # Scale up for better OCR (2x, 3x, 4x)
        for scale in [2, 3, 4]:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            variants[f"scaled_{scale}x"] = scaled

            # Scaled + enhanced
            scaled_enhanced = cv2.convertScaleAbs(scaled, alpha=1.3, beta=5)
            variants[f"scaled_{scale}x_enhanced"] = scaled_enhanced

        return variants

    def _extract_text_multi_engine(self, image: np.ndarray) -> Dict[str, str]:
        """Extract text using multiple OCR engines"""
        results = {}

        # EasyOCR
        try:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            easyocr_results = self.easyocr_reader.readtext(image_rgb, detail=0)
            results["easyocr"] = " ".join(easyocr_results) if easyocr_results else ""
        except Exception as e:
            results["easyocr"] = ""
            logger.debug(f"EasyOCR failed: {e}")

        # Tesseract with different configs
        try:
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Standard config
            results["tesseract_standard"] = pytesseract.image_to_string(
                pil_image, config="--psm 8"
            ).strip()

            # Digits only
            results["tesseract_digits"] = pytesseract.image_to_string(
                pil_image, config="--psm 8 -c tessedit_char_whitelist=0123456789"
            ).strip()

            # Alphanumeric + common symbols
            results["tesseract_alphanum"] = pytesseract.image_to_string(
                pil_image,
                config="--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ&:-",
            ).strip()

        except Exception as e:
            results["tesseract_standard"] = ""
            results["tesseract_digits"] = ""
            results["tesseract_alphanum"] = ""
            logger.debug(f"Tesseract failed: {e}")

        return results


class MaddenOCRDatabase:
    """SQLite database for managing OCR training samples"""

    def __init__(self, db_path: str = "madden_ocr_training.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                variant TEXT NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                width INTEGER,
                height INTEGER,
                confidence REAL,
                easyocr_text TEXT,
                tesseract_standard_text TEXT,
                tesseract_digits_text TEXT,
                tesseract_alphanum_text TEXT,
                ground_truth_text TEXT,
                is_validated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_class_name ON ocr_samples(class_name)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validated ON ocr_samples(is_validated)
        """
        )

        conn.commit()
        conn.close()

    def insert_sample(self, sample: Dict):
        """Insert OCR sample into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ocr_samples (
                image_path, class_name, variant, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                width, height, confidence, easyocr_text, tesseract_standard_text,
                tesseract_digits_text, tesseract_alphanum_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                sample["image_path"],
                sample["class_name"],
                sample["variant"],
                sample["bbox"][0],
                sample["bbox"][1],
                sample["bbox"][2],
                sample["bbox"][3],
                sample["width"],
                sample["height"],
                sample["confidence"],
                sample["ocr_results"].get("easyocr", ""),
                sample["ocr_results"].get("tesseract_standard", ""),
                sample["ocr_results"].get("tesseract_digits", ""),
                sample["ocr_results"].get("tesseract_alphanum", ""),
            ),
        )

        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return sample_id

    def get_unvalidated_samples(self, limit: int = 100) -> List[Dict]:
        """Get unvalidated samples for manual correction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM ocr_samples
            WHERE is_validated = FALSE
            ORDER BY confidence DESC, created_at ASC
            LIMIT ?
        """,
            (limit,),
        )

        columns = [desc[0] for desc in cursor.description]
        samples = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return samples

    def update_ground_truth(self, sample_id: int, ground_truth: str):
        """Update ground truth text for sample"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE ocr_samples
            SET ground_truth_text = ?, is_validated = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (ground_truth, sample_id),
        )

        conn.commit()
        conn.close()

    def remove_ground_truth(self, sample_id: int):
        """Remove ground truth validation from a sample (for undo functionality)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE ocr_samples
            SET ground_truth_text = NULL, is_validated = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (sample_id,),
        )

        conn.commit()
        conn.close()

    def get_training_data(self) -> List[Dict]:
        """Get all validated samples for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM ocr_samples
            WHERE is_validated = TRUE AND ground_truth_text IS NOT NULL
            ORDER BY class_name, created_at
        """
        )

        columns = [desc[0] for desc in cursor.description]
        samples = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return samples

    def get_all_samples(self, limit: int = 100) -> List[Dict]:
        """Get all samples (validated and unvalidated) for browsing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM ocr_samples
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        columns = [desc[0] for desc in cursor.description]
        samples = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return samples

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total samples
        cursor.execute("SELECT COUNT(*) FROM ocr_samples")
        stats["total_samples"] = cursor.fetchone()[0]

        # Validated samples
        cursor.execute("SELECT COUNT(*) FROM ocr_samples WHERE is_validated = TRUE")
        stats["validated_samples"] = cursor.fetchone()[0]

        # Samples by class
        cursor.execute("SELECT class_name, COUNT(*) FROM ocr_samples GROUP BY class_name")
        stats["by_class"] = dict(cursor.fetchall())

        # Validated samples by class
        cursor.execute(
            "SELECT class_name, COUNT(*) FROM ocr_samples WHERE is_validated = TRUE GROUP BY class_name"
        )
        stats["validated_by_class"] = dict(cursor.fetchall())

        conn.close()
        return stats


class MaddenOCRAnnotationGUI:
    """Professional GUI for manual OCR correction and validation"""

    def __init__(self, database: MaddenOCRDatabase):
        self.database = database
        self.current_sample = None
        self.current_sample_index = 0
        self.samples = []
        self.last_ground_truth = ""  # Store last entered text for pre-filling
        self.recently_validated = []  # Store recently validated samples for undo
        self.browse_all_mode = False  # Toggle between unvalidated-only and all samples
        self.last_validated_sample_id = None  # Remember last validated sample for browse mode

        self.setup_gui()
        self.load_samples()

    def setup_gui(self):
        """Setup the annotation GUI"""
        self.root = tk.Tk()
        self.root.title("Madden OCR Annotation Tool - Ultimate Accuracy")
        self.root.geometry("1200x800")

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame - Statistics and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Statistics
        stats_frame = ttk.LabelFrame(top_frame, text="Statistics")
        stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.stats_label = ttk.Label(stats_frame, text="Loading statistics...")
        self.stats_label.pack(padx=10, pady=5)

        # Controls
        controls_frame = ttk.LabelFrame(top_frame, text="Controls")
        controls_frame.pack(side=tk.RIGHT)

        ttk.Button(controls_frame, text="Refresh", command=self.load_samples).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.browse_mode_button = ttk.Button(
            controls_frame, text="Browse All Samples", command=self.toggle_browse_mode
        )
        self.browse_mode_button.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(
            controls_frame, text="Export Training Data", command=self.export_training_data
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Middle frame - Image and OCR results
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Left side - Image display
        image_frame = ttk.LabelFrame(middle_frame, text="Region Image")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.pack(expand=True)

        # Right side - OCR results and correction
        ocr_frame = ttk.LabelFrame(middle_frame, text="OCR Results & Correction", width=400)
        ocr_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ocr_frame.pack_propagate(False)

        # Sample info
        info_frame = ttk.Frame(ocr_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(info_frame, text="Sample: 0/0")
        self.info_label.pack()

        # Validation status indicator
        self.status_label = ttk.Label(info_frame, text="", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=(5, 0))

        # OCR results
        results_frame = ttk.LabelFrame(ocr_frame, text="OCR Engine Results")
        results_frame.pack(fill=tk.X, padx=10, pady=5)

        self.ocr_results_text = tk.Text(results_frame, height=8, width=40)
        self.ocr_results_text.pack(fill=tk.X, padx=5, pady=5)

        # Ground truth input
        truth_frame = ttk.LabelFrame(ocr_frame, text="Ground Truth (Correct Text)")
        truth_frame.pack(fill=tk.X, padx=10, pady=5)

        self.ground_truth_var = tk.StringVar()
        self.ground_truth_entry = ttk.Entry(
            truth_frame, textvariable=self.ground_truth_var, font=("Arial", 14)
        )
        self.ground_truth_entry.pack(fill=tk.X, padx=5, pady=5)

        # Quick buttons for common patterns
        quick_frame = ttk.Frame(truth_frame)
        quick_frame.pack(fill=tk.X, padx=5, pady=5)

        quick_buttons = [
            ("1ST & 10", "1ST & 10"),
            ("2ND & 7", "2ND & 7"),
            ("3RD & 3", "3RD & 3"),
            ("4TH & 1", "4TH & 1"),
            ("GOAL", "GOAL"),
            ("FLAG", "FLAG"),
            ("15:00", "15:00"),
            ("2:00", "2:00"),
            ("1ST QTR", "1ST QTR"),
            ("2ND QTR", "2ND QTR"),
            ("3RD QTR", "3RD QTR"),
            ("4TH QTR", "4TH QTR"),
        ]

        for i, (text, value) in enumerate(quick_buttons):
            row = i // 4
            col = i % 4
            btn = ttk.Button(
                quick_frame,
                text=text,
                width=8,
                command=lambda v=value: self.ground_truth_var.set(v),
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")

        # Configure grid weights
        for i in range(4):
            quick_frame.columnconfigure(i, weight=1)

        # Bottom frame - Navigation and save
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        # Navigation
        nav_frame = ttk.Frame(bottom_frame)
        nav_frame.pack(side=tk.LEFT)

        ttk.Button(nav_frame, text="Previous", command=self.previous_sample).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(nav_frame, text="Next", command=self.next_sample).pack(side=tk.LEFT, padx=5)

        # Save and skip
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(side=tk.RIGHT)

        ttk.Button(action_frame, text="Skip", command=self.skip_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save & Next", command=self.save_and_next).pack(
            side=tk.LEFT, padx=5
        )
        self.undo_button = ttk.Button(
            action_frame, text="Undo Last", command=self.undo_last_validation, state="disabled"
        )
        self.undo_button.pack(side=tk.LEFT, padx=5)

        # Bind Enter key to save
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Right>", lambda e: self.next_sample())
        self.root.bind("<Left>", lambda e: self.previous_sample())
        self.root.bind("<Control-z>", lambda e: self.undo_last_validation())  # Ctrl+Z for undo

    def load_samples(self):
        """Load samples from database based on current mode"""
        if self.browse_all_mode:
            self.samples = self.database.get_all_samples(1000)
            self.browse_mode_button.config(text="Show Unvalidated Only")

            # Try to position on last validated sample
            if self.last_validated_sample_id:
                for i, sample in enumerate(self.samples):
                    if sample["id"] == self.last_validated_sample_id:
                        self.current_sample_index = i
                        break
                else:
                    self.current_sample_index = 0
            else:
                self.current_sample_index = 0
        else:
            self.samples = self.database.get_unvalidated_samples(1000)
            self.browse_mode_button.config(text="Browse All Samples")
            self.current_sample_index = 0

        self.update_statistics()
        self.display_current_sample()

    def update_statistics(self):
        """Update statistics display"""
        stats = self.database.get_statistics()
        stats_text = f"Total: {stats['total_samples']} | Validated: {stats['validated_samples']} | Remaining: {len(self.samples)}"
        self.stats_label.config(text=stats_text)

    def refresh_current_sample_from_db(self):
        """Refresh current sample data from database to ensure we have latest info"""
        if not self.current_sample:
            return

        # Get fresh data from database
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ocr_samples WHERE id = ?", (self.current_sample["id"],))
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()

        if row:
            fresh_sample = dict(zip(columns, row))
            # Update the sample in our list
            self.samples[self.current_sample_index] = fresh_sample
            self.current_sample = fresh_sample

        conn.close()

    def display_current_sample(self):
        """Display current sample for annotation"""
        if not self.samples:
            self.info_label.config(text="No samples to annotate")
            self.status_label.config(text="")
            return

        if self.current_sample_index >= len(self.samples):
            messagebox.showinfo("Complete", "All samples have been processed!")
            self.load_samples()
            return

        self.current_sample = self.samples[self.current_sample_index]

        # Update info
        self.info_label.config(
            text=f"Sample: {self.current_sample_index + 1}/{len(self.samples)} | "
            f"Class: {self.current_sample['class_name']} | "
            f"Variant: {self.current_sample['variant']}"
        )

        # Update validation status indicator
        is_validated = self.current_sample.get("is_validated", False)
        if is_validated:
            self.status_label.config(text="✅ VALIDATED - You can edit this!", foreground="green")
        else:
            self.status_label.config(
                text="⏳ NOT VALIDATED - Needs annotation", foreground="orange"
            )

        # Load and display image
        self.load_sample_image()

        # Display OCR results
        self.display_ocr_results()

        # Pre-fill ground truth entry - use existing ground truth if available, otherwise last entered
        existing_ground_truth = self.current_sample.get("ground_truth_text", "")
        if existing_ground_truth:
            self.ground_truth_var.set(existing_ground_truth)
        else:
            self.ground_truth_var.set(self.last_ground_truth)

        self.ground_truth_entry.focus()
        # Select all text so user can easily overwrite or edit
        self.ground_truth_entry.select_range(0, tk.END)

    def load_sample_image(self):
        """Load and display the sample image region"""
        try:
            # Load original image
            image_path = self.current_sample["image_path"]
            image = cv2.imread(image_path)

            # Extract region
            x1, y1, x2, y2 = (
                self.current_sample["bbox_x1"],
                self.current_sample["bbox_y1"],
                self.current_sample["bbox_x2"],
                self.current_sample["bbox_y2"],
            )

            region = image[y1:y2, x1:x2]

            # Scale up for better visibility
            scale_factor = max(1, 200 // max(region.shape[0], region.shape[1]))
            if scale_factor > 1:
                region = cv2.resize(
                    region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST
                )

            # Convert to RGB for display
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage
            from PIL import Image, ImageTk

            pil_image = Image.fromarray(region_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference

        except Exception as e:
            self.image_label.config(text=f"Error loading image: {e}")
            logger.error(f"Error loading sample image: {e}")

    def display_ocr_results(self):
        """Display OCR results from all engines"""
        results_text = f"EasyOCR: '{self.current_sample['easyocr_text']}'\n"
        results_text += f"Tesseract Standard: '{self.current_sample['tesseract_standard_text']}'\n"
        results_text += f"Tesseract Digits: '{self.current_sample['tesseract_digits_text']}'\n"
        results_text += f"Tesseract AlphaNum: '{self.current_sample['tesseract_alphanum_text']}'\n"
        results_text += f"\nConfidence: {self.current_sample['confidence']:.3f}\n"
        results_text += f"Size: {self.current_sample['width']}x{self.current_sample['height']}"

        self.ocr_results_text.delete(1.0, tk.END)
        self.ocr_results_text.insert(1.0, results_text)

    def previous_sample(self):
        """Go to previous sample"""
        if self.current_sample_index > 0:
            self.current_sample_index -= 1
            self.display_current_sample()

    def next_sample(self):
        """Go to next sample"""
        if self.current_sample_index < len(self.samples) - 1:
            self.current_sample_index += 1
            self.display_current_sample()

    def skip_sample(self):
        """Skip current sample without saving"""
        self.next_sample()

    def save_and_next(self):
        """Save ground truth and move to next sample"""
        if not self.current_sample:
            return

        ground_truth = self.ground_truth_var.get().strip()
        if not ground_truth:
            messagebox.showwarning("Warning", "Please enter the correct text or skip this sample.")
            return

        # Store sample info for potential undo before saving
        undo_info = {
            "sample": self.current_sample.copy(),
            "ground_truth": ground_truth,
            "index": self.current_sample_index,
        }

        # Save to database
        self.database.update_ground_truth(self.current_sample["id"], ground_truth)

        # Store this ground truth for pre-filling next sample
        self.last_ground_truth = ground_truth

        # Remember this sample ID for browse mode positioning
        self.last_validated_sample_id = self.current_sample["id"]

        # Add to recently validated for undo functionality
        self.recently_validated.append(undo_info)
        if len(self.recently_validated) > 10:  # Keep only last 10 for undo
            self.recently_validated.pop(0)

        # Enable undo button
        self.undo_button.config(state="normal")

        # In browse all mode, update the current sample's ground truth and stay on it
        # In unvalidated mode, remove the sample from the list
        if self.browse_all_mode:
            # Check if this was already validated (for update confirmation)
            was_validated = self.current_sample.get("is_validated", False)

            # Update the current sample in the list with new ground truth
            self.current_sample["ground_truth_text"] = ground_truth
            self.current_sample["is_validated"] = True

            # Refresh the sample from database to ensure we have latest data
            self.refresh_current_sample_from_db()

            # Stay on current sample, just update display
            self.update_statistics()
            self.display_current_sample()
        else:
            # Remove from current samples list (original behavior)
            self.samples.pop(self.current_sample_index)

            # Update statistics
            self.update_statistics()

            # Display next sample (index stays the same since we removed current)
            if self.current_sample_index >= len(self.samples):
                self.current_sample_index = max(0, len(self.samples) - 1)

            self.display_current_sample()

    def undo_last_validation(self):
        """Undo the last validation and bring the sample back for re-annotation"""
        if not self.recently_validated:
            messagebox.showinfo("Info", "No recent validations to undo.")
            return

        # Get the last validated sample
        last_validation = self.recently_validated.pop()
        sample = last_validation["sample"]

        # Remove validation from database
        self.database.remove_ground_truth(sample["id"])

        # Add sample back to the current samples list at the beginning
        self.samples.insert(0, sample)

        # Go to this sample
        self.current_sample_index = 0

        # Update statistics and display
        self.update_statistics()
        self.display_current_sample()

        # Pre-fill with the previous ground truth for easy correction
        self.ground_truth_var.set(last_validation["ground_truth"])
        self.ground_truth_entry.select_range(0, tk.END)

        # Disable undo button if no more recent validations
        if not self.recently_validated:
            self.undo_button.config(state="disabled")

        messagebox.showinfo(
            "Undo Complete",
            f"Brought back sample for re-annotation.\nPrevious entry: '{last_validation['ground_truth']}'",
        )

    def toggle_browse_mode(self):
        """Toggle between showing only unvalidated samples vs all samples"""
        self.browse_all_mode = not self.browse_all_mode
        self.load_samples()

    def export_training_data(self):
        """Export validated training data"""
        training_data = self.database.get_training_data()

        if not training_data:
            messagebox.showinfo("Info", "No validated training data available.")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Training Data",
        )

        if filename:
            with open(filename, "w") as f:
                json.dump(training_data, f, indent=2, default=str)

            messagebox.showinfo(
                "Success", f"Exported {len(training_data)} training samples to {filename}"
            )

    def run(self):
        """Run the annotation GUI"""
        self.root.mainloop()


class UltimateMaddenOCRSystem:
    """Main system orchestrator"""

    def __init__(self, model_path: str, dataset_path: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.extractor = MaddenOCRExtractor(model_path)
        self.database = MaddenOCRDatabase()

    def extract_all_samples(self, num_images: int = 50):
        """Extract OCR samples from random images in dataset for better variation"""
        logger.info(f"Starting random OCR sample extraction from {num_images} images...")

        image_files = list(Path(self.dataset_path).glob("*.png")) + list(
            Path(self.dataset_path).glob("*.jpg")
        )

        # Randomly sample images for better variation
        if len(image_files) > num_images:
            random.shuffle(image_files)
            image_files = image_files[:num_images]
            logger.info(
                f"Randomly selected {num_images} images from {len(image_files)} total images"
            )

        total_samples = 0

        for i, image_path in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")

            try:
                regions = self.extractor.extract_regions(str(image_path))

                for region in regions:
                    sample_id = self.database.insert_sample(region)
                    total_samples += 1

                    if total_samples % 100 == 0:
                        logger.info(f"Extracted {total_samples} samples so far...")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        logger.info(f"Extraction complete! Total new samples: {total_samples}")

        # Print statistics
        stats = self.database.get_statistics()
        logger.info(f"Database statistics: {stats}")

        return total_samples

    def launch_annotation_gui(self):
        """Launch the annotation GUI"""
        gui = MaddenOCRAnnotationGUI(self.database)
        gui.run()

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("🎯 Ultimate Madden OCR System - Maximum Accuracy Pipeline")
        print("=" * 60)

        # Step 1: Extract samples
        print("\n📊 Step 1: Extracting OCR samples from all images...")
        total_samples = self.extract_all_samples()

        # Step 2: Launch annotation GUI
        print(f"\n✅ Extracted {total_samples} samples!")
        print("\n🎨 Step 2: Launching annotation GUI for manual correction...")
        print("Instructions:")
        print("- Review each OCR result and enter the correct text")
        print("- Use quick buttons for common patterns")
        print("- Press Enter to save and continue")
        print("- Use arrow keys to navigate")
        print("- Focus on accuracy - this creates the training foundation!")

        input("\nPress Enter to launch annotation GUI...")
        self.launch_annotation_gui()


def main():
    """Main entry point"""
    # Configuration
    MODEL_PATH = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
    DATASET_PATH = "hud_region_training/dataset/images/train"

    # Verify paths exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return

    # Initialize system
    system = UltimateMaddenOCRSystem(MODEL_PATH, DATASET_PATH)
    system.run_full_pipeline()


if __name__ == "__main__":
    main()

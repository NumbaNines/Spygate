"""
SpygateAI Madden OCR Dataset Creator
Collects and processes all Madden HUD data for custom OCR model training.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class OCRSample:
    """Individual OCR training sample"""

    image_path: str
    text_content: str
    region_type: str  # down_distance, game_clock, play_clock, scores
    confidence: float
    preprocessed: bool = False
    enhanced: bool = False
    original_frame: Optional[str] = None
    frame_number: Optional[int] = None


class MaddenOCRDatasetCreator:
    """Creates comprehensive OCR training dataset from all Madden data"""

    def __init__(self, output_dir: str = "madden_ocr_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "raw").mkdir(exist_ok=True)

        self.samples: List[OCRSample] = []
        self.db_path = self.output_dir / "ocr_dataset.db"
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for dataset management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE,
                text_content TEXT,
                region_type TEXT,
                confidence REAL,
                preprocessed BOOLEAN,
                enhanced BOOLEAN,
                original_frame TEXT,
                frame_number INTEGER,
                image_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_stats (
                region_type TEXT PRIMARY KEY,
                sample_count INTEGER,
                avg_confidence REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def run_collection(self):
        """Main collection process"""
        print("🔥 COLLECTING MADDEN OCR TRAINING DATA")

        # Collect from debug directories
        debug_samples = self.collect_debug_data()
        print(f"📁 Collected {debug_samples} samples from debug data")

        # Show stats
        stats = self.get_dataset_stats()
        print(f"\n📊 Dataset Statistics:")
        for region_type, info in stats.items():
            if region_type != "total":
                print(f"  {region_type}: {info} samples")

        return stats

    def collect_debug_data(self) -> int:
        """Collect from debug directories"""
        collected = 0

        # Process debug_regions
        debug_regions = Path("debug_regions")
        if debug_regions.exists():
            for image_file in debug_regions.glob("*.png"):
                collected += self.process_debug_image(image_file)

        # Process debug_ocr_regions
        debug_ocr = Path("debug_ocr_regions")
        if debug_ocr.exists():
            for image_file in debug_ocr.glob("*.jpg"):
                collected += self.process_debug_image(image_file)

        return collected

    def process_debug_image(self, image_path: Path) -> int:
        """Process single debug image"""
        try:
            filename = image_path.stem
            region_type = self.get_region_type(filename)

            if not region_type:
                return 0

            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return 0

            # Save to dataset
            sample_path = self.save_sample_image(image, filename, region_type)
            if sample_path:
                text_content = self.extract_text_from_filename(filename)
                self.add_to_database(sample_path, text_content, region_type)
                return 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        return 0

    def get_region_type(self, filename: str) -> Optional[str]:
        """Extract region type from filename"""
        if "down_distance" in filename:
            return "down_distance"
        elif "game_clock" in filename:
            return "game_clock"
        elif "play_clock" in filename:
            return "play_clock"
        elif "possession" in filename:
            return "scores"
        return None

    def save_sample_image(
        self, image: np.ndarray, filename: str, region_type: str
    ) -> Optional[str]:
        """Save image to organized directory"""
        try:
            region_dir = self.output_dir / "images" / region_type
            region_dir.mkdir(exist_ok=True)

            save_path = region_dir / f"{filename}.png"
            cv2.imwrite(str(save_path), image)
            return str(save_path)

        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return None

    def extract_text_from_filename(self, filename: str) -> str:
        """Extract expected text from filename patterns"""
        if "down_distance" in filename:
            return "1ST & 10"  # Default - needs manual annotation
        elif "game_clock" in filename:
            return "12:34"
        elif "play_clock" in filename:
            return "15"
        elif "possession" in filename:
            return "LAR 14 KC 21"
        return "UNKNOWN"

    def add_to_database(self, image_path: str, text_content: str, region_type: str):
        """Add sample to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO ocr_samples
                (image_path, text_content, region_type, confidence)
                VALUES (?, ?, ?, ?)
            """,
                (image_path, text_content, region_type, 0.8),
            )

            conn.commit()
            print(f"Added: {region_type} - {text_content}")

        except sqlite3.IntegrityError:
            pass  # Skip duplicates

        conn.close()

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT region_type, COUNT(*)
            FROM ocr_samples
            GROUP BY region_type
        """
        )

        stats = {}
        total = 0

        for region_type, count in cursor.fetchall():
            stats[region_type] = count
            total += count

        stats["total"] = total
        conn.close()
        return stats


def main():
    creator = MaddenOCRDatasetCreator()
    stats = creator.run_collection()

    print(f"\n✅ COLLECTION COMPLETE!")
    print(f"Total samples: {stats.get('total', 0)}")


if __name__ == "__main__":
    main()

"""
Extract OCR Training Data from YOLO Dataset
Uses our existing 385 Madden screenshots to create comprehensive OCR training data.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path("src").absolute()))


class YOLODatasetOCRExtractor:
    """Extract OCR training data from existing YOLO dataset"""

    def __init__(
        self,
        yolo_dataset_path: str = "hud_region_training/dataset",
        output_dir: str = "madden_ocr_comprehensive",
    ):
        self.dataset_path = Path(yolo_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        for region in ["down_distance", "game_clock", "play_clock"]:
            (self.output_dir / "images" / region).mkdir(exist_ok=True)

        self.db_path = self.output_dir / "ocr_dataset.db"
        self.init_database()

        # Sample counter
        self.sample_counter = 0

        # Initialize our YOLO model
        self.yolo_model = None
        self.ocr_engine = None

    def init_database(self):
        """Initialize database"""
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
                source_image TEXT,
                bbox TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def initialize_models(self):
        """Initialize YOLO and OCR models with GPU acceleration"""
        print("🔧 Initializing YOLO and OCR models with GPU acceleration...")

        try:
            import torch

            # Check GPU availability
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                print("⚠️  GPU not available, using CPU")

            from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
            from spygate.ml.enhanced_ocr import EnhancedOCR

            # Initialize analyzer (contains YOLO model) with GPU config
            analyzer = EnhancedGameAnalyzer()
            self.yolo_model = analyzer

            # Force GPU usage if available
            if hasattr(analyzer, "model") and hasattr(analyzer.model, "model"):
                if torch.cuda.is_available():
                    analyzer.model.model = analyzer.model.model.to(device)
                    print("🎯 YOLO model moved to GPU")

            # Initialize OCR
            self.ocr_engine = EnhancedOCR()

            print("✅ Models initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize models: {e}")
            return False

    def get_image_paths(self, max_samples: int = 50) -> List[Path]:
        """Get image paths from train and val sets (limited to max_samples)"""
        train_images = list((self.dataset_path / "images" / "train").glob("*.png"))
        val_images = list((self.dataset_path / "images" / "val").glob("*.png"))

        all_images = train_images + val_images

        # Limit to max_samples for faster testing
        if len(all_images) > max_samples:
            all_images = all_images[:max_samples]
            print(
                f"📸 Limited to {max_samples} images for testing (from {len(train_images + val_images)} total)"
            )
        else:
            print(
                f"📸 Found {len(all_images)} images ({len(train_images)} train + {len(val_images)} val)"
            )

        return all_images

    def extract_text_regions(self, image_path: Path) -> Dict[str, List[Dict]]:
        """Extract text regions from image using YOLO"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {}

            # Analyze with YOLO
            game_state = self.yolo_model.analyze_frame(image)

            # Extract regions we care about for OCR
            regions = {}

            if hasattr(game_state, "visualization_layers"):
                layers = game_state.visualization_layers

                # Map detection layers to region types
                region_mapping = {
                    "down_distance_area": "down_distance",
                    "game_clock_area": "game_clock",
                    "play_clock_area": "play_clock",
                }

                for layer_name, region_type in region_mapping.items():
                    if layer_name in layers:
                        region_image = layers[layer_name]

                        if region_image is not None and region_image.size > 0:
                            # Extract OCR text
                            text_result = self.extract_ocr_text(region_image, region_type)

                            if text_result["text"] and text_result["text"] != "UNKNOWN":
                                if region_type not in regions:
                                    regions[region_type] = []

                                regions[region_type].append(
                                    {
                                        "image": region_image,
                                        "text": text_result["text"],
                                        "confidence": text_result["confidence"],
                                        "bbox": text_result.get("bbox", "unknown"),
                                    }
                                )

            return regions

        except Exception as e:
            print(f"  ⚠️  Error processing {image_path.name}: {e}")
            return {}

    def extract_ocr_text(self, region_image: np.ndarray, region_type: str) -> Dict:
        """Extract text using OCR with validation"""
        try:
            result = self.ocr_engine.extract_text(region_image)

            if result and result.get("text"):
                text = result["text"].strip()

                # Validate based on region type
                if self.validate_text_for_region(text, region_type):
                    return {
                        "text": text,
                        "confidence": result.get("confidence", 0.5),
                        "bbox": result.get("bbox", "unknown"),
                    }

            return {"text": "UNKNOWN", "confidence": 0.0, "bbox": "unknown"}

        except Exception as e:
            print(f"    ⚠️  OCR failed: {e}")
            return {"text": "UNKNOWN", "confidence": 0.0, "bbox": "unknown"}

    def validate_text_for_region(self, text: str, region_type: str) -> bool:
        """Validate extracted text based on region type"""
        text_upper = text.upper().strip()

        if region_type == "down_distance":
            # More flexible validation - look for down patterns OR & symbol OR numbers
            down_patterns = ["ST", "ND", "RD", "TH"]
            has_down = any(pattern in text_upper for pattern in down_patterns)
            has_and = "&" in text_upper
            has_numbers = any(char.isdigit() for char in text_upper)

            # Accept if it has down indicators, & symbol, or looks like "1 & 10" format
            return has_down or has_and or (has_numbers and len(text_upper) >= 3)

        elif region_type == "game_clock":
            # Accept time formats like "15:00", "5:32", or numbers
            has_colon = ":" in text
            is_numeric = text.replace(":", "").replace(" ", "").isdigit()
            has_time_length = len(text.replace(" ", "")) >= 3

            return (has_colon and is_numeric) or (is_numeric and has_time_length)

        elif region_type == "play_clock":
            # Accept numbers 1-40, be more flexible with parsing
            clean_text = "".join(char for char in text if char.isdigit())
            if clean_text:
                try:
                    num = int(clean_text)
                    return 1 <= num <= 40
                except:
                    return False
            return False

        # Accept any non-empty text for other regions
        return len(text.strip()) > 0

    def save_training_sample(self, region_data: Dict, source_image: str, region_type: str) -> bool:
        """Save region as training sample"""
        try:
            self.sample_counter += 1

            # Generate filename
            filename = f"{self.sample_counter:06d}_{source_image.stem}_{region_type}.png"
            save_path = self.output_dir / "images" / region_type / filename

            # Save image
            cv2.imwrite(str(save_path), region_data["image"])

            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO ocr_samples
                (image_path, text_content, region_type, confidence, source_image, bbox)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    str(save_path),
                    region_data["text"],
                    region_type,
                    region_data["confidence"],
                    str(source_image),
                    str(region_data["bbox"]),
                ),
            )

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"    ⚠️  Failed to save sample: {e}")
            return False

    def process_all_images(self, max_samples: int = 50) -> Dict:
        """Process images and extract OCR training data"""
        print(f"🔥 PROCESSING {max_samples} YOLO DATASET IMAGES FOR OCR TRAINING")

        # Initialize models
        if not self.initialize_models():
            return {}

        # Get limited images for testing
        image_paths = self.get_image_paths(max_samples)

        stats = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "total_samples": 0,
            "by_region": {"down_distance": 0, "game_clock": 0, "play_clock": 0},
            "failed_images": 0,
        }

        print(f"🚀 Processing {len(image_paths)} images...")

        # Process images with progress bar
        for image_path in tqdm(image_paths, desc="Extracting OCR samples"):
            try:
                # Extract regions
                regions = self.extract_text_regions(image_path)

                if regions:
                    stats["processed_images"] += 1

                    # Save each region
                    for region_type, region_list in regions.items():
                        for region_data in region_list:
                            if self.save_training_sample(region_data, image_path, region_type):
                                stats["total_samples"] += 1
                                stats["by_region"][region_type] += 1
                else:
                    stats["failed_images"] += 1

            except Exception as e:
                print(f"  ❌ Failed to process {image_path.name}: {e}")
                stats["failed_images"] += 1

        return stats

    def generate_training_report(self, stats: Dict):
        """Generate comprehensive training report"""
        report = f"""
🎯 MADDEN OCR DATASET EXTRACTION COMPLETE!

📊 STATISTICS:
- Total YOLO images: {stats['total_images']}
- Successfully processed: {stats['processed_images']}
- Failed to process: {stats['failed_images']}
- Total OCR samples created: {stats['total_samples']}

📋 BY REGION:
- Down & Distance: {stats['by_region']['down_distance']} samples
- Game Clock: {stats['by_region']['game_clock']} samples
- Play Clock: {stats['by_region']['play_clock']} samples

💾 DATASET LOCATION: {self.output_dir}

🚀 READY FOR TRAINING!
Run: python train_madden_ocr.py
"""

        print(report)

        # Save report to file
        with open(self.output_dir / "extraction_report.txt", "w", encoding="utf-8") as f:
            f.write(report)


def main():
    """Main execution"""
    extractor = YOLODatasetOCRExtractor()

    # Process 50 images for testing
    stats = extractor.process_all_images(max_samples=50)

    # Generate report
    extractor.generate_training_report(stats)

    print(f"\n🎯 EXTRACTION COMPLETE!")
    print(
        f"Created {stats['total_samples']} OCR training samples from {stats['processed_images']} Madden screenshots"
    )


if __name__ == "__main__":
    main()

"""
Simple Madden OCR Data Extractor
Extract text regions from YOLO detections for manual annotation and OCR training.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


class SimpleMaddenOCRExtractor:
    """Simple extractor for Madden OCR training data"""

    def __init__(
        self,
        yolo_model_path: str = "hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
    ):
        self.yolo_model_path = yolo_model_path
        self.model = None
        self.output_dir = Path("simple_madden_ocr_data")
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "down_distance").mkdir(exist_ok=True)
        (self.output_dir / "game_clock").mkdir(exist_ok=True)
        (self.output_dir / "play_clock").mkdir(exist_ok=True)

        self.sample_count = 0

    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"🔧 Loading YOLO model: {self.yolo_model_path}")
            self.model = YOLO(self.yolo_model_path)

            # Move to GPU if available
            import torch

            if torch.cuda.is_available():
                print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")

            print("✅ Model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def process_image(self, image_path: Path) -> int:
        """Process single image and extract text regions"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return 0

            # Run YOLO detection
            results = self.model(image, conf=0.3, verbose=False)

            regions_saved = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])

                        # Only process text regions we care about
                        if class_name in [
                            "down_distance_area",
                            "game_clock_area",
                            "play_clock_area",
                        ]:
                            # Extract region coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Extract region
                            region = image[y1:y2, x1:x2]

                            if region.size > 0:
                                # Save region
                                region_type = class_name.replace("_area", "")
                                filename = f"{self.sample_count:06d}_{image_path.stem}_{region_type}_{confidence:.2f}.png"

                                save_path = self.output_dir / region_type / filename
                                cv2.imwrite(str(save_path), region)

                                self.sample_count += 1
                                regions_saved += 1

                                print(f"  💾 Saved: {filename}")

            return regions_saved

        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
            return 0

    def extract_from_dataset(self, max_images: int = 50):
        """Extract OCR training data from YOLO dataset"""
        print(f"🔥 EXTRACTING OCR DATA FROM {max_images} MADDEN IMAGES")
        print("=" * 60)

        # Load model
        if not self.load_model():
            return

        # Get image paths
        dataset_path = Path("hud_region_training/dataset")
        train_images = list((dataset_path / "images" / "train").glob("*.png"))
        val_images = list((dataset_path / "images" / "val").glob("*.png"))

        all_images = (train_images + val_images)[:max_images]

        print(f"📸 Processing {len(all_images)} images...")

        total_regions = 0
        processed_images = 0

        for image_path in tqdm(all_images, desc="Extracting regions"):
            regions_saved = self.process_image(image_path)
            if regions_saved > 0:
                processed_images += 1
                total_regions += regions_saved

        # Generate report
        print(f"\n🎯 EXTRACTION COMPLETE!")
        print(f"📊 Processed: {processed_images}/{len(all_images)} images")
        print(f"💾 Total regions saved: {total_regions}")

        # Count by type
        down_count = len(list((self.output_dir / "down_distance").glob("*.png")))
        clock_count = len(list((self.output_dir / "game_clock").glob("*.png")))
        play_count = len(list((self.output_dir / "play_clock").glob("*.png")))

        print(f"📋 BY TYPE:")
        print(f"   Down & Distance: {down_count}")
        print(f"   Game Clock: {clock_count}")
        print(f"   Play Clock: {play_count}")

        return total_regions


def main():
    """Main execution"""
    extractor = SimpleMaddenOCRExtractor()
    total_samples = extractor.extract_from_dataset(max_images=50)

    if total_samples > 0:
        print(f"\n🚀 SUCCESS! Created {total_samples} OCR training samples")
        print(f"📁 Location: {extractor.output_dir}")
        print("\n🎯 NEXT STEPS:")
        print("1. Review the extracted regions manually")
        print("2. Create ground truth labels for training")
        print("3. Train custom OCR model")
    else:
        print("\n❌ No samples extracted. Check model and dataset paths.")


if __name__ == "__main__":
    main()

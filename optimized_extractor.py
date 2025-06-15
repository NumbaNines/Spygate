#!/usr/bin/env python3
"""
Optimized Random Extractor for 20,000 Sample Target
Maximum efficiency and variation for 100% OCR accuracy
Avoids re-processing already annotated images
"""

import random
import time
from pathlib import Path

from ultimate_madden_ocr_system import MaddenOCRDatabase, UltimateMaddenOCRSystem


class OptimizedExtractor:
    def __init__(self, model_path: str, dataset_path: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.system = UltimateMaddenOCRSystem(model_path, dataset_path)
        self.db = MaddenOCRDatabase()

    def get_processed_images(self):
        """Get list of images that have already been processed"""
        import sqlite3

        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT image_path FROM ocr_samples")
        processed_paths = [row[0] for row in cursor.fetchall()]

        conn.close()

        # Convert to just filenames for comparison
        processed_filenames = set()
        for path in processed_paths:
            filename = Path(path).name
            processed_filenames.add(filename)

        return processed_filenames

    def get_unprocessed_images(self):
        """Get list of images that haven't been processed yet"""
        # Get all available images
        all_images = list(Path(self.dataset_path).glob("*.png")) + list(
            Path(self.dataset_path).glob("*.jpg")
        )

        # Get already processed images
        processed_filenames = self.get_processed_images()

        # Filter out already processed images
        unprocessed_images = []
        for img_path in all_images:
            if img_path.name not in processed_filenames:
                unprocessed_images.append(img_path)

        print(f"📊 Image Analysis:")
        print(f"  Total images available: {len(all_images)}")
        print(f"  Already processed: {len(processed_filenames)}")
        print(f"  Unprocessed remaining: {len(unprocessed_images)}")

        return unprocessed_images

    def get_image_batches(self, total_images: int, batch_size: int = 50):
        """Split UNPROCESSED images into random batches for processing"""
        # Get only unprocessed images
        unprocessed_images = self.get_unprocessed_images()

        if not unprocessed_images:
            print("⚠️  No unprocessed images remaining!")
            return []

        # Shuffle for maximum randomness
        random.shuffle(unprocessed_images)

        # Limit to requested amount or available amount
        images_to_process = unprocessed_images[: min(total_images, len(unprocessed_images))]

        # Create batches
        batches = []
        for i in range(0, len(images_to_process), batch_size):
            batch = images_to_process[i : i + batch_size]
            batches.append(batch)

        return batches

    def extract_batch(self, image_batch, batch_num: int, total_batches: int):
        """Extract samples from a batch of images"""
        print(f"\n🔄 Processing Batch {batch_num}/{total_batches}")
        print(f"📁 Images in batch: {len(image_batch)}")

        start_time = time.time()
        total_samples = 0

        for i, image_path in enumerate(image_batch):
            print(f"  📸 {i+1}/{len(image_batch)}: {image_path.name}")

            try:
                regions = self.system.extractor.extract_regions(str(image_path))

                for region in regions:
                    sample_id = self.db.insert_sample(region)
                    total_samples += 1

            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue

        batch_time = time.time() - start_time
        print(f"  ✅ Batch complete: {total_samples} samples in {batch_time:.1f}s")

        return total_samples

    def show_progress_stats(self):
        """Show current progress toward 20,000 target"""
        stats = self.db.get_statistics()
        total = stats["total_samples"]
        validated = stats["validated_samples"]
        remaining = 20000 - total

        print(f"\n📊 Progress Update:")
        print(f"  Total samples: {total:,}")
        print(f"  Validated: {validated:,}")
        print(f"  Unvalidated: {total - validated:,}")
        print(f"  Target: 20,000")
        print(f"  Remaining: {remaining:,}")
        print(f"  Progress: {(total/20000)*100:.1f}%")

        return remaining

    def extract_toward_target(self, target: int = 20000, batch_size: int = 50):
        """Extract samples in batches toward target"""
        print("🎯 Optimized Extractor for 100% Accuracy")
        print("=" * 50)

        # Check current progress
        remaining = self.show_progress_stats()

        if remaining <= 0:
            print("🎉 Target already reached!")
            return

        # Check available unprocessed images
        unprocessed_images = self.get_unprocessed_images()

        if not unprocessed_images:
            print("⚠️  No unprocessed images remaining!")
            print("   All available images have been processed.")
            return

        # Calculate how many images we need
        # Estimate ~30 samples per image (10 per class × 3 classes)
        images_needed = max(50, remaining // 25)  # Conservative estimate
        images_available = len(unprocessed_images)
        images_to_process = min(images_needed, images_available)

        print(f"\n🎯 Plan: Extract {images_to_process} unprocessed images")
        print(f"📦 Batch size: {batch_size} images per batch")
        print(f"🔍 Available unprocessed: {images_available}")

        # Get random batches from unprocessed images only
        batches = self.get_image_batches(images_to_process, batch_size)

        if not batches:
            print("❌ No batches to process!")
            return

        print(f"📋 Total batches: {len(batches)}")

        # Process each batch
        total_new_samples = 0
        for i, batch in enumerate(batches, 1):
            batch_samples = self.extract_batch(batch, i, len(batches))
            total_new_samples += batch_samples

            # Show progress after each batch
            remaining = self.show_progress_stats()

            # Check if we've reached target
            if remaining <= 0:
                print("🎉 Target reached!")
                break

            # Ask if user wants to continue after each batch
            if i < len(batches):
                continue_choice = input(f"\nContinue with batch {i+1}? (y/n/auto): ").lower()
                if continue_choice == "n":
                    break
                elif continue_choice == "auto":
                    print("🚀 Auto mode - processing all remaining batches...")
                    # Continue without asking

        print(f"\n✅ Extraction session complete!")
        print(f"📈 New samples added: {total_new_samples:,}")
        self.show_progress_stats()


def main():
    # Configuration
    MODEL_PATH = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
    DATASET_PATH = "hud_region_training/dataset/images/train"

    extractor = OptimizedExtractor(MODEL_PATH, DATASET_PATH)
    extractor.extract_toward_target()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Random Sample Extractor for Madden OCR Training
Extracts samples from random images for better variation
"""

from ultimate_madden_ocr_system import MaddenOCRDatabase, UltimateMaddenOCRSystem


def main():
    # Configuration
    MODEL_PATH = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
    DATASET_PATH = "hud_region_training/dataset/images/train"

    print("🎯 Random Madden OCR Sample Extractor")
    print("=" * 50)

    # Check current stats
    db = MaddenOCRDatabase()
    stats = db.get_statistics()
    print(f"Current samples: {stats['total_samples']}")
    print(f"Validated: {stats['validated_samples']}")
    print(f"Unvalidated: {stats['total_samples'] - stats['validated_samples']}")

    # Ask how many images to process
    try:
        num_images = int(input("\nHow many random images to process? (default 50): ") or "50")
    except ValueError:
        num_images = 50

    print(f"\n📊 Extracting from {num_images} random images...")

    # Run extraction
    system = UltimateMaddenOCRSystem(MODEL_PATH, DATASET_PATH)
    new_samples = system.extract_all_samples(num_images)

    # Show final stats
    final_stats = db.get_statistics()
    print(f"\n✅ Extraction Complete!")
    print(f"New samples added: {new_samples}")
    print(f"Total samples: {final_stats['total_samples']}")
    print(
        f"Ready for annotation: {final_stats['total_samples'] - final_stats['validated_samples']}"
    )


if __name__ == "__main__":
    main()

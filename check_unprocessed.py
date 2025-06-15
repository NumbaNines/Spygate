#!/usr/bin/env python3
"""Check how many unprocessed images remain"""

from optimized_extractor import OptimizedExtractor


def main():
    MODEL_PATH = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
    DATASET_PATH = "hud_region_training/dataset/images/train"

    print("🔍 Checking for unprocessed images...")

    extractor = OptimizedExtractor(MODEL_PATH, DATASET_PATH)
    unprocessed = extractor.get_unprocessed_images()

    print(f"\n✅ Analysis complete!")
    print(f"📸 Unprocessed images available: {len(unprocessed)}")

    if len(unprocessed) > 0:
        print(f"🎯 Ready to extract ~{len(unprocessed) * 25} new samples")
        print(f"   (estimated 25 samples per image)")
    else:
        print("⚠️  All images have been processed!")


if __name__ == "__main__":
    main()

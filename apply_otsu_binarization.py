import time
from pathlib import Path

import cv2
import numpy as np


def apply_otsu_binarization():
    """Apply Otsu's thresholding for binarization to all sample images"""

    # Get sample images directory
    sample_dir = Path("preprocessing_test_samples")
    if not sample_dir.exists():
        print(f"❌ Error: {sample_dir} directory not found!")
        return

    image_files = list(sample_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No PNG files found in {sample_dir}")
        return

    print(f"🔧 Applying Otsu's thresholding binarization to {len(image_files)} images")
    print(f"📊 Method: cv2.THRESH_BINARY + cv2.THRESH_OTSU")
    print(f"📁 Target directory: {sample_dir}")

    processed_count = 0
    threshold_values = []
    start_time = time.time()

    for img_path in image_files:
        try:
            # Load image (grayscale, already preprocessed)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path.name}")
                continue

            # Get original dimensions for reference
            height, width = img.shape

            # Apply Otsu's thresholding
            threshold_value, binary_img = cv2.threshold(
                img,
                0,  # threshold value (ignored with OTSU)
                255,  # max value assigned to pixel
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            # Store threshold value for analysis
            threshold_values.append(threshold_value)

            # Overwrite the original file
            cv2.imwrite(str(img_path), binary_img)

            processed_count += 1
            print(f"✅ {img_path.name}: {width}x{height} - Otsu threshold: {threshold_value:.1f}")

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    elapsed_time = time.time() - start_time

    # Calculate threshold statistics
    if threshold_values:
        avg_threshold = np.mean(threshold_values)
        min_threshold = np.min(threshold_values)
        max_threshold = np.max(threshold_values)
        std_threshold = np.std(threshold_values)
    else:
        avg_threshold = min_threshold = max_threshold = std_threshold = 0

    print(f"\n🎯 OTSU'S BINARIZATION COMPLETE")
    print(f"✅ Successfully processed: {processed_count}/{len(image_files)} images")
    print(f"⚡ Total time: {elapsed_time:.2f} seconds")
    print(f"💾 Files overwritten in: {sample_dir}")

    print(f"\n📊 OTSU THRESHOLD ANALYSIS:")
    print(f"   Average threshold: {avg_threshold:.1f}")
    print(f"   Min threshold: {min_threshold:.1f}")
    print(f"   Max threshold: {max_threshold:.1f}")
    print(f"   Std deviation: {std_threshold:.1f}")

    print(f"\n📈 COMPLETE PREPROCESSING PIPELINE:")
    print(f"   1. ✅ Grayscale conversion")
    print(f"   2. ✅ LANCZOS4 scaling (1.5x)")
    print(f"   3. ✅ CLAHE enhancement (clip=2.0, grid=4x4)")
    print(f"   4. ✅ Light Gaussian blur (3x3, σ=0.5)")
    print(f"   5. ✅ Otsu's binarization (adaptive thresholding)")
    print(f"\n🚀 Images are now FULLY OPTIMIZED for OCR!")
    print(f"💡 Binary images provide maximum contrast for text detection")


if __name__ == "__main__":
    apply_otsu_binarization()

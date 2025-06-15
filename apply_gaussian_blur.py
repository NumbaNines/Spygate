import time
from pathlib import Path

import cv2
import numpy as np


def apply_gaussian_blur():
    """Apply light Gaussian blur (3x3 kernel, low sigma) to all sample images"""

    # Gaussian blur parameters
    kernel_size = (3, 3)  # 3x3 kernel
    sigma_x = 0.5  # Low sigma for light blur
    sigma_y = 0.5  # Low sigma for light blur

    # Get sample images directory
    sample_dir = Path("preprocessing_test_samples")
    if not sample_dir.exists():
        print(f"❌ Error: {sample_dir} directory not found!")
        return

    image_files = list(sample_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No PNG files found in {sample_dir}")
        return

    print(f"🔧 Applying light Gaussian blur to {len(image_files)} images")
    print(f"📊 Settings: kernel_size={kernel_size}, sigma_x={sigma_x}, sigma_y={sigma_y}")
    print(f"📁 Target directory: {sample_dir}")

    processed_count = 0
    start_time = time.time()

    for img_path in image_files:
        try:
            # Load image (grayscale, already CLAHE enhanced)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path.name}")
                continue

            # Get original dimensions for reference
            height, width = img.shape

            # Apply light Gaussian blur
            blurred_img = cv2.GaussianBlur(img, kernel_size, sigma_x, sigmaY=sigma_y)

            # Overwrite the original file
            cv2.imwrite(str(img_path), blurred_img)

            processed_count += 1
            print(f"✅ {img_path.name}: {width}x{height} - Gaussian blur applied")

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    elapsed_time = time.time() - start_time

    print(f"\n🎯 GAUSSIAN BLUR APPLICATION COMPLETE")
    print(f"✅ Successfully processed: {processed_count}/{len(image_files)} images")
    print(f"⚡ Total time: {elapsed_time:.2f} seconds")
    print(f"🔧 Blur settings: kernel={kernel_size}, sigma_x={sigma_x}, sigma_y={sigma_y}")
    print(f"💾 Files overwritten in: {sample_dir}")
    print(f"\n📈 Images now have optimal preprocessing pipeline:")
    print(f"   1. ✅ Grayscale conversion")
    print(f"   2. ✅ LANCZOS4 scaling (1.5x)")
    print(f"   3. ✅ CLAHE enhancement (clip=2.0, grid=4x4)")
    print(f"   4. ✅ Light Gaussian blur (3x3, σ=0.5)")
    print(f"\n🚀 Ready for maximum OCR accuracy!")


if __name__ == "__main__":
    apply_gaussian_blur()

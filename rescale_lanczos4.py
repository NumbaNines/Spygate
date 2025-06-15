import os
import time
from pathlib import Path

import cv2


def rescale_images_lanczos4():
    """Rescale images to 1.5x using LANCZOS4 interpolation and overwrite existing files"""

    # Source directory with current scaled images
    source_dir = Path("preprocessing_test_samples")

    if not source_dir.exists():
        print(f"❌ Error: {source_dir} directory not found!")
        return

    # Get all PNG files
    image_files = list(source_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No PNG files found in {source_dir}")
        return

    print(f"🔄 Rescaling {len(image_files)} images using LANCZOS4 interpolation...")
    print(f"📁 Target directory: {source_dir}")

    processed_count = 0
    start_time = time.time()

    for img_path in image_files:
        try:
            # Load the current image (grayscale)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path.name}")
                continue

            # Get current dimensions
            height, width = img.shape

            # Calculate new dimensions (1.5x scale)
            new_width = int(width * 1.5)
            new_height = int(height * 1.5)

            # Rescale using LANCZOS4 interpolation
            rescaled_img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )

            # Overwrite the existing file
            cv2.imwrite(str(img_path), rescaled_img)

            processed_count += 1
            print(f"✅ {img_path.name}: {width}x{height} → {new_width}x{new_height}")

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    elapsed_time = time.time() - start_time

    print(f"\n🎯 RESCALING COMPLETE")
    print(f"✅ Successfully processed: {processed_count}/{len(image_files)} images")
    print(f"⚡ Total time: {elapsed_time:.2f} seconds")
    print(f"🔧 Interpolation method: LANCZOS4")
    print(f"📏 Scale factor: 1.5x")
    print(f"💾 Files overwritten in: {source_dir}")


if __name__ == "__main__":
    rescale_images_lanczos4()

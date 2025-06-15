import time
from pathlib import Path

import cv2
import numpy as np


def apply_clahe_gpu(image, clip_limit=2.0, grid_size=(4, 4)):
    """Apply CLAHE using GPU acceleration"""
    try:
        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)

        # Create CLAHE object for GPU
        clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        # Apply CLAHE on GPU
        gpu_result = cv2.cuda_GpuMat()
        clahe.apply(gpu_img, gpu_result)

        # Download result from GPU
        result = gpu_result.download()
        return result, True
    except:
        return None, False


def apply_clahe_cpu(image, clip_limit=2.0, grid_size=(4, 4)):
    """Apply CLAHE using CPU"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def apply_optimal_clahe():
    """Apply optimal CLAHE settings (clip=2.0, grid=4x4) to all sample images"""

    # Optimal parameters from testing
    clip_limit = 2.0
    grid_size = (4, 4)

    # Check GPU availability
    test_img = np.ones((100, 100), dtype=np.uint8) * 128
    _, gpu_available = apply_clahe_gpu(test_img, clip_limit, grid_size)

    if gpu_available:
        print("✅ Using GPU CLAHE acceleration")
    else:
        print("⚠️  Using CPU CLAHE (GPU not available)")

    # Get sample images directory
    sample_dir = Path("preprocessing_test_samples")
    if not sample_dir.exists():
        print(f"❌ Error: {sample_dir} directory not found!")
        return

    image_files = list(sample_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No PNG files found in {sample_dir}")
        return

    print(f"🔧 Applying optimal CLAHE to {len(image_files)} images")
    print(f"📊 Settings: clip_limit={clip_limit}, grid_size={grid_size}")
    print(f"📁 Target directory: {sample_dir}")

    processed_count = 0
    start_time = time.time()

    for img_path in image_files:
        try:
            # Load image (grayscale)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Warning: Could not load {img_path.name}")
                continue

            # Get original dimensions for reference
            height, width = img.shape

            # Apply optimal CLAHE
            if gpu_available:
                enhanced_img, success = apply_clahe_gpu(img, clip_limit, grid_size)
                if not success:
                    enhanced_img = apply_clahe_cpu(img, clip_limit, grid_size)
            else:
                enhanced_img = apply_clahe_cpu(img, clip_limit, grid_size)

            # Overwrite the original file
            cv2.imwrite(str(img_path), enhanced_img)

            processed_count += 1
            print(f"✅ {img_path.name}: {width}x{height} - CLAHE applied")

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    elapsed_time = time.time() - start_time

    print(f"\n🎯 OPTIMAL CLAHE APPLICATION COMPLETE")
    print(f"✅ Successfully processed: {processed_count}/{len(image_files)} images")
    print(f"⚡ Total time: {elapsed_time:.2f} seconds")
    print(f"🔧 CLAHE settings applied: clip={clip_limit}, grid={grid_size}")
    print(f"🚀 Processing method: {'GPU' if gpu_available else 'CPU'}")
    print(f"💾 Files overwritten in: {sample_dir}")
    print(f"\n📈 Images are now optimized for maximum OCR accuracy!")


if __name__ == "__main__":
    apply_optimal_clahe()

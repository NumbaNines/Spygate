#!/usr/bin/env python3
"""
Extract Grayscale Samples for Preprocessing Optimization
Extract 25 screenshots from YOLO dataset, convert to grayscale, and prepare for systematic preprocessing testing
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def extract_grayscale_samples():
    """Extract 25 random screenshots from YOLO dataset and convert to grayscale"""

    # Source directory
    source_dir = Path("hud_region_training/dataset/images/train")

    # Output directory
    output_dir = Path("preprocessing_test_samples")
    output_dir.mkdir(exist_ok=True)

    print("🎯 EXTRACTING GRAYSCALE SAMPLES FOR PREPROCESSING OPTIMIZATION")
    print("=" * 70)

    # Get all PNG files from training dataset
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    all_images = list(source_dir.glob("*.png"))

    if len(all_images) == 0:
        print(f"❌ No PNG files found in {source_dir}")
        return

    print(f"📸 Found {len(all_images)} images in YOLO training dataset")

    # Randomly select 25 images
    num_samples = min(25, len(all_images))
    selected_images = random.sample(all_images, num_samples)

    print(f"🎲 Randomly selected {num_samples} images for preprocessing optimization")
    print()

    successful_conversions = 0

    for i, image_path in enumerate(selected_images, 1):
        try:
            print(f"Processing {i:2d}/{num_samples}: {image_path.name}")

            # Load original image
            original = cv2.imread(str(image_path))
            if original is None:
                print(f"  ❌ Failed to load image")
                continue

            # Convert to grayscale
            grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            # Create output filename
            output_filename = f"sample_{i:02d}_{image_path.stem}_grayscale.png"
            output_path = output_dir / output_filename

            # Save grayscale image
            success = cv2.imwrite(str(output_path), grayscale)

            if success:
                print(f"  ✅ Converted to grayscale: {output_filename}")
                successful_conversions += 1
            else:
                print(f"  ❌ Failed to save grayscale image")

        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")

    print()
    print("📊 CONVERSION SUMMARY:")
    print(f"   Total processed: {num_samples}")
    print(f"   Successful conversions: {successful_conversions}")
    print(f"   Output directory: {output_dir}")

    if successful_conversions > 0:
        print()
        print("🚀 NEXT STEPS:")
        print("   1. Use these grayscale samples for systematic preprocessing testing")
        print("   2. Test each preprocessing technique individually")
        print("   3. Find optimal thresholds for each technique")
        print("   4. Build the best preprocessing pipeline")

        # List the created files
        print()
        print("📁 CREATED GRAYSCALE SAMPLES:")
        for sample_file in sorted(output_dir.glob("sample_*_grayscale.png")):
            file_size = sample_file.stat().st_size / 1024  # KB
            print(f"   - {sample_file.name} ({file_size:.1f} KB)")

    return output_dir, successful_conversions


def create_preprocessing_test_info():
    """Create an info file about the preprocessing test setup"""

    info_content = """# Preprocessing Optimization Test Setup

## Purpose
Systematic testing of preprocessing techniques to find optimal parameters for PaddleOCR on Madden HUD elements.

## Test Samples
- 25 randomly selected screenshots from YOLO training dataset
- Converted to grayscale for consistent testing
- Representative of actual Madden HUD content

## Preprocessing Techniques to Test
1. **Contrast Enhancement** (cv2.convertScaleAbs)
   - Alpha: 0.5 - 3.0 (contrast multiplier)
   - Beta: 0 - 100 (brightness offset)

2. **Gaussian Blur** (cv2.GaussianBlur)
   - Kernel size: 1 - 15 (noise reduction)

3. **Sharpening** (Unsharp masking)
   - Strength: 0.0 - 2.0 (edge enhancement)

4. **Upscaling** (cv2.resize)
   - Scale factor: 1.0 - 5.0 (size increase)
   - Interpolation: CUBIC, LANCZOS4

5. **Gamma Correction** (LUT)
   - Gamma: 0.3 - 3.0 (brightness curve)

6. **Morphological Operations**
   - Opening, Closing, Gradient
   - Kernel sizes: 3x3, 5x5, 7x7

7. **Adaptive Thresholding**
   - ADAPTIVE_THRESH_MEAN_C
   - ADAPTIVE_THRESH_GAUSSIAN_C

## Testing Strategy
1. Test each technique individually on all 25 samples
2. Measure OCR confidence and text detection count
3. Find optimal parameters for each technique
4. Test combinations of best-performing techniques
5. Build final optimized preprocessing pipeline

## Success Metrics
- OCR confidence score (higher is better)
- Number of texts detected (more is better)
- Accuracy of detected text (manual verification)
- Processing speed (faster is better)
"""

    info_path = Path("preprocessing_test_samples/README.md")
    with open(info_path, "w") as f:
        f.write(info_content)

    print(f"📝 Created test info file: {info_path}")


def main():
    """Main execution function"""

    # Set random seed for reproducible results
    random.seed(42)

    # Extract grayscale samples
    output_dir, successful_conversions = extract_grayscale_samples()

    if successful_conversions > 0:
        # Create info file
        create_preprocessing_test_info()

        print()
        print("✅ GRAYSCALE SAMPLE EXTRACTION COMPLETE!")
        print(f"📁 Ready for preprocessing optimization in: {output_dir}")
        print()
        print("🔧 NEXT: Run systematic preprocessing tests to find optimal parameters")


if __name__ == "__main__":
    main()

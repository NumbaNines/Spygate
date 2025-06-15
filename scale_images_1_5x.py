#!/usr/bin/env python3
"""
Scale Images to 1.5x
Scale all 25 grayscale sample images to 1.5x and replace the original files
"""

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def scale_images_to_1_5x():
    """Scale all grayscale sample images to 1.5x and replace originals"""

    samples_dir = Path("preprocessing_test_samples")
    backup_dir = Path("preprocessing_test_samples_backup")

    print("🔧 SCALING IMAGES TO 1.5x")
    print("=" * 40)

    # Check if samples directory exists
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return

    # Get all grayscale sample images
    sample_images = list(samples_dir.glob("sample_*_grayscale.png"))

    if not sample_images:
        print(f"❌ No grayscale samples found in {samples_dir}")
        return

    print(f"📸 Found {len(sample_images)} grayscale samples")
    print(f"🎯 Scaling all images to 1.5x using CUBIC interpolation")
    print()

    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    print(f"💾 Creating backup in: {backup_dir}")

    successful_scales = 0
    failed_scales = 0

    for i, image_path in enumerate(sorted(sample_images), 1):
        try:
            print(f"[{i:2d}/25] Processing: {image_path.name}")

            # Create backup of original
            backup_path = backup_dir / image_path.name
            shutil.copy2(image_path, backup_path)
            print(f"         ✅ Backed up to: {backup_path.name}")

            # Load original grayscale image
            original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if original is None:
                print(f"         ❌ Failed to load image")
                failed_scales += 1
                continue

            # Get original dimensions
            orig_height, orig_width = original.shape
            print(f"         📏 Original size: {orig_width}x{orig_height}")

            # Calculate new dimensions (1.5x scale)
            new_width = int(orig_width * 1.5)
            new_height = int(orig_height * 1.5)

            # Scale image using CUBIC interpolation
            scaled = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"         🔍 Scaled size: {new_width}x{new_height}")

            # Save scaled image (replace original)
            success = cv2.imwrite(str(image_path), scaled)

            if success:
                print(f"         ✅ Scaled and saved successfully")
                successful_scales += 1
            else:
                print(f"         ❌ Failed to save scaled image")
                failed_scales += 1

        except Exception as e:
            print(f"         ❌ Error processing {image_path.name}: {e}")
            failed_scales += 1

    print()
    print("📊 SCALING SUMMARY:")
    print(f"   Total images: {len(sample_images)}")
    print(f"   Successfully scaled: {successful_scales}")
    print(f"   Failed: {failed_scales}")
    print(f"   Backup location: {backup_dir}")

    if successful_scales > 0:
        print()
        print("✅ SCALING COMPLETE!")
        print("🔍 All grayscale samples are now scaled to 1.5x")
        print("💾 Original images backed up for safety")
        print()
        print("📁 UPDATED SAMPLES:")

        # Show updated file sizes
        for sample_file in sorted(samples_dir.glob("sample_*_grayscale.png")):
            file_size = sample_file.stat().st_size / 1024  # KB

            # Load to get dimensions
            img = cv2.imread(str(sample_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                height, width = img.shape
                print(f"   - {sample_file.name}")
                print(f"     Size: {width}x{height} ({file_size:.1f} KB)")

    return successful_scales, failed_scales


def verify_scaling():
    """Verify that images were properly scaled"""
    print("\n🔍 VERIFYING SCALING...")

    samples_dir = Path("preprocessing_test_samples")
    backup_dir = Path("preprocessing_test_samples_backup")

    sample_images = list(samples_dir.glob("sample_*_grayscale.png"))
    backup_images = list(backup_dir.glob("sample_*_grayscale.png"))

    if not sample_images or not backup_images:
        print("❌ Cannot verify - missing files")
        return

    # Compare first image as example
    sample_img = cv2.imread(str(sample_images[0]), cv2.IMREAD_GRAYSCALE)
    backup_img = cv2.imread(str(backup_images[0]), cv2.IMREAD_GRAYSCALE)

    if sample_img is not None and backup_img is not None:
        sample_h, sample_w = sample_img.shape
        backup_h, backup_w = backup_img.shape

        scale_w = sample_w / backup_w
        scale_h = sample_h / backup_h

        print(f"📏 Verification (using {sample_images[0].name}):")
        print(f"   Original: {backup_w}x{backup_h}")
        print(f"   Scaled: {sample_w}x{sample_h}")
        print(f"   Scale factor: {scale_w:.2f}x width, {scale_h:.2f}x height")

        if abs(scale_w - 1.5) < 0.01 and abs(scale_h - 1.5) < 0.01:
            print("   ✅ Scaling verified - images are properly scaled to 1.5x")
        else:
            print("   ⚠️  Scaling may not be exactly 1.5x")


def main():
    """Main execution function"""
    print("🚀 Starting 1.5x Image Scaling")
    print("This will scale all 25 grayscale samples to 1.5x size")
    print("and replace the original files (with backup).")
    print()

    # Confirm action
    response = input("⚠️  This will replace your original grayscale samples. Continue? (y/N): ")
    if response.lower() != "y":
        print("❌ Operation cancelled.")
        return

    # Scale images
    successful, failed = scale_images_to_1_5x()

    if successful > 0:
        # Verify scaling
        verify_scaling()

        print()
        print("🎯 NEXT STEPS:")
        print("   1. Your grayscale samples are now 1.5x larger")
        print("   2. Use these for further preprocessing optimization")
        print("   3. Original images are safely backed up")
        print("   4. Test other preprocessing techniques on these scaled images")


if __name__ == "__main__":
    main()

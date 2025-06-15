#!/usr/bin/env python3
"""Check image formats to find problematic images"""

import json
import os
from pathlib import Path

import cv2

# Load the data
with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
    training_data = json.load(f)

print(f"📊 Checking {len(training_data)} image files...")

problematic_images = []
valid_images = 0
missing_files = 0

for i, sample in enumerate(training_data[:100]):  # Check first 100
    image_path = sample.get("image_path", "")

    if not image_path:
        print(f"❌ Sample {i}: No image path")
        continue

    if not os.path.exists(image_path):
        missing_files += 1
        print(f"❌ Sample {i}: File not found: {image_path}")
        continue

    try:
        # Try to load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as color first
        if image is None:
            problematic_images.append((i, image_path, "Failed to load"))
            print(f"❌ Sample {i}: Failed to load: {image_path}")
            continue

        print(f"📷 Sample {i}: {image.shape} - {image_path}")

        # Try grayscale conversion
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"  -> Grayscale: {gray.shape}")
        else:
            print(f"  -> Already grayscale: {image.shape}")

        # Try the preprocessing steps one by one
        try:
            # Step 1: Bilateral filter
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            filtered = cv2.bilateralFilter(image, 9, 75, 75)
            print(f"  -> Bilateral filter: {filtered.shape}")

            # Step 2: CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            print(f"  -> CLAHE: {enhanced.shape}")

            # Step 3: Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            print(f"  -> Morphology: {morphed.shape}")

            # Step 4: Resize
            resized = cv2.resize(morphed, (256, 64), interpolation=cv2.INTER_CUBIC)
            print(f"  -> Resized: {resized.shape}")

            valid_images += 1

        except Exception as e:
            problematic_images.append((i, image_path, f"Preprocessing error: {e}"))
            print(f"❌ Sample {i}: Preprocessing failed: {e}")

    except Exception as e:
        problematic_images.append((i, image_path, f"Load error: {e}"))
        print(f"❌ Sample {i}: Load error: {e}")

print(f"\n📈 Summary:")
print(f"  Valid images: {valid_images}")
print(f"  Missing files: {missing_files}")
print(f"  Problematic images: {len(problematic_images)}")

if problematic_images:
    print(f"\n🚨 Problematic images:")
    for i, path, error in problematic_images:
        print(f"  Sample {i}: {error} - {path}")
else:
    print(f"\n✅ All checked images are valid!")

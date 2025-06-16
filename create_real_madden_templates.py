#!/usr/bin/env python3
"""
Create Real Madden Templates by cropping down/distance areas from full screenshots.
"""

from pathlib import Path

import cv2
import numpy as np


def create_templates_from_real_madden():
    """Create templates by cropping real Madden screenshots."""
    print("🎮 Creating Templates from REAL Madden Screenshots")
    print("=" * 60)

    # Create output directory
    output_dir = Path("down_templates_real")
    output_dir.mkdir(exist_ok=True)

    # Source directory with real Madden screenshots
    source_dir = Path("down templates")

    # Mapping of real files to template names
    file_mapping = {
        # Normal downs
        "1.png": "1ST.png",
        "2.png": "2ND.png",
        "3rd.png": "3RD.png",
        "4th.png": "4TH.png",
        # GOAL situations
        "1st goal.png": "1ST_GOAL.png",
        "2nd goal.png": "2ND_GOAL.png",
        "real 2ndgoal.png": "2ND_GOAL_ALT.png",  # Alternative if needed
        "3rd goal.png": "3RD_GOAL.png",
        "4th goal.png": "4TH_GOAL.png",
    }

    # Down/distance area coordinates (from SpygateAI memory)
    # These are the actual coordinates used in SpygateAI for down_distance_area
    crop_coords = {
        "x": 1300,  # Left edge of down_distance_area
        "y": 50,  # Top edge of down_distance_area
        "width": 125,  # Width (1425 - 1300 = 125)
        "height": 50,  # Height (100 - 50 = 50)
    }

    print(
        f"📍 Using crop coordinates: x={crop_coords['x']}, y={crop_coords['y']}, "
        f"width={crop_coords['width']}, height={crop_coords['height']}"
    )
    print()

    success_count = 0

    for source_file, template_name in file_mapping.items():
        source_path = source_dir / source_file
        output_path = output_dir / template_name

        if not source_path.exists():
            print(f"⚠️  {source_file} not found, skipping...")
            continue

        # Load the full screenshot
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"❌ Failed to load {source_file}")
            continue

        print(f"📸 Processing {source_file} -> {template_name}")
        print(f"   Original size: {img.shape[1]}x{img.shape[0]}")

        # Crop the down/distance area
        x, y, w, h = crop_coords["x"], crop_coords["y"], crop_coords["width"], crop_coords["height"]

        # Make sure crop is within image bounds
        if x + w > img.shape[1] or y + h > img.shape[0]:
            print(f"   ⚠️  Crop coordinates exceed image bounds, adjusting...")
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

        cropped = img[y : y + h, x : x + w]

        if cropped.size == 0:
            print(f"   ❌ Empty crop result")
            continue

        # Save the template
        cv2.imwrite(str(output_path), cropped)
        print(f"   ✅ Created template: {cropped.shape[1]}x{cropped.shape[0]}")
        success_count += 1

    print()
    print(f"🎯 SUMMARY: Created {success_count} templates from real Madden screenshots")

    # Show what we created
    print("\n📋 Created Templates:")
    for template_file in sorted(output_dir.glob("*.png")):
        img = cv2.imread(str(template_file))
        if img is not None:
            print(f"   {template_file.name}: {img.shape[1]}x{img.shape[0]}")

    return success_count > 0


if __name__ == "__main__":
    create_templates_from_real_madden()

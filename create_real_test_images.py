#!/usr/bin/env python3
"""
Create real test images by cropping down numbers from full Madden screenshots.
"""

from pathlib import Path

import cv2


def create_real_test_images():
    """Crop real down numbers from full Madden screenshots for testing."""
    print("🎮 Creating Real Test Images from Full Madden Screenshots")
    print("=" * 60)

    # Source: Full Madden screenshots
    source_dir = Path("down templates")

    # Destination: Test images directory
    dest_dir = Path("templates/raw_gameplay")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # File mapping: source screenshot → test image name
    file_mapping = {
        # Normal downs
        "1.png": "1st_10.png",
        "2.png": "2nd_7.png",
        "3rd.png": "3rd_3.png",
        "4th.png": "4th_1.png",
        # GOAL situations
        "1st goal.png": "1st_goal.png",  # 1ST & GOAL (if needed)
        "real 2ndgoal.png": "2nd_goal.png",  # 2ND & GOAL (use the good one)
        "3rd goal.png": "3rd_goal.png",  # 3RD & GOAL
        "4th goal.png": "4th_goal.png",  # 4TH & GOAL
    }

    # Use the same crop coordinates that worked for creating templates
    # (These should be the coordinates you used in the down template creator)
    crop_coords = {
        "x": 1300,  # Adjust based on your actual coordinates
        "y": 50,  # Adjust based on your actual coordinates
        "width": 125,  # Adjust based on your actual coordinates
        "height": 50,  # Adjust based on your actual coordinates
    }

    print("🔧 Processing real Madden screenshots...")

    for source_file, dest_file in file_mapping.items():
        source_path = source_dir / source_file
        dest_path = dest_dir / dest_file

        if not source_path.exists():
            print(f"   ⚠️ {source_file}: Source file not found, skipping")
            continue

        # Load the full screenshot
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"   ❌ {source_file}: Failed to load image")
            continue

        # Crop the down number area
        x, y, w, h = crop_coords["x"], crop_coords["y"], crop_coords["width"], crop_coords["height"]
        cropped = img[y : y + h, x : x + w]

        # Save the cropped down number
        cv2.imwrite(str(dest_path), cropped)
        print(f"   ✅ {source_file} → {dest_file} ({cropped.shape[1]}x{cropped.shape[0]})")

    print("\n🎯 Real test images created! Now the template detection should work properly.")
    print("📋 Next step: Run the test again with real Madden down numbers!")


if __name__ == "__main__":
    create_real_test_images()

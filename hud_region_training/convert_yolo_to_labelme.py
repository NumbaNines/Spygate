"""
Convert existing 5-class YOLO labels to labelme JSON format.
This allows you to build on existing annotations and just add the 3 new classes.
"""

import json
import os
from pathlib import Path

from PIL import Image

# 5-class mapping (existing)
YOLO_TO_LABELME_CLASSES = {
    0: "hud",
    1: "possession_triangle_area",
    2: "territory_triangle_area",
    3: "preplay_indicator",
    4: "play_call_screen",
}


def yolo_to_labelme(yolo_file, image_file, output_file):
    """Convert a single YOLO annotation file to labelme JSON format."""

    # Get image dimensions
    with Image.open(image_file) as img:
        img_width, img_height = img.size

    # Read YOLO annotations
    shapes = []

    if os.path.exists(yolo_file):
        with open(yolo_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert normalized coordinates to pixel coordinates
                x1 = (center_x - width / 2) * img_width
                y1 = (center_y - height / 2) * img_height
                x2 = (center_x + width / 2) * img_width
                y2 = (center_y + height / 2) * img_height

                # Create labelme shape
                if class_id in YOLO_TO_LABELME_CLASSES:
                    shape = {
                        "label": YOLO_TO_LABELME_CLASSES[class_id],
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                    shapes.append(shape)

    # Create labelme JSON structure
    labelme_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_file),
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
    }

    # Write labelme JSON file
    with open(output_file, "w") as f:
        json.dump(labelme_data, f, indent=2)

    return len(shapes)


def convert_directory(yolo_labels_dir, images_dir, output_dir):
    """Convert all YOLO labels in a directory to labelme format."""

    yolo_path = Path(yolo_labels_dir)
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    if not yolo_path.exists():
        print(f"❌ YOLO labels directory not found: {yolo_labels_dir}")
        return

    if not images_path.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    # Get all image files
    image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))

    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return

    converted_count = 0
    total_annotations = 0

    for image_file in image_files:
        # Find corresponding YOLO label file
        yolo_file = yolo_path / f"{image_file.stem}.txt"
        output_file = output_path / f"{image_file.stem}.json"

        try:
            annotations_count = yolo_to_labelme(yolo_file, image_file, output_file)
            total_annotations += annotations_count
            converted_count += 1

            if annotations_count > 0:
                print(f"✅ Converted {image_file.name}: {annotations_count} existing annotations")
            else:
                print(f"📝 Created empty {image_file.name}: ready for new annotations")

        except Exception as e:
            print(f"❌ Error converting {image_file.name}: {e}")

    print(f"\n📊 Conversion Summary:")
    print(f"Images processed: {converted_count}/{len(image_files)}")
    print(f"Total existing annotations: {total_annotations}")
    print(f"Ready to add 3 new classes: down_distance_area, game_clock_area, play_clock_area")


def main():
    # Paths for conversion
    yolo_labels_dir = "hud_region_training/dataset/labels/train"
    images_dir = "hud_region_training/hud_region_training_8class/datasets_8class/train/images"
    output_dir = images_dir  # Save JSON files in same directory as images

    print("Converting existing 5-class YOLO labels to labelme JSON format...")
    print(f"YOLO labels: {yolo_labels_dir}")
    print(f"Images: {images_dir}")
    print(f"Output: {output_dir}")
    print()

    convert_directory(yolo_labels_dir, images_dir, output_dir)


if __name__ == "__main__":
    main()

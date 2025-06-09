#!/usr/bin/env python3
"""
Convert labelme JSON annotations to YOLO format for SpygateAI HUD training.

This script converts labelme JSON files to YOLO format for training the YOLOv8 model
with the 4 classes: hud, qb_position, left_hash_mark, right_hash_mark
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple


def load_classes(classes_file: str) -> dict:
    """Load class names and create class to index mapping."""
    try:
        with open(classes_file) as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        print(f"Loaded {len(classes)} classes: {classes}")
        return class_to_idx
    except FileNotFoundError:
        print(f"Classes file not found: {classes_file}")
        return {}


def polygon_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert polygon points to bounding box (x_min, y_min, x_max, y_max)."""
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, y_min, x_max, y_max


def convert_to_yolo_format(
    bbox: tuple[float, float, float, float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Convert bounding box to YOLO format (normalized center_x, center_y, width, height)."""
    x_min, y_min, x_max, y_max = bbox

    # Calculate center point and dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize to [0, 1]
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height

    return center_x, center_y, width, height


def convert_labelme_json(json_file: str, class_to_idx: dict, output_dir: str) -> bool:
    """Convert a single labelme JSON file to YOLO format."""
    try:
        with open(json_file) as f:
            data = json.load(f)

        # Get image dimensions
        img_height = data["imageHeight"]
        img_width = data["imageWidth"]

        # Extract annotations
        annotations = []
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in class_to_idx:
                print(f"Warning: Unknown class '{label}' in {json_file}, skipping...")
                continue

            class_idx = class_to_idx[label]
            points = shape["points"]

            # Convert polygon to bounding box
            bbox = polygon_to_bbox(points)

            # Convert to YOLO format
            yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)

            # Format: class_id center_x center_y width height
            annotation = f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            annotations.append(annotation)

        # Save YOLO format file
        json_path = Path(json_file)
        output_file = Path(output_dir) / f"{json_path.stem}.txt"

        with open(output_file, "w") as f:
            f.write("\n".join(annotations))

        print(f"Converted {json_file} -> {output_file} ({len(annotations)} annotations)")
        return True

    except Exception as e:
        print(f"Error converting {json_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert labelme JSON to YOLO format")
    parser.add_argument(
        "--input-dir",
        default="training_data/images",
        help="Directory containing labelme JSON files",
    )
    parser.add_argument(
        "--output-dir", default="training_data/labels", help="Directory to save YOLO format files"
    )
    parser.add_argument(
        "--classes-file", default="training_data/classes.txt", help="File containing class names"
    )

    args = parser.parse_args()

    # Load classes
    class_to_idx = load_classes(args.classes_file)
    if not class_to_idx:
        print("No classes loaded. Please check your classes file.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSON files
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_path}")
        print("Make sure you've saved your labelme annotations as JSON files.")
        return

    print(f"Found {len(json_files)} JSON files to convert...")

    # Convert each JSON file
    success_count = 0
    for json_file in json_files:
        if convert_labelme_json(str(json_file), class_to_idx, args.output_dir):
            success_count += 1

    print(f"\nConversion complete: {success_count}/{len(json_files)} files successfully converted")
    print(f"YOLO format files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

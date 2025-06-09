#!/usr/bin/env python3
"""Test dataset configuration and label format for YOLO training."""

import os
from pathlib import Path

import yaml


def test_dataset_config():
    """Test the dataset configuration."""
    config_path = "training_data/dataset.yaml"

    print("Testing dataset configuration...")

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"✅ Config loaded: {config}")

    # Check required fields
    required_fields = ["names", "nc", "path", "train", "val"]
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False

    print(f"✅ All required fields present")

    # Check if paths exist
    base_path = Path(config["path"])
    train_path = base_path / config["train"]
    labels_path = base_path / "labels"

    print(f"Base path: {base_path}")
    print(f"Train path: {train_path}")
    print(f"Labels path: {labels_path}")

    if not train_path.exists():
        print(f"❌ Train images path doesn't exist: {train_path}")
        return False
    if not labels_path.exists():
        print(f"❌ Labels path doesn't exist: {labels_path}")
        return False

    print(f"✅ Paths exist")

    # Count images and labels
    image_files = list(train_path.glob("*.png")) + list(train_path.glob("*.jpg"))
    label_files = list(labels_path.glob("*.txt"))

    print(f"Found {len(image_files)} images")
    print(f"Found {len(label_files)} label files")

    # Check if we have corresponding labels for images
    matched_pairs = 0
    unmatched_images = []

    for img_file in image_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            matched_pairs += 1
        else:
            unmatched_images.append(img_file.name)

    print(f"✅ Matched pairs: {matched_pairs}")
    if unmatched_images:
        print(f"⚠️ Unmatched images: {len(unmatched_images)}")
        for img in unmatched_images[:5]:  # Show first 5
            print(f"   - {img}")
        if len(unmatched_images) > 5:
            print(f"   ... and {len(unmatched_images) - 5} more")

    # Check sample label format
    if label_files:
        sample_label = label_files[0]
        print(f"\nChecking sample label: {sample_label.name}")

        with open(sample_label) as f:
            lines = f.readlines()

        print(f"Lines in label: {len(lines)}")
        for i, line in enumerate(lines[:3]):  # Show first 3 lines
            parts = line.strip().split()
            print(f"  Line {i+1}: {line.strip()}")
            if len(parts) == 5:
                class_id, x, y, w, h = parts
                try:
                    class_id = int(class_id)
                    x, y, w, h = float(x), float(y), float(w), float(h)

                    # Check if values are in valid range
                    if 0 <= class_id < config["nc"]:
                        print(f"    ✅ Valid class ID: {class_id}")
                    else:
                        print(f"    ❌ Invalid class ID: {class_id} (should be 0-{config['nc']-1})")

                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        print(
                            f"    ✅ Valid coordinates: x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}"
                        )
                    else:
                        print(
                            f"    ❌ Invalid coordinates (should be 0-1): x={x}, y={y}, w={w}, h={h}"
                        )

                except ValueError as e:
                    print(f"    ❌ Error parsing numbers: {e}")
            else:
                print(f"    ❌ Wrong number of values: {len(parts)} (should be 5)")

    return True


def test_yolo_import():
    """Test if YOLO can be imported and basic functionality works."""
    try:
        from ultralytics import YOLO

        print("✅ YOLO imported successfully")

        # Try to load a basic model
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8n model loaded successfully")

        return True
    except Exception as e:
        print(f"❌ YOLO import/loading failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET VALIDATION TEST")
    print("=" * 60)

    # Test YOLO import
    if not test_yolo_import():
        print("\n❌ YOLO import failed - cannot proceed with training")
        exit(1)

    print("\n" + "-" * 60)

    # Test dataset
    if test_dataset_config():
        print("\n✅ Dataset validation passed!")
        print("You can now proceed with training.")
    else:
        print("\n❌ Dataset validation failed!")
        print("Please fix the issues above before training.")

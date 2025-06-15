"""
Train 8-class YOLOv8 model for enhanced HUD detection.
Uses the annotated data from labelme conversion.
"""

import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def setup_training_environment():
    """Set up the training environment and data structure."""

    # Create runs directory for this training
    runs_dir = Path("hud_region_training/hud_region_training_8class/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create validation split (use 20% of training data)
    train_images = Path(
        "hud_region_training/hud_region_training_8class/datasets_8class/train/images"
    )
    train_labels = Path(
        "hud_region_training/hud_region_training_8class/datasets_8class/train/labels"
    )
    val_images = Path("hud_region_training/hud_region_training_8class/datasets_8class/val/images")
    val_labels = Path("hud_region_training/hud_region_training_8class/datasets_8class/val/labels")

    # Get list of all training files
    image_files = list(train_images.glob("*.png"))

    # Take every 5th image for validation (20% split)
    val_files = image_files[::5]

    print(f"Total images: {len(image_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Training images: {len(image_files) - len(val_files)}")

    # Move validation files
    for img_file in val_files:
        # Move image
        val_img_path = val_images / img_file.name
        if not val_img_path.exists():
            shutil.copy2(img_file, val_img_path)

        # Move corresponding label
        label_file = train_labels / f"{img_file.stem}.txt"
        val_label_path = val_labels / f"{img_file.stem}.txt"
        if label_file.exists() and not val_label_path.exists():
            shutil.copy2(label_file, val_label_path)

    return len(image_files) - len(val_files), len(val_files)


def create_dataset_yaml():
    """Create the dataset configuration file."""

    dataset_config = {
        "path": str(
            Path("hud_region_training/hud_region_training_8class/datasets_8class").absolute()
        ),
        "train": "train/images",
        "val": "val/images",
        "nc": 8,
        "names": [
            "hud",
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ],
    }

    yaml_path = Path("hud_region_training/hud_region_training_8class/dataset_8class_training.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"✅ Created dataset config: {yaml_path}")
    return yaml_path


def train_8class_model():
    """Train the 8-class YOLOv8 model."""

    print("🚀 Starting 8-Class YOLOv8 Training")
    print("=" * 50)

    # Setup environment
    train_count, val_count = setup_training_environment()

    # Create dataset config
    dataset_yaml = create_dataset_yaml()

    # Load base model (start from YOLOv8n for faster training)
    model = YOLO("yolov8n.pt")

    # Training parameters optimized for RTX 4070 SUPER (12GB VRAM)
    training_args = {
        "data": str(dataset_yaml),
        "epochs": 50,  # Quick test run
        "imgsz": 640,
        "batch": 32,  # Large batch for RTX 4070 SUPER
        "device": 0,  # Use GPU 0
        "project": "hud_region_training/hud_region_training_8class/runs",
        "name": "hud_8class_v1",
        "save_period": 5,  # Save every 5 epochs
        "patience": 10,  # Early stopping patience
        "lr0": 0.01,  # Initial learning rate
        "lrf": 0.1,  # Final learning rate factor
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,  # Box loss gain
        "cls": 0.5,  # Class loss gain
        "dfl": 1.5,  # DFL loss gain
        "pose": 12.0,  # Pose loss gain
        "kobj": 1.0,  # Keypoint obj loss gain
        "label_smoothing": 0.0,
        "nbs": 64,  # Nominal batch size
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        "plots": True,
        "save": True,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show_labels": True,
        "show_conf": True,
        "visualize": False,
        "augment": True,
        "agnostic_nms": False,
        "retina_masks": False,
        "format": "torchscript",
        "keras": False,
        "optimize": False,
        "int8": False,
        "dynamic": False,
        "simplify": False,
        "opset": None,
        "workspace": 4,
        "nms": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "resume": False,
        "amp": True,  # Automatic Mixed Precision for speed
        "cache": True,  # Cache images for faster training
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        "multi_scale": False,
        "copy_paste": 0.0,
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "crop_fraction": 1.0,
    }

    print(f"📊 Training with {train_count} images, validating with {val_count} images")
    print(f"🎯 Target classes: 8 (5 existing + 3 new)")
    print(f"⚙️ Model: YOLOv8n (nano) optimized for RTX 4070 SUPER")
    print(f"🚀 GPU: RTX 4070 SUPER with 12GB VRAM, batch size: {training_args['batch']}")
    print(f"🔥 Epochs: {training_args['epochs']} with early stopping")

    # Start training
    try:
        results = model.train(**training_args)

        print("\n🎉 Training completed successfully!")
        print(
            f"📁 Results saved to: hud_region_training/hud_region_training_8class/runs/hud_8class_v1"
        )
        print(
            f"🏆 Best model: hud_region_training/hud_region_training_8class/runs/hud_8class_v1/weights/best.pt"
        )

        return results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


if __name__ == "__main__":
    results = train_8class_model()

    if results:
        print("\n📈 Training Summary:")
        print("- Check the results folder for training plots")
        print("- Best model weights saved for inference")
        print("- Validation metrics available in results")
        print("\n🔄 Next steps:")
        print("1. Test the model with inference")
        print("2. Compare with 5-class model performance")
        print("3. Continue annotation if needed")

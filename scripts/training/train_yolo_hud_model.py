#!/usr/bin/env python3
"""
Train YOLOv8 model on perfectly annotated HUD training data.
Uses the 300 duplicated perfect examples for training.
"""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def setup_training_environment():
    """Set up the training environment and directories."""
    print("🏈 Setting up YOLOv8 HUD Training Environment...")

    # Create training run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/train/hud_detection_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Training run directory: {run_dir}")
    return run_dir


def verify_dataset():
    """Verify the dataset is properly set up."""
    print("\n🔍 Verifying dataset...")

    # Check images and labels
    images_dir = Path("training_data/images")
    labels_dir = Path("training_data/labels")

    image_files = list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))

    print(f"📸 Total images: {len(image_files)}")
    print(f"🏷️  Total labels: {len(label_files)}")

    # Check for perfect matches
    perfect_copies = [f for f in image_files if "copy" in f.name]
    perfect_labels = [f for f in label_files if "copy" in f.name]

    print(f"✨ Perfect duplicated images: {len(perfect_copies)}")
    print(f"✨ Perfect duplicated labels: {len(perfect_labels)}")

    # Check dataset.yaml
    dataset_yaml = Path("training_data/dataset.yaml")
    if dataset_yaml.exists():
        print(f"✅ Dataset config found: {dataset_yaml}")
        with open(dataset_yaml) as f:
            config = yaml.safe_load(f)
            print(f"🎯 Classes: {config.get('nc', 'Unknown')} - {config.get('names', [])}")
    else:
        print("❌ Dataset config not found!")
        return False

    return len(image_files) > 0 and len(label_files) > 0


def create_train_val_split():
    """Create train/validation split from the perfect data."""
    print("\n📊 Creating train/validation split...")

    # Get all image files
    images_dir = Path("training_data/images")
    all_images = list(images_dir.glob("*.png"))

    # Separate perfect copies from original images
    perfect_copies = [f for f in all_images if "copy" in f.name]
    original_images = [f for f in all_images if "copy" not in f.name]

    print(f"✨ Perfect copies for training: {len(perfect_copies)}")
    print(f"📚 Original images for validation: {len(original_images)}")

    # Create train/val directories
    train_img_dir = Path("training_data/train/images")
    train_lbl_dir = Path("training_data/train/labels")
    val_img_dir = Path("training_data/val/images")
    val_lbl_dir = Path("training_data/val/labels")

    # Create directories
    for dir_path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy perfect copies to training set
    print("📁 Setting up training set with perfect copies...")
    for img_file in perfect_copies:
        # Copy image
        shutil.copy2(img_file, train_img_dir / img_file.name)

        # Copy corresponding label
        label_file = Path("training_data/labels") / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, train_lbl_dir / label_file.name)

    # Copy subset of original images to validation set
    val_images = original_images[:50]  # Use 50 original images for validation
    print(f"📁 Setting up validation set with {len(val_images)} original images...")
    for img_file in val_images:
        # Copy image
        shutil.copy2(img_file, val_img_dir / img_file.name)

        # Copy corresponding label
        label_file = Path("training_data/labels") / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, val_lbl_dir / label_file.name)

    print(f"✅ Training set: {len(perfect_copies)} perfect images")
    print(f"✅ Validation set: {len(val_images)} diverse images")

    return len(perfect_copies), len(val_images)


def create_training_config():
    """Create the training configuration file."""
    print("\n⚙️ Creating training configuration...")

    config = {
        "path": str(Path("training_data").absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": 13,  # number of classes
        "names": [
            "hud",
            "qb_position",
            "left_hash_mark",
            "right_hash_mark",
            "preplay",
            "playcall",
            "possession_indicator",
            "territory_indicator",
            "score_bug",
            "down_distance",
            "game_clock",
            "play_clock",
            "yards_to_goal",
        ],
    }

    config_path = Path("training_data/hud_dataset.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"📝 Training config saved: {config_path}")
    return config_path


def train_model(config_path: str, run_dir: str):
    """Train the YOLOv8 model with optimized parameters for HUD detection.

    Key improvements:
    1. Focused augmentation strategy for HUD elements
    2. Multi-scale training for different resolutions
    3. Optimized hyperparameters for static UI detection
    4. Enhanced early stopping and model selection
    5. Custom loss weights for better HUD region detection
    """
    print("\n🚀 Starting YOLOv8 Training with Enhanced Parameters...")

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    # Initialize model - use nano for faster training, will be scaled up later
    model = YOLO("yolov8n.pt")

    # Enhanced training parameters for HUD detection
    training_args = {
        "data": str(config_path),
        "epochs": 100,  # Increased epochs for better convergence
        "patience": 15,  # Increased patience for better model selection
        "save_period": 5,
        "cache": True,  # Cache images for faster training
        "device": device,
        "project": "runs/train",
        "name": f'hud_detection_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        # Multi-scale training
        "imgsz": [640, 960],  # Train on multiple sizes
        "rect": False,  # Allow for multi-scale training
        # Batch size optimization
        "batch": 32 if device == "cuda" else 8,
        "workers": 8 if device == "cuda" else 4,
        # Optimizer settings
        "optimizer": "AdamW",  # Better convergence for our task
        "lr0": 0.001,  # Initial learning rate
        "lrf": 0.01,  # Final learning rate fraction
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Focused augmentation for HUD detection
        "hsv_h": 0.015,  # Slight hue variation
        "hsv_s": 0.2,  # Moderate saturation variation
        "hsv_v": 0.1,  # Slight brightness variation
        "degrees": 0.0,  # No rotation (HUD is always horizontal)
        "translate": 0.05,  # Minimal translation
        "scale": 0.05,  # Minimal scaling
        "shear": 0.0,  # No shear
        "perspective": 0.0,  # No perspective
        "flipud": 0.0,  # No vertical flip
        "fliplr": 0.0,  # No horizontal flip
        "mosaic": 0.25,  # Reduced mosaic for stable training
        "mixup": 0.0,  # No mixup
        "copy_paste": 0.0,  # No copy-paste
        # Enhanced loss weights for better HUD detection
        "box": 7.5,  # Increased box loss weight
        "cls": 0.3,  # Reduced classification weight (fewer classes)
        "dfl": 1.5,  # DFL loss weight
        "cls_pw": 1.0,  # Cls BCELoss positive_weight
        "obj": 1.0,  # Obj loss gain
        "obj_pw": 1.0,  # Obj BCELoss positive_weight
        # Validation settings
        "val": True,
        "plots": True,
        "save": True,
        "save_json": True,
        "save_hybrid": True,  # Save hybrid version of labels
        "conf": 0.001,  # NMS confidence threshold
        "iou": 0.6,  # NMS IoU threshold
        "max_det": 300,  # Maximum detections per image
        "half": True,  # Use FP16 half-precision inference
        # Logging and monitoring
        "verbose": True,
        "exist_ok": False,  # Increment run name if exists
    }

    print("\n📋 Enhanced Training Parameters:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")

    print("\n🏈 Starting enhanced training pipeline...")
    print("=" * 50)

    try:
        # Start training
        results = model.train(**training_args)

        print("\n✅ Training completed successfully!")
        print(f"📊 Results saved to: {training_args['project']}/{training_args['name']}/")

        # Return best model path
        return str(Path(training_args["project"]) / training_args["name"] / "weights" / "best.pt")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return None


def evaluate_model(results):
    """Evaluate the trained model."""
    print("\n📊 Evaluating trained model...")

    # Load the best model
    best_model_path = results.save_dir / "weights" / "best.pt"
    model = YOLO(str(best_model_path))

    # Run validation
    val_results = model.val()

    print(f"📈 Validation Results:")
    print(f"   mAP50: {val_results.box.map50:.4f}")
    print(f"   mAP50-95: {val_results.box.map:.4f}")
    print(f"   Precision: {val_results.box.mp:.4f}")
    print(f"   Recall: {val_results.box.mr:.4f}")

    return val_results


def verify_training_data(data_path: str) -> bool:
    """Verify training data before starting."""
    try:
        # Load and verify dataset
        model = YOLO("yolov8n.pt")
        model.train(data=data_path, epochs=1, imgsz=640, verbose=False)
        return True
    except Exception as e:
        logger.error(f"Data verification failed: {e}")
        return False


def main():
    """Main training function."""
    print("🏈 YOLOv8 HUD Detection Training Script")
    print("=" * 50)

    try:
        # Setup
        run_dir = setup_training_environment()

        # Verify dataset
        if not verify_dataset():
            print("❌ Dataset verification failed!")
            return

        # Create train/val split
        train_count, val_count = create_train_val_split()

        # Create training config
        config_path = create_training_config()

        # Verify training data
        if not verify_training_data(config_path):
            print("❌ Training data verification failed!")
            return

        # Train model
        best_model_path = train_model(config_path, run_dir)

        if best_model_path:
            # Evaluate model
            model = YOLO(best_model_path)
            val_results = model.val()

            print("\n🎉 Training Complete!")
            print(f"📁 Best model: {best_model_path}")
            print(f"📁 Last model: {model.path}")
            print("\n🚀 Ready to test on your gameplay footage!")

            # Evaluate model
            val_results = evaluate_model(val_results)

            print("\n📊 Evaluated model results:")
            print(f"   mAP50: {val_results.box.map50:.4f}")
            print(f"   mAP50-95: {val_results.box.map:.4f}")
            print(f"   Precision: {val_results.box.mp:.4f}")
            print(f"   Recall: {val_results.box.mr:.4f}")
        else:
            print("\n❌ Training failed. No model path found.")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optimized YOLOv8 training script for HUD detection with RTX 4070 Super."""

import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def create_dataset_config():
    """Create YOLO dataset configuration file."""
    config = {
        "names": ["hud", "qb_position", "left_hash_mark", "right_hash_mark"],
        "nc": 4,
        "path": str(Path("training_data").absolute()),
        "train": "images",
        "val": "images",
        "test": "images",
    }

    config_file = Path("training_data/dataset.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_file


def train_model(epochs=25, img_size=512, batch_size=32):  # Optimized parameters
    """Train YOLOv8 model with optimized settings for RTX 4070 Super."""
    print("🏈 SpygateAI HUD Detection Training")
    print("=" * 50)

    # Check for training data
    images_dir = Path("training_data/images")
    labels_dir = Path("training_data/labels")

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Training data not found!")
        print(f"Expected: {images_dir} and {labels_dir}")
        return None

    print(f"✅ Found {len(list(images_dir.glob('*.png')))} images")
    print(f"✅ Found {len(list(labels_dir.glob('*.txt')))} labels")

    # Create dataset config
    config_file = create_dataset_config()
    print(f"✅ Dataset config: {config_file}")

    # Load YOLOv8s (smaller, faster model for HUD detection)
    print("🚀 Loading YOLOv8s model (optimized for speed)...")
    model = YOLO("yolov8s.pt")  # Changed from 'yolov8m.pt' to 'yolov8s.pt'

    print(f"🎯 Training settings:")
    print(f"   - Model: YOLOv8s (faster)")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {img_size}px")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Device: GPU (RTX 4070 Super)")
    print(f"   - Mixed precision: Enabled")

    # Train the model with optimized settings
    results = model.train(
        data=str(config_file),
        epochs=epochs,  # Reduced from 50 to 25
        imgsz=img_size,  # Reduced from 640 to 512
        batch=batch_size,  # Increased from 16 to 32
        name="spygate_hud_detection_fast",
        project="runs/detect",
        patience=8,  # Reduced early stopping patience
        save=True,
        plots=True,
        val=True,
        device="0",  # Use GPU 0 (RTX 4070 Super)
        half=True,  # Enable mixed precision
        amp=True,  # Enable Automatic Mixed Precision
        workers=8,  # Optimize data loading
        cache=True,  # Cache images in RAM for faster access
        close_mosaic=5,  # Disable mosaic augmentation in last 5 epochs for stability
        save_period=5,  # Save checkpoint every 5 epochs
        optimizer="AdamW",  # Use AdamW optimizer (often faster convergence)
        lr0=0.01,  # Higher learning rate for faster training
        weight_decay=0.0005,  # Regularization
        warmup_epochs=2,  # Reduced warmup
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain
        dfl=1.5,  # Distribution Focal Loss gain
        pose=12.0,  # Pose loss gain (if applicable)
        kobj=1.0,  # Keypoint objectness loss gain
        label_smoothing=0.0,  # Label smoothing
        nbs=64,  # Nominal batch size
        hsv_h=0.015,  # Image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # Image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # Image HSV-Value augmentation (fraction)
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,  # Image scale (+/- gain)
        shear=0.0,  # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # Image flip up-down (probability)
        fliplr=0.5,  # Image flip left-right (probability)
        mosaic=1.0,  # Image mosaic (probability)
        mixup=0.0,  # Image mixup (probability)
        copy_paste=0.0,  # Segment copy-paste (probability)
    )

    print("\n🎉 Training completed!")
    print(f"📊 Results saved in: runs/detect/spygate_hud_detection_fast/")

    return results


if __name__ == "__main__":
    try:
        # Run optimized training for RTX 4070 Super
        results = train_model(epochs=25, img_size=512, batch_size=32)  # Much faster settings
        if results:
            print("✅ Training successful!")
            print(f"📈 Best model saved at: {results.save_dir}")
        else:
            print("❌ Training failed - check your data setup")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This is normal for a first test - let's debug the issue.")

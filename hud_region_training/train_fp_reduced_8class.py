"""
Enhanced 8-Class YOLOv8 Training with False Positive Reduction.
Optimized for RTX 4070 SUPER (12GB VRAM) - Maximum Speed Configuration.
"""

import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def train_fp_reduced_model():
    """Train 8-class model with false positive reduction strategies - SPEED OPTIMIZED."""

    print("🚀 Training 8-Class Model - RTX 4070 SUPER SPEED OPTIMIZED")
    print("=" * 70)

    # Load base model
    model = YOLO("yolov8n.pt")

    # SPEED OPTIMIZED training configuration for RTX 4070 SUPER
    training_args = {
        # Dataset
        "data": "hud_region_training_8class/dataset_8class_training.yaml",
        # Training parameters - OPTIMIZED FOR SPEED
        "epochs": 50,  # Reduced for faster training while maintaining quality
        "batch": 32,  # INCREASED: Maximum batch size for 12GB VRAM
        "imgsz": 640,  # Optimal size for speed/accuracy balance
        "device": 0,  # Force GPU usage
        "workers": 12,  # INCREASED: More workers for faster data loading
        # Loss function weights (ENHANCED FOR PRECISION)
        "box": 7.5,  # Box regression loss
        "cls": 1.5,  # INCREASED: Higher classification loss (reduces FP)
        "dfl": 1.5,  # Distribution focal loss
        # Learning rate (OPTIMIZED FOR SPEED)
        "lr0": 0.01,  # INCREASED: Higher learning rate for faster convergence
        "lrf": 0.01,  # Final learning rate
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Regularization (ADDED FOR FP REDUCTION)
        "dropout": 0.1,  # Dropout to reduce overfitting
        "label_smoothing": 0.1,  # Label smoothing for better calibration
        # Augmentation (CONSERVATIVE FOR HUD - NO DESTRUCTIVE AUGMENTATIONS)
        "hsv_h": 0.015,  # Minimal hue changes
        "hsv_s": 0.7,  # Saturation changes OK
        "hsv_v": 0.4,  # Value changes OK
        "degrees": 0.0,  # NO rotation (HUD is always horizontal)
        "translate": 0.1,  # Minimal translation
        "scale": 0.5,  # Scale changes OK
        "shear": 0.0,  # NO shear (HUD elements are rectangular)
        "perspective": 0.0,  # NO perspective (HUD is 2D overlay)
        "flipud": 0.0,  # NO vertical flip (HUD has fixed orientation)
        "fliplr": 0.0,  # NO horizontal flip (possession triangles have meaning)
        "mosaic": 0.0,  # NO mosaic (can create false HUD combinations)
        "mixup": 0.0,  # NO mixup (HUD elements shouldn't blend)
        "copy_paste": 0.0,  # NO copy-paste (HUD elements are contextual)
        # Training strategy (OPTIMIZED FOR SPEED + PRECISION)
        "optimizer": "AdamW",  # AdamW often better for precision than SGD
        "cos_lr": True,  # Cosine learning rate schedule
        "warmup_epochs": 3,  # REDUCED: Less warmup for faster start
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "close_mosaic": 10,  # REDUCED: Close mosaic augmentation earlier
        # Validation and saving
        "patience": 15,  # REDUCED: Less patience for faster training
        "save_period": 5,  # REDUCED: Save every 5 epochs for monitoring
        "val": True,
        "plots": True,
        "save": True,
        # Output
        "project": "hud_region_training_8class/runs",
        "name": "hud_8class_fp_reduced_speed",
        "exist_ok": True,
        # Performance and reproducibility - SPEED OPTIMIZED
        "cache": True,  # Cache images for faster training
        "amp": True,  # Automatic Mixed Precision for speed
        "half": False,  # Keep full precision for better accuracy
        "seed": 42,  # Reproducible results
        "deterministic": False,  # CHANGED: Allow non-deterministic for speed
        "pretrained": True,  # Use pretrained weights
        "verbose": True,
        # Additional speed optimizations
        "rect": False,  # Rectangular training can be slower
        "multi_scale": False,  # Disable multi-scale for consistent speed
        "profile": False,  # Disable profiling for speed
    }

    print("🚀 RTX 4070 SUPER SPEED OPTIMIZATIONS ACTIVE:")
    print(f"⚡ Batch size: {training_args['batch']} (MAXIMIZED for 12GB VRAM)")
    print(f"🔥 Workers: {training_args['workers']} (INCREASED for faster data loading)")
    print(f"📈 Learning rate: {training_args['lr0']} (INCREASED for faster convergence)")
    print(f"⏱️ Epochs: {training_args['epochs']} (OPTIMIZED for speed/quality balance)")
    print(f"🎯 Classification loss weight: {training_args['cls']} (increased)")
    print(f"🛡️ Dropout: {training_args['dropout']} (added for regularization)")
    print(f"🎪 Label smoothing: {training_args['label_smoothing']} (added for calibration)")
    print(f"⚡ Optimizer: {training_args['optimizer']} (better for precision)")
    print(f"🔄 Cosine LR schedule: {training_args['cos_lr']}")
    print(f"⚡ AMP Enabled: {training_args['amp']} (SPEED BOOST)")
    print(f"🚫 Disabled destructive augmentations (mosaic, mixup, flips)")

    try:
        print("\n🔥 STARTING MAXIMUM SPEED TRAINING...")
        results = model.train(**training_args)

        print("\n🎉 Enhanced training completed!")
        print("📈 Expected improvements:")
        print("  ✅ Higher precision (fewer false positives)")
        print("  ✅ Better class separation")
        print("  ✅ More robust detections")
        print("  ✅ Improved confidence calibration")
        print("  ⚡ MAXIMUM TRAINING SPEED on RTX 4070 SUPER")

        print(f"\n📁 Results saved to: hud_region_training_8class/runs/hud_8class_fp_reduced_speed")
        print(
            f"🏆 Best model: hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
        )

        return results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


if __name__ == "__main__":
    results = train_fp_reduced_model()

    if results:
        print("\n🔄 Next steps:")
        print("1. Test the FP-reduced model with inference")
        print("2. Compare precision/recall with original model")
        print("3. Apply post-processing filters for additional FP reduction")
        print("4. Validate on real gameplay footage")
        print("5. 🚀 ENJOY THE SPEED BOOST!")

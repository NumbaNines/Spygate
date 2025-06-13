"""
Enhanced 8-Class YOLOv8 Training with False Positive Reduction.
Implements multiple strategies to improve precision and reduce unwanted detections.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_fp_reduced_model():
    """Train 8-class model with false positive reduction strategies."""
    
    print("🎯 Training 8-Class Model with False Positive Reduction")
    print("=" * 60)
    
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Enhanced training configuration for FALSE POSITIVE REDUCTION
    training_args = {
        # Dataset
        'data': 'hud_region_training/hud_region_training_8class/dataset_8class_training.yaml',
        
        # Training parameters
        'epochs': 100,           # More epochs for better convergence
        'batch': 16,             # Smaller batch for better gradients
        'imgsz': 640,
        'device': 0,
        'workers': 8,
        
        # Loss function weights (ENHANCED FOR PRECISION)
        'box': 7.5,              # Box regression loss
        'cls': 1.5,              # INCREASED: Higher classification loss (reduces FP)
        'dfl': 1.5,              # Distribution focal loss
        
        # Learning rate (CONSERVATIVE FOR PRECISION)
        'lr0': 0.005,            # REDUCED: Lower initial learning rate
        'lrf': 0.01,             # Lower final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Regularization (ADDED FOR FP REDUCTION)
        'dropout': 0.1,          # Dropout to reduce overfitting
        'label_smoothing': 0.1,  # Label smoothing for better calibration
        
        # Augmentation (CONSERVATIVE FOR HUD - NO DESTRUCTIVE AUGMENTATIONS)
        'hsv_h': 0.015,          # Minimal hue changes
        'hsv_s': 0.7,            # Saturation changes OK
        'hsv_v': 0.4,            # Value changes OK
        'degrees': 0.0,          # NO rotation (HUD is always horizontal)
        'translate': 0.1,        # Minimal translation
        'scale': 0.5,            # Scale changes OK
        'shear': 0.0,            # NO shear (HUD elements are rectangular)
        'perspective': 0.0,      # NO perspective (HUD is 2D overlay)
        'flipud': 0.0,           # NO vertical flip (HUD has fixed orientation)
        'fliplr': 0.0,           # NO horizontal flip (possession triangles have meaning)
        'mosaic': 0.0,           # NO mosaic (can create false HUD combinations)
        'mixup': 0.0,            # NO mixup (HUD elements shouldn't blend)
        'copy_paste': 0.0,       # NO copy-paste (HUD elements are contextual)
        
        # Training strategy (OPTIMIZED FOR PRECISION)
        'optimizer': 'AdamW',    # AdamW often better for precision than SGD
        'cos_lr': True,          # Cosine learning rate schedule
        'warmup_epochs': 5,      # More warmup for stable training
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'close_mosaic': 15,      # Close mosaic augmentation early
        
        # Validation and saving
        'patience': 20,          # More patience for early stopping
        'save_period': 10,       # Save every 10 epochs
        'val': True,
        'plots': True,
        'save': True,
        
        # Output
        'project': 'hud_region_training/hud_region_training_8class/runs',
        'name': 'hud_8class_fp_reduced',
        'exist_ok': True,
        
        # Performance and reproducibility
        'cache': True,           # Cache images for faster training
        'amp': True,             # Automatic Mixed Precision
        'seed': 42,              # Reproducible results
        'deterministic': True,   # Deterministic training
        'pretrained': True,      # Use pretrained weights
        'verbose': True
    }
    
    print("🚀 Starting enhanced training with FP reduction strategies...")
    print(f"📊 Batch size: {training_args['batch']} (smaller for better gradients)")
    print(f"🎯 Classification loss weight: {training_args['cls']} (increased)")
    print(f"🛡️ Dropout: {training_args['dropout']} (added for regularization)")
    print(f"🎪 Label smoothing: {training_args['label_smoothing']} (added for calibration)")
    print(f"⚡ Optimizer: {training_args['optimizer']} (better for precision)")
    print(f"📚 Disabled destructive augmentations (mosaic, mixup, flips)")
    print(f"🔄 Cosine LR schedule: {training_args['cos_lr']}")
    
    try:
        results = model.train(**training_args)
        
        print("\n🎉 Enhanced training completed!")
        print("📈 Expected improvements:")
        print("  ✅ Higher precision (fewer false positives)")
        print("  ✅ Better class separation")
        print("  ✅ More robust detections")
        print("  ✅ Improved confidence calibration")
        
        print(f"\n📁 Results saved to: hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced")
        print(f"🏆 Best model: hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced/weights/best.pt")
        
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
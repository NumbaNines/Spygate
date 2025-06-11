#!/usr/bin/env python3
"""
Train YOLOv8 model with CORRECTED classes (only the 4 that actually exist in annotations).
"""

import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def create_corrected_dataset_config():
    """Create dataset config with only the classes that actually exist."""
    print("🔧 Creating corrected dataset configuration...")
    
    # The 4 classes that actually exist in the annotations (classes 0-3)
    actual_classes = [
        "hud",                # Class 0 - Main HUD bar
        "qb_position",        # Class 1 - QB/ball position 
        "left_hash_mark",     # Class 2 - Left hash mark
        "right_hash_mark"     # Class 3 - Right hash mark
    ]
    
    print(f"✅ Using {len(actual_classes)} actual classes: {actual_classes}")
    
    # Create corrected dataset configuration
    dataset_config = {
        'path': str(Path.cwd() / 'training_data'),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(actual_classes),  # Number of classes = 4
        'names': actual_classes
    }
    
    # Save corrected dataset.yaml
    config_path = Path('training_data/dataset_corrected.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Saved corrected config: {config_path}")
    return str(config_path)

def verify_training_data():
    """Verify we have the training data properly set up."""
    print("🔍 Verifying training data...")
    
    train_img_dir = Path("training_data/train/images")
    train_lbl_dir = Path("training_data/train/labels")
    val_img_dir = Path("training_data/val/images")
    val_lbl_dir = Path("training_data/val/labels")
    
    train_imgs = len(list(train_img_dir.glob("*.png")))
    train_lbls = len(list(train_lbl_dir.glob("*.txt")))
    val_imgs = len(list(val_img_dir.glob("*.png")))
    val_lbls = len(list(val_lbl_dir.glob("*.txt")))
    
    print(f"📊 Training: {train_imgs} images, {train_lbls} labels")
    print(f"📊 Validation: {val_imgs} images, {val_lbls} labels")
    
    if train_imgs == 0 or train_lbls == 0:
        print("❌ No training data found!")
        return False
        
    if train_imgs != train_lbls:
        print(f"⚠️  Mismatch: {train_imgs} images vs {train_lbls} labels")
        return False
    
    print("✅ Training data verified!")
    return True

def train_corrected_model():
    """Train YOLOv8 with corrected classes."""
    print("🏈 Training YOLOv8 HUD Model (Corrected Classes)")
    print("=" * 50)
    
    # Verify data first
    if not verify_training_data():
        print("❌ Training data verification failed!")
        return
    
    # Create corrected config
    dataset_config_path = create_corrected_dataset_config()
    
    # Initialize model
    print("📦 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Training parameters optimized for HUD detection
    training_args = {
        'data': dataset_config_path,
        'epochs': 50,
        'batch': 4,
        'imgsz': 640,
        'device': 'cpu',
        'patience': 10,
        'save': True,
        'cache': True,
        'workers': 4,
        
        # Optimization for HUD detection
        'augment': False,       # Disable augmentation since HUD is static
        'mixup': 0.0,          # No mixup for HUD
        'mosaic': 0.0,         # No mosaic for HUD
        'copy_paste': 0.0,     # No copy-paste
        'degrees': 0.0,        # No rotation
        'translate': 0.0,      # No translation  
        'scale': 0.0,          # No scaling
        'shear': 0.0,          # No shear
        'perspective': 0.0,    # No perspective
        'flipud': 0.0,         # No vertical flip
        'fliplr': 0.0,         # No horizontal flip
        
        # Loss weights
        'cls': 1.0,            # Classification loss weight
        'box': 7.5,            # Box regression loss weight
        'dfl': 1.5,            # Distribution focal loss weight
        
        # Learning parameters
        'lr0': 0.01,           # Initial learning rate
        'momentum': 0.937,     # SGD momentum
        'weight_decay': 0.0005,# Weight decay
        'warmup_epochs': 3,    # Warmup epochs
        'warmup_momentum': 0.8,# Warmup momentum
        
        # Confidence and IoU
        'conf': 0.25,          # Confidence threshold
        'iou': 0.7,            # IoU threshold for NMS
        
        'project': 'runs/train',
        'name': f'hud_corrected_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': False,
        'verbose': True
    }
    
    print("🚀 Starting training with corrected 4-class configuration...")
    print(f"   Classes: hud, qb_position, left_hash_mark, right_hash_mark")
    print(f"   Training images: 300 (perfect duplicates)")
    print(f"   Validation images: 50 (diverse)")
    print(f"   Epochs: {training_args['epochs']}")
    print(f"   Batch size: {training_args['batch']}")
    print(f"   No augmentation (HUD is static)")
    
    try:
        # Start training
        results = model.train(**training_args)
        
        print("🎉 Training completed successfully!")
        print(f"✅ Results saved to: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'maps'):
            print(f"📊 Final mAP50: {results.maps[0]:.4f}")
            print(f"📊 Final mAP50-95: {results.maps[1]:.4f}")
        
        return results.save_dir
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    train_corrected_model() 
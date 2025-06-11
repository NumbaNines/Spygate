#!/usr/bin/env python3
"""
Final YOLOv8 training with ALL 8 classes including triangles!
"""

import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def create_final_dataset_config():
    """Create the final dataset configuration with all 8 classes."""
    print("🔧 Creating final dataset configuration with triangles...")
    
    # All 8 classes (matching classes.txt order)
    all_classes = [
        "hud",                    # Class 0
        "qb_position",            # Class 1
        "left_hash_mark",         # Class 2
        "right_hash_mark",        # Class 3
        "preplay",                # Class 4
        "playcall",               # Class 5
        "possession_indicator",   # Class 6 - LEFT TRIANGLE!
        "territory_indicator"     # Class 7 - RIGHT TRIANGLE!
    ]
    
    print(f"✅ Using ALL {len(all_classes)} classes: {all_classes}")
    
    # Create final dataset configuration
    dataset_config = {
        'path': str(Path('training_data').absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(all_classes),
        'names': all_classes
    }
    
    config_path = Path("training_data/dataset_final.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Created final dataset config: {config_path}")
    return config_path

def train_final_model():
    """Train the final YOLOv8 model with all classes including triangles."""
    print("🏈 Training Final YOLOv8 Model with Triangles!")
    print("=" * 50)
    
    # Create dataset config
    config_path = create_final_dataset_config()
    
    # Initialize YOLOv8 model
    print("📦 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')  # Start with nano for faster training
    
    # Training parameters optimized for triangle detection
    training_params = {
        'data': str(config_path),
        'epochs': 100,           # More epochs for triangle learning
        'batch': 16,             # Larger batch for RTX 4070 Super
        'device': 'cuda',        # Use GPU for faster training
        'imgsz': 640,           # Standard image size
        'patience': 20,          # Early stopping patience
        'save': True,
        'save_period': 10,       # Save every 10 epochs
        'cache': False,          # Don't cache images
        'project': 'runs/detect',
        'name': f'spygate_triangles_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',    # Good for small objects like triangles
        'verbose': True,
        'seed': 42,              # Reproducible results
        'deterministic': True,
        'single_cls': False,     # Multi-class detection
        'rect': False,           # Don't use rectangular training
        'cos_lr': True,          # Cosine learning rate
        'close_mosaic': 10,      # Disable mosaic last 10 epochs
        'resume': False,
        'amp': True,             # Mixed precision for RTX 4070 Super
        'fraction': 1.0,         # Use all data
        'profile': False,
        'freeze': None,          # Don't freeze layers
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.01,            # Final learning rate factor
        'momentum': 0.937,       # SGD momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,    # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,   # Warmup bias learning rate
        'box': 7.5,             # Box loss gain
        'cls': 0.5,             # Class loss gain
        'dfl': 1.5,             # DFL loss gain
        'pose': 12.0,           # Pose loss gain
        'kobj': 1.0,            # Keypoint object loss gain
        'label_smoothing': 0.0,  # Label smoothing
        'nbs': 64,              # Nominal batch size
        'hsv_h': 0.015,         # Hue augmentation
        'hsv_s': 0.7,           # Saturation augmentation  
        'hsv_v': 0.4,           # Value augmentation
        'degrees': 0.0,         # Rotation augmentation (degrees)
        'translate': 0.1,       # Translation augmentation
        'scale': 0.5,           # Scale augmentation
        'shear': 0.0,           # Shear augmentation (degrees)
        'perspective': 0.0,     # Perspective augmentation
        'flipud': 0.0,          # Vertical flip augmentation
        'fliplr': 0.5,          # Horizontal flip augmentation
        'mosaic': 1.0,          # Mosaic augmentation
        'mixup': 0.0,           # Mixup augmentation
        'copy_paste': 0.0,      # Copy-paste augmentation
    }
    
    print(f"🎯 Training with {training_params['epochs']} epochs...")
    print(f"🎯 Using device: {training_params['device']}")
    print(f"🎯 Batch size: {training_params['batch']}")
    print(f"🎯 Learning rate: {training_params['lr0']}")
    
    try:
        # Start training
        print("🚀 Starting training...")
        results = model.train(**training_params)
        
        print("✅ Training completed successfully!")
        print(f"📊 Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_final_model()
    if success:
        print("🎉 YOLOv8 training with triangles completed!")
    else:
        print("�� Training failed!") 
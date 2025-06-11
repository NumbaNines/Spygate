#!/usr/bin/env python3
"""
Train YOLOv8 model on perfectly annotated HUD training data.
Uses the 300 duplicated perfect examples for training.
"""

import os
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch

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
        'path': str(Path('training_data').absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 13,  # number of classes
        'names': [
            'hud',
            'qb_position', 
            'left_hash_mark',
            'right_hash_mark',
            'preplay',
            'playcall',
            'possession_indicator',
            'territory_indicator',
            'score_bug',
            'down_distance',
            'game_clock',
            'play_clock',
            'yards_to_goal'
        ]
    }
    
    config_path = Path("training_data/hud_dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Training config saved: {config_path}")
    return config_path

def train_model(config_path, run_dir):
    """Train the YOLOv8 model."""
    print("\n🚀 Starting YOLOv8 Training...")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with YOLOv8 nano pretrained weights
    
    # Training parameters optimized for HUD detection
    training_args = {
        'data': str(config_path),
        'epochs': 50,  # Start with 50 epochs
        'imgsz': 640,
        'batch': 16 if device == 'cuda' else 4,
        'device': device,
        'workers': 4,
        'patience': 10,  # Early stopping patience
        'save_period': 5,  # Save every 5 epochs
        'val': True,
        'plots': True,
        'verbose': True,
        'project': 'runs/train',
        'name': f'hud_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Optimization parameters
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        
        # Augmentation (minimal for perfect data)
        'degrees': 0.0,  # No rotation
        'translate': 0.1,  # Slight translation
        'scale': 0.1,  # Slight scaling
        'shear': 0.0,  # No shearing
        'perspective': 0.0,  # No perspective
        'flipud': 0.0,  # No vertical flip
        'fliplr': 0.0,  # No horizontal flip
        'mosaic': 0.5,  # Reduced mosaic
        'mixup': 0.0,  # No mixup
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    print("📋 Training parameters:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print("\n🏈 Training starting... This may take a while!")
    print("📊 Monitor progress at: runs/train/<run_name>/")
    
    # Start training
    results = model.train(**training_args)
    
    print(f"\n✅ Training completed!")
    print(f"📈 Results saved to: {results.save_dir}")
    
    return results

def evaluate_model(results):
    """Evaluate the trained model."""
    print("\n📊 Evaluating trained model...")
    
    # Load the best model
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    # Run validation
    val_results = model.val()
    
    print(f"📈 Validation Results:")
    print(f"   mAP50: {val_results.box.map50:.4f}")
    print(f"   mAP50-95: {val_results.box.map:.4f}")
    print(f"   Precision: {val_results.box.mp:.4f}")
    print(f"   Recall: {val_results.box.mr:.4f}")
    
    return val_results

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
        
        # Train model
        results = train_model(config_path, run_dir)
        
        # Evaluate model
        val_results = evaluate_model(results)
        
        print("\n🎉 Training Complete!")
        print(f"📁 Best model: {results.save_dir}/weights/best.pt")
        print(f"📁 Last model: {results.save_dir}/weights/last.pt")
        print("\n🚀 Ready to test on your gameplay footage!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
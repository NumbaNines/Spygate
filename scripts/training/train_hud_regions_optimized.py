#!/usr/bin/env python3
"""
OPTIMIZED SpygateAI HUD Region Detection Training for RTX 4070 SUPER
Maximum performance configuration for 12GB VRAM - FRESH MODEL TRAINING
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import time

def main():
    """Train the HUD region detection model with MAXIMUM OPTIMIZATION - FRESH START."""
    print("🚀 Starting FRESH SpygateAI HUD Region Detection Training...")
    print("🧹 This will be a completely new model with no previous weights")
    
    # Check if dataset exists
    dataset_path = Path("hud_region_training/dataset/dataset.yaml")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run convert_labelme_to_yolo.py first")
        return
    
    # Force CUDA and optimize
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("⚠️  Training will run on CPU (much slower)")
        device = 'cpu'
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = '0'  # Use first GPU
        
        # Set memory management
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        print(f"🔥 CUDA optimized! Using GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    # FRESH MODEL - Start from scratch
    print("✨ Creating FRESH YOLOv8 model (no previous weights)")
    model = YOLO('yolov8n.pt')  # Start fresh from pretrained base
    
    # MAXIMUM PERFORMANCE SETTINGS for RTX 4070 SUPER
    training_args = {
        'data': str(dataset_path),
        'epochs': 100,               # More epochs for better training
        'imgsz': 640,               # Optimal image size for speed/accuracy
        'batch': 32,                # Maximum batch size for 12GB VRAM
        'workers': 8,               # Maximum workers for data loading
        'device': device,
        'project': 'hud_region_training/runs',
        'name': f'hud_regions_fresh_{int(time.time())}',  # Unique name with timestamp
        'exist_ok': False,          # Don't overwrite - create new
        'save': True,
        'save_period': 10,          # Save every 10 epochs
        'val': True,
        'plots': True,
        'verbose': True,
        
        # SPEED OPTIMIZATIONS
        'amp': True,                # Automatic Mixed Precision for speed
        'half': False,              # Keep full precision for accuracy
        'dnn': True,                # Use OpenCV DNN for speed
        'multi_scale': False,       # Disable for consistent speed
        'single_cls': False,        # We have multiple classes
        'optimizer': 'AdamW',       # Best optimizer for our task
        'lr0': 0.01,               # Initial learning rate
        'lrf': 0.01,               # Final learning rate
        'momentum': 0.937,          # SGD momentum
        'weight_decay': 0.0005,     # Weight decay
        'warmup_epochs': 3,         # Warmup epochs
        'warmup_momentum': 0.8,     # Warmup momentum
        'warmup_bias_lr': 0.1,      # Warmup bias learning rate
        'copy_paste': 0.0,          # Copy paste augmentation
        'mosaic': 1.0,              # Mosaic augmentation
        'mixup': 0.0,               # Mixup augmentation
        'degrees': 0.0,             # Rotation degrees
        'translate': 0.1,           # Translation
        'scale': 0.5,               # Scale
        'shear': 0.0,               # Shear
        'perspective': 0.0,         # Perspective
        'flipud': 0.0,              # Flip up-down
        'fliplr': 0.5,              # Flip left-right
        'hsv_h': 0.015,             # HSV hue
        'hsv_s': 0.7,               # HSV saturation
        'hsv_v': 0.4,               # HSV value
        'close_mosaic': 10,         # Close mosaic epochs
    }
    
    print("🎯 TRAINING CONFIGURATION:")
    print(f"   • Model: FRESH YOLOv8n (no previous weights)")
    print(f"   • Device: {device}")
    print(f"   • Batch Size: {training_args['batch']}")
    print(f"   • Workers: {training_args['workers']}")
    print(f"   • Image Size: {training_args['imgsz']}")
    print(f"   • Epochs: {training_args['epochs']}")
    print(f"   • AMP: {training_args['amp']}")
    print(f"   • Project: {training_args['project']}")
    print(f"   • Name: {training_args['name']}")
    print()
    
    # Start training
    print("🏁 Starting FRESH training...")
    results = model.train(**training_args)
    
    print(f"✅ Training complete!")
    print(f"📊 Results: {results}")
    
    # Show final model location
    model_path = Path(training_args['project']) / training_args['name'] / 'weights' / 'best.pt'
    print(f"🎯 FRESH model saved to: {model_path}")
    
    return results

if __name__ == "__main__":
    main() 
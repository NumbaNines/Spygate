#!/usr/bin/env python3
"""
MASSIVE Triangle Detection Model Training Script - GPU Optimized
Trains YOLOv8 on 2,000+ augmented triangle detection images for maximum performance
"""

import torch
import gc
from ultralytics import YOLO
from pathlib import Path
import time
import psutil
import subprocess
import sys
import os

def check_gpu():
    """Check GPU availability and memory"""
    print("=" * 60)
    print("🚀 GPU STATUS CHECK")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"💾 Total VRAM: {gpu_memory:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"🆓 Available VRAM: {free_memory / 1024**3:.1f} GB")
        
        return True
    else:
        print("❌ No GPU available! Training will be SLOW!")
        return False

def optimize_system():
    """Optimize system for maximum training performance"""
    print("\n" + "=" * 60)
    print("⚡ SYSTEM OPTIMIZATION")
    print("=" * 60)
    
    # Set environment variables for maximum performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ GPU memory cleared")
    print("✅ Environment optimized for training")

def train_massive_model():
    """Train the massive triangle detection model"""
    
    # Check GPU first
    has_gpu = check_gpu()
    if not has_gpu:
        print("⚠️  Training without GPU will be extremely slow!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Optimize system
    optimize_system()
    
    print("\n" + "=" * 60)
    print("🏋️  STARTING MASSIVE TRIANGLE TRAINING")
    print("=" * 60)
    
    # Dataset paths
    dataset_path = Path("yolo_massive_triangle_dataset/dataset.yaml")
    output_dir = Path("triangle_training_massive")
    
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"📁 Output: {output_dir}")
    
    # Load YOLOv8 model - Use the largest model for best accuracy
    print("\n🤖 Loading YOLOv8x model (largest, most accurate)...")
    model = YOLO('yolov8x.pt')  # Largest model for maximum accuracy
    
    # Training parameters optimized for RTX 4070 SUPER
    training_params = {
        # Dataset
        'data': str(dataset_path),
        
        # Training duration - More epochs for 2000 images
        'epochs': 150,  # Increased for larger dataset
        
        # Image size - Optimized for GPU memory
        'imgsz': 640,
        
        # Batch size - Optimized for RTX 4070 SUPER (12GB VRAM)
        'batch': 32,  # Aggressive batch size for faster training
        
        # Performance
        'device': '0',  # Force GPU 0
        'workers': 8,   # Multi-threading
        'amp': True,    # Mixed precision for speed
        
        # Model optimization
        'optimizer': 'AdamW',  # Best optimizer for YOLO
        'lr0': 0.01,          # Learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Data augmentation (already have augmented data)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,    # Reduced since we have rotation augmentation
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        
        # Output
        'project': str(output_dir.parent),
        'name': output_dir.name,
        'exist_ok': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        
        # Validation
        'val': True,
        'split': 0.2,  # 20% for validation
        
        # Early stopping
        'patience': 30,  # Stop if no improvement for 30 epochs
        
        # Visualization
        'plots': True,
        'verbose': True,
    }
    
    print("\n📊 TRAINING CONFIGURATION:")
    print("=" * 40)
    for key, value in training_params.items():
        if key not in ['data', 'project']:  # Skip long paths
            print(f"  {key}: {value}")
    
    print(f"\n🎯 Training on ~2,000 augmented images!")
    print(f"🔥 Expected training time: 45-90 minutes on RTX 4070 SUPER")
    
    # Ask for confirmation
    response = input("\n🚀 Start massive training? (Y/n): ")
    if response.lower() == 'n':
        print("❌ Training cancelled")
        return
    
    start_time = time.time()
    
    try:
        print("\n" + "🔥" * 60)
        print("TRAINING STARTED - LET'S MAKE THE BEST TRIANGLE DETECTOR EVER!")
        print("🔥" * 60)
        
        # Start training
        results = model.train(**training_params)
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        print("\n" + "🎉" * 60)
        print("MASSIVE TRAINING COMPLETED!")
        print("🎉" * 60)
        
        print(f"⏱️  Training time: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
        
        # Find best model
        best_model_path = output_dir / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"🏆 Best model saved: {best_model_path}")
            print(f"📁 Full results in: {output_dir}")
            
            # Quick performance summary
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    map50 = metrics['metrics/mAP50(B)']
                    print(f"🎯 Final mAP50: {map50:.2%}")
        
        print("\n🔍 Next steps:")
        print("1. Test the model with test_triangle_model.py")
        print("2. Update GUI to use the new massive model")
        print("3. Run live detection tests")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("💡 Try reducing batch size if out of memory")

if __name__ == "__main__":
    print("🔺 MASSIVE Triangle Detection Training 🔺")
    print("Training on 2,000+ augmented images for ultimate accuracy!")
    
    train_massive_model() 
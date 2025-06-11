#!/usr/bin/env python3
"""
Train triangle detection model with HUD using OPTIMIZED settings for RTX 4070 SUPER.

This script trains a YOLOv8 model specifically for detecting:
- HUD (class 0) 
- possession_indicator (class 1)
- territory_indicator (class 2)

Optimized for RTX 4070 SUPER with 12GB VRAM.
"""

import os
import logging
from pathlib import Path
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_gpu_settings():
    """Optimize GPU settings for RTX 4070 SUPER maximum performance."""
    if torch.cuda.is_available():
        print("🚀 Optimizing for RTX 4070 SUPER...")
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("✅ GPU optimizations enabled")
        return True
    return False

def train_triangle_model():
    """Train the triangle detection model with optimized settings."""
    print("🎯 Training Triangle Detection Model (3-Class: HUD + Triangles)")
    print("=" * 70)
    
    # GPU optimization
    gpu_available = optimize_gpu_settings()
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Error importing ultralytics: {e}")
        print("💡 Install with: pip install ultralytics")
        return
    
    # Configuration optimized for RTX 4070 SUPER (12GB VRAM)
    config = {
        # Model configuration
        "model": "yolov8s.pt",  # YOLOv8 Small - optimal for RTX 4070 SUPER
        "data": "yolo_triangle_dataset/dataset.yaml",
        
        # Training parameters - OPTIMIZED for RTX 4070 SUPER 
        "epochs": 50,  # More epochs for better training
        "batch": 32,   # OPTIMAL batch size for 12GB VRAM
        "imgsz": 640,  # Standard image size
        "device": "0" if gpu_available else "cpu",  # Use GPU 0
        
        # Learning parameters
        "lr0": 0.01,      # Initial learning rate
        "lrf": 0.01,      # Final learning rate factor
        "momentum": 0.937, # SGD momentum
        "weight_decay": 0.0005,  # Weight decay
        "warmup_epochs": 3,      # Warmup epochs
        "warmup_momentum": 0.8,  # Warmup momentum
        "warmup_bias_lr": 0.1,   # Warmup bias learning rate
        
        # Optimization settings - MAXIMUM PERFORMANCE
        "optimizer": "auto",  # Auto-select best optimizer
        "close_mosaic": 10,   # Close mosaic augmentation
        "amp": True,          # Mixed precision training (FP16) - CRUCIAL for speed
        "fraction": 1.0,      # Use 100% of dataset
        "profile": False,     # Disable profiling for speed
        "freeze": None,       # Don't freeze any layers
        
        # Data augmentation - Moderate for good performance
        "hsv_h": 0.015,       # Hue augmentation
        "hsv_s": 0.7,         # Saturation augmentation  
        "hsv_v": 0.4,         # Value augmentation
        "degrees": 5.0,       # Rotation degrees
        "translate": 0.1,     # Translation
        "scale": 0.5,         # Scale
        "shear": 0.0,         # Shear
        "perspective": 0.0,   # Perspective
        "flipud": 0.0,        # Vertical flip
        "fliplr": 0.5,        # Horizontal flip - important for triangles
        "mosaic": 1.0,        # Mosaic augmentation
        "mixup": 0.1,         # Mixup augmentation
        "copy_paste": 0.1,    # Copy-paste augmentation
        
        # Output settings
        "project": "triangle_training",
        "name": "triangle_detection_rtx4070_optimized",
        "save": True,
        "save_period": 5,     # Save every 5 epochs
        "cache": "ram",       # Cache images in RAM for speed
        "workers": 8,         # OPTIMAL for RTX 4070 SUPER system
        "verbose": True,
        "plots": True,
        
        # Advanced optimizations
        "deterministic": False,  # Allow non-deterministic for speed
        "single_cls": False,     # Multi-class detection
        "rect": False,           # Rectangular training
        "cos_lr": True,          # Cosine learning rate scheduler
        "patience": 15,          # Early stopping patience
        "val": True,             # Enable validation
        "save_json": True,       # Save JSON results
        "save_hybrid": True,     # Save hybrid labels
        "conf": None,            # Default confidence threshold
        "iou": 0.6,              # NMS IoU threshold
        "max_det": 100,          # Maximum detections per image
        "half": False,           # Don't use half precision for model weights (amp handles this)
        "dnn": False,            # Don't use OpenCV DNN
        "plots": True,           # Generate training plots
        "overlap_mask": True,    # Overlap masks
        "mask_ratio": 4,         # Mask downsample ratio
        "dropout": 0.0,          # Dropout (disabled)
        "val_split": 0.2,        # Validation split
    }
    
    print("\n🔧 Training Configuration (RTX 4070 SUPER Optimized):")
    print(f"   Model: {config['model']}")
    print(f"   Batch Size: {config['batch']} (OPTIMAL for 12GB VRAM)")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Device: {config['device']}")
    print(f"   Mixed Precision: {config['amp']} (FP16 for SPEED)")
    print(f"   Workers: {config['workers']}")
    print(f"   Image Size: {config['imgsz']}")
    print(f"   Cache: {config['cache']} (RAM caching for SPEED)")
    
    # Verify dataset exists
    dataset_path = Path(config["data"])
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("💡 Run: python convert_labelme_to_yolo.py --input augmented_triangle_annotations")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Load model with optimizations
    print("\n🤖 Loading YOLOv8s model...")
    try:
        model = YOLO(config["model"])
        
        # Enable model compilation for extra speed (if available)
        if hasattr(torch, 'compile') and gpu_available:
            print("🚀 Enabling torch.compile for maximum speed...")
            try:
                model.model = torch.compile(model.model, mode="reduce-overhead")
                print("✅ Model compilation enabled")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
        
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Start training with all optimizations
    print(f"\n🚀 Starting OPTIMIZED training on {config['device'].upper()}...")
    print("💪 Using maximum RTX 4070 SUPER performance settings!")
    
    try:
        # Train the model
        results = model.train(**config)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        print(f"📊 Best model: {results.save_dir}/weights/best.pt")
        print(f"📈 Final metrics:")
        
        if hasattr(results, 'metrics'):
            metrics = results.metrics
            if hasattr(metrics, 'results_dict'):
                for key, value in metrics.results_dict.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        
        # Performance summary
        print(f"\n⚡ RTX 4070 SUPER Performance Summary:")
        print(f"   🔥 Batch Size: {config['batch']} (12GB VRAM utilized)")
        print(f"   ⚡ Mixed Precision: Enabled (FP16)")
        print(f"   🚀 Torch Compile: {'Enabled' if gpu_available else 'Not Available'}")
        print(f"   💾 RAM Caching: Enabled")
        print(f"   🔄 Workers: {config['workers']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    # Clear any previous GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🎯 Triangle Detection Training - RTX 4070 SUPER OPTIMIZED")
    print("=" * 70)
    print("🔥 Maximum performance configuration enabled!")
    print("⚡ Batch size: 32 | Mixed precision: FP16 | RAM caching: ON")
    print("=" * 70)
    
    results = train_triangle_model()
    
    if results:
        print("\n🎉 Training successful!")
        print(f"📁 Best model: {results.save_dir}/weights/best.pt")
        print(f"🔍 Next steps:")
        print(f"1. Test the model with: python test_triangle_model.py --model {results.save_dir}/weights/best.pt")
        print(f"2. Update GUI to use new model")
        print(f"3. Run live detection tests")
        print("🚀 Your RTX 4070 SUPER delivered MAXIMUM PERFORMANCE! 💪")
    else:
        print("\n❌ Training failed. Check the errors above.") 
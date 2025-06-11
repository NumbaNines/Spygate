#!/usr/bin/env python3
"""
Improved Triangle Detection Training - Optimized for Higher Confidence
Based on analysis of current working model performance.
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def train_improved_triangle_model():
    """Train with optimized parameters for triangle detection confidence."""
    print("🎯 Training Improved Triangle Detection Model")
    print("=" * 50)
    
    # Use smaller model for small object detection
    model = YOLO("yolov8n.pt")  # Nano model better for small objects
    
    # Check if dataset exists (using curated 500-image dataset)
    dataset_yaml = "yolo_triangle_dataset_500/dataset.yaml"
    if not Path(dataset_yaml).exists():
        print(f"❌ Dataset not found: {dataset_yaml}")
        print("Please create the triangle dataset first!")
        return
    
    # Optimized training parameters
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.001,  # Lower learning rate
        patience=20,
        save=True,
        cache=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        
        # Small object optimization
        mosaic=0.5,  # Reduce mosaic for better small object detection
        mixup=0.0,   # Disable mixup for triangles
        copy_paste=0.0,  # Disable copy-paste
        
        # Augmentation for triangles
        degrees=15.0,     # Moderate rotation
        translate=0.1,    # Small translation
        scale=0.2,        # Small scale variation
        shear=5.0,        # Minimal shear
        perspective=0.0,  # No perspective for small objects
        flipud=0.0,       # No vertical flip for triangles
        fliplr=0.5,       # Horizontal flip OK
        
        # Loss optimization
        box=7.5,          # Higher box loss weight
        cls=0.5,          # Default class loss
        dfl=1.5,          # Default distribution focal loss
        
        # Validation
        val=True,
        split="val",
        
        # Output
        project="triangle_training_improved",
        name="high_confidence_triangles",
        exist_ok=True,
        verbose=True
    )
    
    print(f"✅ Training complete!")
    print(f"📊 Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"📁 Model saved to: triangle_training_improved/high_confidence_triangles/weights/best.pt")

if __name__ == "__main__":
    train_improved_triangle_model()

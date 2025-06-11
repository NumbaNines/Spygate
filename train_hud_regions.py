#!/usr/bin/env python3
"""
Train SpygateAI HUD Region Detection Model
5-class system: hud, possession_triangle_area, territory_triangle_area, preplay_indicator, play_call_screen
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import os

def main():
    """Train the HUD region detection model."""
    print("🚀 Starting SpygateAI HUD Region Detection Training...")
    
    # Check if dataset exists
    dataset_path = Path("hud_region_training/dataset/dataset.yaml")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run convert_labelme_to_yolo.py first")
        return
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Create output directory
    output_dir = Path("hud_region_training/runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize YOLOv8 model (medium for good balance)
    print("📦 Loading YOLOv8m model...")
    model = YOLO('yolov8m.pt')  # medium model for good accuracy/speed balance
    
    # Training parameters
    training_params = {
        'data': str(dataset_path),
        'epochs': 100,
        'batch': 16 if device == 'cuda' else 8,
        'imgsz': 640,
        'device': device,
        'project': str(output_dir),
        'name': 'hud_regions_v1',
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 20,     # Early stopping patience
        'workers': 8,
        'cos_lr': True,     # Cosine learning rate scheduler
        'close_mosaic': 20, # Close mosaic augmentation in last 20 epochs
        'amp': True,        # Automatic Mixed Precision
        'cache': 'ram',     # Cache dataset in RAM for faster training
    }
    
    print("🎯 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    print("\n🔥 Starting training...")
    
    # Start training
    try:
        results = model.train(**training_params)
        print("✅ Training completed successfully!")
        
        # Print final results
        if hasattr(results, 'box'):
            map50 = results.box.map50
            map50_95 = results.box.map50_95
            print(f"📊 Final Results:")
            print(f"   mAP@0.5: {map50:.4f}")
            print(f"   mAP@0.5:0.95: {map50_95:.4f}")
        
        # Find best weights
        best_weights = output_dir / "hud_regions_v1" / "weights" / "best.pt"
        if best_weights.exists():
            print(f"🏆 Best weights saved at: {best_weights}")
            
            # Copy to main models directory
            models_dir = Path("spygate/ml/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = models_dir / "hud_regions_best.pt"
            
            import shutil
            shutil.copy2(best_weights, final_model_path)
            print(f"📦 Model copied to: {final_model_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    print("🎉 Training complete! Ready for inference testing.")

if __name__ == "__main__":
    main() 
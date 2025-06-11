#!/usr/bin/env python3
"""Check model information and compare working vs massive models."""

from ultralytics import YOLO
from pathlib import Path

def check_models():
    """Check both models and compare."""
    print("🔍 Model Comparison Report")
    print("=" * 50)
    
    models = [
        ("WORKING MODEL", "triangle_training/triangle_detection_correct/weights/best.pt"),
        ("MASSIVE MODEL", "triangle_training_massive/weights/best.pt")
    ]
    
    for name, path in models:
        print(f"\n🔍 {name}")
        print("-" * 30)
        
        if Path(path).exists():
            file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
            print(f"📁 File size: {file_size:.1f} MB")
            
            try:
                model = YOLO(path)
                print(f"📊 Classes: {model.names}")
                print(f"🔢 Number of classes: {len(model.names)}")
                print(f"✅ Model loads successfully")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        else:
            print(f"❌ Model file not found: {path}")

if __name__ == "__main__":
    check_models() 
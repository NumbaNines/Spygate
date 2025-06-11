#!/usr/bin/env python3
"""Improve triangle detection confidence through better training data and methods."""

import cv2
import numpy as np
from pathlib import Path
import json
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_current_model_performance():
    """Analyze why confidence is low in current working model."""
    print("🔍 Analyzing Current Model Performance")
    print("=" * 50)
    
    # Load working model
    model_path = "triangle_training/triangle_detection_correct/weights/best.pt"
    if not Path(model_path).exists():
        print("❌ Working model not found!")
        return
    
    model = YOLO(model_path)
    print(f"✅ Loaded model: {model_path}")
    
    # Check training metrics
    results_path = "triangle_training/triangle_detection_correct/results.csv"
    if Path(results_path).exists():
        print(f"📊 Checking training results...")
        # Could analyze the CSV for overfitting, loss curves, etc.
    
    # Test on various confidence levels
    test_image = None
    test_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6.png",
        "images_to_annotate/monitor3_screenshot_20250608_021042_6.png",
        "triangle_visualization_3.jpg"
    ]
    
    for img in test_images:
        if Path(img).exists():
            test_image = img
            break
    
    if test_image:
        print(f"📷 Testing confidence levels on: {test_image}")
        image = cv2.imread(test_image)
        
        confidence_levels = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        for conf in confidence_levels:
            results = model(image, conf=conf, verbose=False)
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"  Conf {conf:.2f}: {detections} detections")

def create_high_quality_dataset():
    """Create a smaller, higher-quality dataset for retraining."""
    print("\n🎯 Creating High-Quality Dataset Strategy")
    print("=" * 50)
    
    # Analyze current training data
    dataset_dir = Path("yolo_triangle_dataset")
    if dataset_dir.exists():
        train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
        val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
        
        print(f"📊 Current dataset:")
        print(f"  Training labels: {len(train_labels)}")
        print(f"  Validation labels: {len(val_labels)}")
        
        # Analyze annotation quality
        triangle_counts = defaultdict(int)
        small_boxes = 0
        
        for label_file in train_labels[:50]:  # Sample first 50
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            w, h = float(parts[3]), float(parts[4])
                            if w < 0.02 or h < 0.02:  # Very small boxes
                                small_boxes += 1
                            triangle_counts[cls] += 1
        
        print(f"📊 Annotation Analysis (sample):")
        print(f"  Small boxes (< 2% image): {small_boxes}")
        print(f"  Class distribution: {dict(triangle_counts)}")

def recommend_training_improvements():
    """Recommend specific improvements for higher confidence."""
    print("\n💡 Training Improvement Recommendations")
    print("=" * 50)
    
    improvements = [
        {
            "strategy": "🎯 Precision Annotation",
            "description": "Manually review and improve bounding box precision",
            "action": "Use labelme to re-annotate 50-100 best quality images",
            "expected_gain": "High confidence improvement"
        },
        {
            "strategy": "📏 Minimum Size Filtering", 
            "description": "Remove very small triangle annotations that are hard to detect",
            "action": "Filter out boxes smaller than 15x15 pixels",
            "expected_gain": "Reduces false negatives, improves precision"
        },
        {
            "strategy": "🔄 Balanced Augmentation",
            "description": "Use moderate, realistic augmentation",
            "action": "Rotation ±15°, brightness ±20%, no extreme distortions",
            "expected_gain": "Better generalization without overfitting"
        },
        {
            "strategy": "📊 Model Size Optimization",
            "description": "Use YOLOv8n or YOLOv8s for small objects",
            "action": "Train with yolov8n.pt for triangle-specific detection",
            "expected_gain": "Better small object detection"
        },
        {
            "strategy": "⚡ Training Parameters",
            "description": "Optimize for small object detection",
            "action": "Lower learning rate, more epochs, mosaic=0.5",
            "expected_gain": "More stable convergence"
        }
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement['strategy']}")
        print(f"   📝 {improvement['description']}")
        print(f"   🔧 Action: {improvement['action']}")
        print(f"   📈 Expected: {improvement['expected_gain']}\n")

def create_improved_training_script():
    """Generate an improved training script based on recommendations."""
    print("📝 Creating Improved Training Script")
    print("=" * 30)
    
    script_content = '''#!/usr/bin/env python3
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
    
    # Optimized training parameters
    results = model.train(
        data="yolo_triangle_dataset/triangle_dataset.yaml",
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
'''
    
    # Save the script
    script_path = "train_improved_triangle_model.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: {script_path}")
    print("🚀 Run with: python train_improved_triangle_model.py")

def main():
    """Main analysis and improvement pipeline."""
    analyze_current_model_performance()
    create_high_quality_dataset()
    recommend_training_improvements()
    create_improved_training_script()
    
    print("\n🎯 NEXT STEPS FOR HIGHER CONFIDENCE:")
    print("1. Review and improve annotation quality")
    print("2. Create filtered, high-quality dataset")
    print("3. Run improved training script")
    print("4. Test and compare confidence levels")
    print("5. Iterate until confidence > 0.5")

if __name__ == "__main__":
    main() 
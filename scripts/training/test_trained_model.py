#!/usr/bin/env python3
"""Test the trained YOLOv8 HUD model."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_model():
    """Test the trained model."""
    print("🏈 Testing Trained YOLOv8 HUD Model")
    print("=" * 40)
    
    # Load the best trained model
    model_path = "runs/train/hud_detection_20250610_114636/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Test on some training images (should work perfectly)
    test_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6_copy_0001.png",
        "training_data/images/monitor3_screenshot_20250608_021427_50_copy_0001.png", 
        "training_data/images/monitor3_screenshot_20250608_021044_7_copy_0001.png"
    ]
    
    print(f"\n🧪 Testing on {len(test_images)} training images...")
    
    for i, img_path in enumerate(test_images, 1):
        if not Path(img_path).exists():
            print(f"❌ Image not found: {img_path}")
            continue
            
        print(f"\n🖼️  Testing image {i}: {Path(img_path).name}")
        
        # Run detection
        results = model(img_path, conf=0.1, verbose=False)  # Low confidence for testing
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   ✅ Found {len(boxes)} detections:")
                
                for j, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    confidence = float(conf)
                    print(f"      {j+1}. {class_name}: {confidence:.3f}")
            else:
                print("   ⚠️  No detections found")
        else:
            print("   ❌ No results returned")
    
    # Test on one validation image
    print(f"\n🔍 Testing on validation images...")
    val_images = list(Path("training_data/val/images").glob("*.png"))
    if val_images:
        test_val_image = val_images[0]
        print(f"🖼️  Testing validation image: {test_val_image.name}")
        
        results = model(str(test_val_image), conf=0.1, verbose=False)
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   ✅ Found {len(boxes)} detections on validation image")
                for j, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    confidence = float(conf)
                    print(f"      {j+1}. {class_name}: {confidence:.3f}")
            else:
                print("   ⚠️  No detections on validation image")
    else:
        print("   ❌ No validation images found")
    
    # Print model info
    print(f"\n📋 Model Information:")
    print(f"   Classes: {list(model.names.values())}")
    print(f"   Total classes: {len(model.names)}")

if __name__ == "__main__":
    test_model() 
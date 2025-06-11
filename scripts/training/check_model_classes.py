#!/usr/bin/env python3
"""Check triangle model classes and test detection."""

import cv2
import numpy as np
from ultralytics import YOLO

def check_model():
    """Check the triangle model classes."""
    
    # Load the triangle model
    model_path = "runs/detect/spygate_triangles_20250610_120853/weights/best.pt"
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Print all classes
    print(f"\nModel has {len(model.names)} classes:")
    for i, name in model.names.items():
        print(f"  {i}: {name}")
    
    # Test with actual training images that should have triangles
    test_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6_copy_0001.png",
        "training_data/images/monitor3_screenshot_20250608_021042_6_copy_0002.png", 
        "training_data/images/monitor3_screenshot_20250608_021042_6_copy_0003.png"
    ]
    
    for img_path in test_images:
        try:
            print(f"\n🖼️ Testing {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Could not load {img_path}")
                continue
                
            results = model(image, conf=0.01, iou=0.45, verbose=False)
            
            triangle_found = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for j in range(len(boxes)):
                        cls_id = int(boxes.cls[j].item())
                        conf = boxes.conf[j].item()
                        class_name = model.names[cls_id]
                        
                        print(f"  {class_name}: {conf:.3f}")
                        
                        if 'indicator' in class_name:
                            triangle_found = True
                            print(f"    🔺 TRIANGLE: {class_name}")
            
            if not triangle_found:
                print("  ❌ No triangles detected")
                        
        except Exception as e:
            print(f"Error testing {img_path}: {e}")

if __name__ == "__main__":
    check_model() 
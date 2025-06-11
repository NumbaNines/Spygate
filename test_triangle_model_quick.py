#!/usr/bin/env python3
"""Quick test of triangle model to verify classes and detection."""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_triangle_model():
    """Test the triangle model and check its classes."""
    
    # Load the triangle model
    model_path = "runs/detect/spygate_triangles_20250610_120853/weights/best.pt"
    
    print(f"🔍 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check model info
    print(f"📋 Model classes: {model.names}")
    print(f"📊 Number of classes: {len(model.names)}")
    
    # Test with a sample image from training
    training_images = list(Path("training_data/images").glob("*.jpg"))
    if training_images:
        test_image = training_images[0]
        print(f"🖼️ Testing on: {test_image}")
        
        # Load and test
        image = cv2.imread(str(test_image))
        results = model(image, conf=0.1, iou=0.45, verbose=True)
        
        print(f"\n🎯 Detection Results:")
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    conf = boxes.conf[j].item()
                    class_name = model.names[cls_id]
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                    
                    print(f"  - {class_name}: {conf:.3f} at [{x1},{y1},{x2},{y2}]")
                    
                    # Highlight triangles
                    if 'indicator' in class_name:
                        print(f"    🔺 TRIANGLE DETECTED: {class_name}")
            else:
                print("  No detections found")
    else:
        print("❌ No training images found to test with")

if __name__ == "__main__":
    test_triangle_model() 
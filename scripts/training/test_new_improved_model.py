#!/usr/bin/env python3
"""Test the newly trained improved triangle model."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

def test_improved_model():
    """Test the new improved model confidence."""
    print("🚀 Testing NEW Improved Triangle Model")
    print("=" * 50)
    
    # Use the NEW improved model
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ New model not found: {model_path}")
        return False
    
    print(f"✅ Loading NEW improved model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📊 Model classes: {model.names}")
    
    # Find a test image
    test_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6.png",
        "images_to_annotate/monitor3_screenshot_20250608_021042_6.png",
        "triangle_visualization_3.jpg"
    ]
    
    test_image = None
    for img in test_images:
        if Path(img).exists():
            test_image = img
            break
    
    if not test_image:
        print("❌ No test images found!")
        return False
    
    print(f"📷 Testing with: {test_image}")
    
    # Load image
    image = cv2.imread(test_image)
    if image is None:
        print(f"❌ Failed to load image: {test_image}")
        return False
    
    print("\n🔍 CONFIDENCE COMPARISON TEST:")
    print("=" * 40)
    
    # Test different confidence levels
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for conf in confidence_levels:
        results = model(image, conf=conf, verbose=False)
        
        detections_found = 0
        triangles_found = 0
        hud_found = 0
        max_triangle_conf = 0
        max_hud_conf = 0
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                detections_found = len(boxes)
                
                for score, cls in zip(scores, classes):
                    if cls == 0:  # HUD
                        hud_found += 1
                        max_hud_conf = max(max_hud_conf, score)
                    elif cls in [1, 2]:  # Triangles
                        triangles_found += 1
                        max_triangle_conf = max(max_triangle_conf, score)
        
        # Color coding for confidence levels
        if triangles_found > 0:
            status = "🟢 TRIANGLES!" if conf >= 0.3 else "🟡 triangles" if conf >= 0.1 else "🔵 triangles"
        else:
            status = "❌ no triangles"
        
        print(f"  Conf {conf:.2f}: {detections_found:2d} total | HUD: {hud_found} | Triangles: {triangles_found} | {status}")
        if max_triangle_conf > 0:
            print(f"           Max triangle conf: {max_triangle_conf:.3f}")
        if max_hud_conf > 0:
            print(f"           Max HUD conf: {max_hud_conf:.3f}")
    
    # Run detailed detection at low confidence
    print(f"\n🎯 DETAILED DETECTION (conf=0.05):")
    print("=" * 40)
    
    results = model(image, conf=0.05, verbose=False)
    
    if results and len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            print(f"📊 Found {len(boxes)} detections:")
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                
                if cls == 0:  # HUD
                    print(f"  {i+1}. 🎮 HUD: {class_name} (conf: {score:.3f}) ⭐")
                elif cls in [1, 2]:  # Triangles
                    triangle_type = "POSSESSION" if cls == 1 else "TERRITORY"
                    if score >= 0.3:
                        print(f"  {i+1}. 🟢 {triangle_type} TRIANGLE: {class_name} (conf: {score:.3f}) 🔥 EXCELLENT!")
                    elif score >= 0.1:
                        print(f"  {i+1}. 🟡 {triangle_type} TRIANGLE: {class_name} (conf: {score:.3f}) ✅ GOOD!")
                    else:
                        print(f"  {i+1}. 🔵 {triangle_type} TRIANGLE: {class_name} (conf: {score:.3f}) 📈 IMPROVED!")
                else:
                    print(f"  {i+1}. 📦 {class_name}: {score:.3f}")
    
    return True

if __name__ == "__main__":
    success = test_improved_model()
    if success:
        print("\n✅ New improved model tested!")
        print("🚀 Ready to update GUI to use this model!") 
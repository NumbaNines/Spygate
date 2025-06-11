#!/usr/bin/env python3
"""
Find Triangle Detection Confidence Level
Systematically test different confidence levels to find where triangles appear
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def find_triangle_confidence():
    """Find the confidence level where triangles start appearing."""
    print("🔍 Finding Triangle Detection Confidence Level")
    print("=" * 50)
    
    # Load the NEW improved model
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Test image
    test_image = "triangle_visualization_3.jpg"
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"✅ Loading test image: {test_image}")
    image = cv2.imread(test_image)
    
    # Test at very low confidence levels
    confidence_levels = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    
    triangle_first_found = None
    
    for confidence in confidence_levels:
        print(f"\n🔍 Testing confidence: {confidence:.3f}")
        
        results = model(image, conf=confidence, verbose=False)
        
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                hud_count = 0
                triangle_count = 0
                triangle_confs = []
                
                for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
                    class_id = int(cls.item())
                    class_name = model.names[class_id]
                    conf_val = conf.item()
                    
                    if class_name == 'hud':
                        hud_count += 1
                    elif class_name in ['possession_indicator', 'territory_indicator']:
                        triangle_count += 1
                        triangle_confs.append((class_name, conf_val))
                
                if triangle_count > 0 and triangle_first_found is None:
                    triangle_first_found = confidence
                    print(f"  🎯 FIRST TRIANGLES DETECTED!")
                    for name, conf_val in triangle_confs:
                        print(f"    🔺 {name}: {conf_val:.4f}")
                
                print(f"  Total: {len(detections)} | HUD: {hud_count} | Triangles: {triangle_count}")
            else:
                print(f"  No detections")
        else:
            print(f"  Detection failed")
    
    if triangle_first_found:
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Set GUI confidence to: {triangle_first_found:.3f} or lower")
        print(f"Triangles first appeared at: {triangle_first_found:.3f}")
        
        # Suggest GUI slider value
        gui_value = int(triangle_first_found * 100)  # Convert to GUI slider range
        print(f"GUI slider value: {gui_value}")
    else:
        print(f"\n❌ No triangles found at any tested confidence level!")
        print(f"The model may need retraining with better triangle examples.")

if __name__ == "__main__":
    find_triangle_confidence() 
#!/usr/bin/env python3
"""
Test Triangle Detection at GUI Confidence Level
Test the new improved model at confidence 0.03 (GUI default)
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def test_at_gui_confidence():
    """Test triangle detection at GUI confidence level."""
    print("🎯 Testing Triangle Detection at GUI Confidence (0.03)")
    print("=" * 55)
    
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
    
    # Run detection at GUI confidence (0.03)
    confidence = 0.03
    print(f"\n🔍 Running detection at confidence: {confidence}")
    
    results = model(image, conf=confidence, verbose=False)
    
    # Analyze results
    if results and len(results) > 0:
        detections = results[0].boxes
        if detections is not None and len(detections) > 0:
            print(f"\n📊 Found {len(detections)} detections at conf={confidence}:")
            
            hud_count = 0
            triangle_count = 0
            triangle_confs = []
            
            for i, (box, conf, cls) in enumerate(zip(detections.xyxy, detections.conf, detections.cls)):
                class_id = int(cls.item())
                class_name = model.names[class_id]
                conf_val = conf.item()
                
                if class_name == 'hud':
                    hud_count += 1
                    icon = "🎮"
                elif class_name in ['possession_indicator', 'territory_indicator']:
                    triangle_count += 1
                    triangle_confs.append(conf_val)
                    icon = "🔺"
                else:
                    icon = "❓"
                
                print(f"  {i+1}. {icon} {class_name}: {conf_val:.3f}")
            
            print(f"\n📈 Summary:")
            print(f"  HUD detections: {hud_count}")
            print(f"  Triangle detections: {triangle_count}")
            
            if triangle_count > 0:
                max_tri_conf = max(triangle_confs)
                avg_tri_conf = sum(triangle_confs) / len(triangle_confs)
                print(f"  ✅ TRIANGLES DETECTED!")
                print(f"  Max triangle confidence: {max_tri_conf:.3f}")
                print(f"  Avg triangle confidence: {avg_tri_conf:.3f}")
                print(f"\n🚀 GUI should now detect triangles at confidence 0.03!")
            else:
                print(f"  ❌ No triangles detected at confidence {confidence}")
                print(f"  Try lowering GUI confidence further...")
        else:
            print(f"❌ No detections found at confidence {confidence}")
    else:
        print(f"❌ Detection failed")

if __name__ == "__main__":
    test_at_gui_confidence() 
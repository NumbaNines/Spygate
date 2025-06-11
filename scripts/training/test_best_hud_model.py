#!/usr/bin/env python3
"""
Test the best performing HUD model to evaluate its detection capability
"""

import cv2
import numpy as np
import mss
from ultralytics import YOLO
import time

def test_hud_model():
    """Test HUD model performance"""
    print("🔍 Testing Best HUD Model Performance")
    print("=" * 50)
    
    # Test different models
    models_to_test = [
        {
            'name': 'spygate_hud_detection_fast',
            'path': 'runs/detect/spygate_hud_detection_fast/weights/best.pt',
            'description': 'Fast HUD Detection (mAP50: ~0.407)'
        },
        {
            'name': 'spygate_hud_detection_fast2', 
            'path': 'runs/detect/spygate_hud_detection_fast2/weights/best.pt',
            'description': 'Fast HUD Detection v2 (mAP50: ~0.001)'
        },
        {
            'name': 'triangle_training_improved',
            'path': 'triangle_training_improved/high_confidence_triangles/weights/best.pt',
            'description': 'Triangle Model (HUD: 99.5% mAP50)'
        }
    ]
    
    # Screen capture setup
    sct = mss.mss()
    monitor = sct.monitors[1]
    
    for model_info in models_to_test:
        print(f"\n🎯 Testing: {model_info['name']}")
        print(f"📄 Description: {model_info['description']}")
        print(f"📁 Path: {model_info['path']}")
        
        try:
            # Load model
            model = YOLO(model_info['path'])
            print(f"✅ Model loaded successfully")
            
            # Take test screenshot
            screenshot = np.array(sct.grab(monitor))
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Run inference
            start_time = time.time()
            results = model(screenshot, verbose=False)
            inference_time = time.time() - start_time
            
            print(f"⚡ Inference time: {inference_time:.3f}s")
            
            # Analyze detections
            total_detections = 0
            hud_detections = 0
            triangle_detections = 0
            other_detections = 0
            max_confidence = 0.0
            
            for result in results:
                for box in result.boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    
                    total_detections += 1
                    max_confidence = max(max_confidence, confidence)
                    
                    # Classify detection type based on class names
                    class_names = model.names
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    if 'hud' in class_name.lower():
                        hud_detections += 1
                    elif 'indicator' in class_name.lower() or 'triangle' in class_name.lower():
                        triangle_detections += 1
                    else:
                        other_detections += 1
                    
                    if confidence > 0.1:  # Show high confidence detections
                        print(f"   🎯 {class_name}: {confidence:.3f} confidence")
            
            print(f"📊 Detection Summary:")
            print(f"   Total detections: {total_detections}")
            print(f"   HUD detections: {hud_detections}")
            print(f"   Triangle detections: {triangle_detections}")
            print(f"   Other detections: {other_detections}")
            print(f"   Max confidence: {max_confidence:.3f}")
            
            # Evaluate model suitability
            if hud_detections > 0 and max_confidence > 0.5:
                print(f"✅ {model_info['name']}: EXCELLENT for HUD detection!")
            elif hud_detections > 0 and max_confidence > 0.1:
                print(f"🟡 {model_info['name']}: Good for HUD detection")
            elif total_detections > 0:
                print(f"🟠 {model_info['name']}: Detects objects but poor HUD performance")
            else:
                print(f"❌ {model_info['name']}: No detections")
                
        except Exception as e:
            print(f"❌ Error loading {model_info['name']}: {e}")
    
    print(f"\n🏆 Recommendation:")
    print(f"Based on previous training results, the triangle_training_improved model")
    print(f"achieved 99.5% mAP50 for HUD detection and should be used for:")
    print(f"- YOLO: HUD region detection (99.5% accuracy)")
    print(f"- OpenCV: Element detection within HUD regions")

if __name__ == "__main__":
    test_hud_model() 
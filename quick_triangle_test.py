#!/usr/bin/env python3
"""Quick test to verify the working triangle model is detecting properly."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

def quick_test():
    """Quick test of the working triangle model."""
    print("🔺 Quick Triangle Model Test")
    print("=" * 40)
    
    # Use the WORKING model
    model_path = "triangle_training/triangle_detection_correct/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Working model not found: {model_path}")
        return False
    
    print(f"✅ Loading working model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📊 Model classes: {model.names}")
    
    # Find a test image
    test_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6.png",
        "images_to_annotate/monitor3_screenshot_20250608_021042_6.png",
        "triangle_visualization_3.jpg"  # Use the attached visualization
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
    
    # Run detection with VERY low confidence
    print("🔍 Running detection with conf=0.01...")
    start_time = time.time()
    results = model(image, conf=0.01, iou=0.45)
    inference_time = time.time() - start_time
    
    print(f"⚡ Inference time: {inference_time:.3f}s")
    
    # Process results
    detections_found = 0
    triangles_found = 0
    
    if results and len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            detections_found = len(boxes)
            
            print(f"🎯 Found {detections_found} detections:")
            
            for j, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                
                if cls in [1, 2]:  # Triangle classes
                    triangles_found += 1
                    triangle_type = "POSSESSION" if cls == 1 else "TERRITORY"
                    print(f"  ✅ {triangle_type} TRIANGLE: {class_name} (conf: {score:.3f})")
                else:
                    print(f"  📦 {class_name}: {score:.3f}")
    
    print(f"\n📊 RESULT SUMMARY:")
    print(f"   🔺 Triangles found: {triangles_found}")
    print(f"   📦 Total detections: {detections_found}")
    
    if triangles_found > 0:
        print(f"🎉 SUCCESS! Working model is detecting triangles!")
        return True
    else:
        print(f"⚠️ No triangles detected - model may need investigation")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Working model verified - Ready to use!")
    else:
        print("\n❌ Model verification failed") 
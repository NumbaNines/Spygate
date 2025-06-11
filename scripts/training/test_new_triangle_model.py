#!/usr/bin/env python3
"""
Test the newly trained triangle detection model to verify it's detecting the correct triangles.

This script loads the trained model and tests it on some images to visualize the detections.

Usage:
    python test_new_triangle_model.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os

def test_triangle_model():
    """Test the newly trained triangle model"""
    
    print("🎯 Testing New Triangle Detection Model")
    print("=" * 50)
    
    # Find the latest trained model
    model_paths = [
        "triangle_training/triangle_detection_correct/weights/best.pt",
        "triangle_training/triangle_detection_correct/weights/last.pt",
        "runs/detect/train/weights/best.pt",
        "runs/detect/train/weights/last.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            print(f"Found model: {model_path}")
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Available paths checked:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded model from: {model_path}")
        
        # Print model info
        print(f"Model classes: {model.names}")
        print(f"Number of classes: {len(model.names)}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Find test images
    test_dirs = ["images_to_annotate", "training_data/images", "clean_madden_screenshots"]
    test_images = []
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            images = list(Path(test_dir).glob("*.png"))[:5]  # Take first 5 images
            test_images.extend(images)
            if len(test_images) >= 3:  # We just need a few test images
                break
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"Testing on {len(test_images)} images...")
    
    # Create output directory
    output_dir = Path("triangle_test_results")
    output_dir.mkdir(exist_ok=True)
    
    total_detections = 0
    
    # Test each image
    for i, image_path in enumerate(test_images[:3]):  # Test first 3 images
        print(f"\n📸 Testing image {i+1}: {image_path.name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load {image_path}")
            continue
        
        # Run detection
        results = model(img, conf=0.1, iou=0.5, verbose=False)  # Lower confidence for testing
        
        # Process results
        detections_found = 0
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                detections_found = len(boxes)
                total_detections += detections_found
                
                # Draw detections
                for j, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class name
                    class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                    
                    # Choose color based on class
                    if class_name == "possession_indicator":
                        color = (0, 255, 0)  # Green for possession
                    elif class_name == "territory_indicator":
                        color = (255, 0, 255)  # Magenta for territory
                    else:
                        color = (0, 0, 255)  # Red for unknown
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label
                    label = f"{class_name}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    print(f"  🎯 Detection {j+1}: {class_name} ({score:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
        
        print(f"  Total detections: {detections_found}")
        
        # Save result image
        output_path = output_dir / f"test_result_{i+1}_{image_path.stem}.png"
        cv2.imwrite(str(output_path), img)
        print(f"  💾 Saved result: {output_path}")
    
    print(f"\n🎯 Test Summary:")
    print(f"  Total images tested: {len(test_images[:3])}")
    print(f"  Total triangles detected: {total_detections}")
    print(f"  Average detections per image: {total_detections / len(test_images[:3]):.1f}")
    print(f"  Results saved to: {output_dir}")
    
    print(f"\n🔍 Check the results in '{output_dir}' folder to see what triangles were detected!")

if __name__ == "__main__":
    test_triangle_model() 
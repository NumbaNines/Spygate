"""
Test the trained triangle detection model to verify correct triangle detection.

This script loads the best trained model and tests it on the original training images
to show what triangles it's detecting.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

def test_triangle_model():
    """Test the trained triangle model on original images"""
    
    print("🔍 Testing Trained Triangle Detection Model")
    print("=" * 50)
    
    # Load the best trained model
    model_path = "triangle_training/triangle_detection_correct/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        return
    
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Class names for the 2-class triangle model
    class_names = {
        0: "possession_indicator", 
        1: "territory_indicator"
    }
    
    # Test on original annotated images
    test_images = [
        "images_to_annotate/monitor3_screenshot_20250608_021042_6.png",
        "images_to_annotate/monitor3_screenshot_20250608_021217_24.png", 
        "images_to_annotate/monitor3_screenshot_20250608_021532_63.png"
    ]
    
    # Create output directory
    output_dir = Path("triangle_detections_test")
    output_dir.mkdir(exist_ok=True)
    
    total_detections = 0
    
    for i, img_path in enumerate(test_images):
        if not Path(img_path).exists():
            print(f"⚠️  Image not found: {img_path}")
            continue
            
        print(f"\n🖼️  Testing image {i+1}: {Path(img_path).name}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Failed to load image: {img_path}")
            continue
            
        # Run detection
        results = model(image, conf=0.25, iou=0.5)
        
        # Process results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy() 
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for j, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    detection_info = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'class': class_name,
                        'class_id': cls
                    }
                    detections.append(detection_info)
                    
                    print(f"  ✅ {class_name}: [{x1}, {y1}, {x2}, {y2}] confidence: {score:.3f}")
                    
                    # Draw detection on image
                    color = (0, 255, 0) if cls == 0 else (255, 0, 255)  # Green for possession, Purple for territory
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    
                    # Add label
                    label = f"{class_name}: {score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        total_detections += len(detections)
        print(f"  📊 Found {len(detections)} triangles")
        
        # Save annotated image
        output_path = output_dir / f"test_result_{i+1}_{Path(img_path).stem}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"  💾 Saved result to: {output_path}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total images tested: {len([p for p in test_images if Path(p).exists()])}")
    print(f"   Total triangles detected: {total_detections}")
    print(f"   Results saved to: {output_dir}/")
    
    # Also test on a few augmented images to see variety
    print(f"\n🔄 Testing on a few augmented images...")
    augmented_dir = Path("augmented_triangle_annotations")
    if augmented_dir.exists():
        augmented_images = list(augmented_dir.glob("*.png"))[:3]
        
        aug_detections = 0
        for i, img_path in enumerate(augmented_images):
            print(f"\n🖼️  Testing augmented image {i+1}: {img_path.name}")
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            results = model(image, conf=0.25, iou=0.5)
            
            detections = 0
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    detections = len(boxes)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = class_names.get(cls, f"class_{cls}")
                        print(f"  ✅ {class_name}: [{x1}, {y1}, {x2}, {y2}] confidence: {score:.3f}")
            
            aug_detections += detections
            print(f"  📊 Found {detections} triangles")
        
        print(f"\n📈 AUGMENTED SUMMARY:")
        print(f"   Augmented images tested: {len(augmented_images)}")
        print(f"   Total triangles detected: {aug_detections}")

if __name__ == "__main__":
    test_triangle_model() 
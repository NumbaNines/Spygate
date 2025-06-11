#!/usr/bin/env python3
"""Test the improved YOLOv8 model with focus on hashmarks and QB position detection."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

def load_trained_model():
    """Load the latest trained model."""
    # Look for the latest training run
    runs_dir = Path("runs/detect")
    if not runs_dir.exists():
        print("❌ No training runs found!")
        return None
    
    # Find the latest spygate_hud_detection_fast run
    model_dirs = list(runs_dir.glob("spygate_hud_detection_fast*"))
    if not model_dirs:
        print("❌ No spygate_hud_detection_fast runs found!")
        return None
    
    # Get the latest one (by modification time)
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model weights not found at {model_path}")
        return None
    
    print(f"✅ Loading model from: {model_path}")
    return YOLO(str(model_path))

def test_model_on_images(model, num_images=5):
    """Test the model on sample images focusing on key elements."""
    
    # Class names for our 6 classes
    class_names = ["hud", "qb_position", "left_hash_mark", "right_hash_mark", "preplay", "playcall"]
    
    # Get test images
    images_dir = Path("training_data/images")
    image_files = list(images_dir.glob("*.png"))[:num_images]
    
    print(f"\n🧪 Testing model on {len(image_files)} images...")
    print("=" * 60)
    
    detection_summary = {class_name: 0 for class_name in class_names}
    
    for i, img_path in enumerate(image_files):
        print(f"\n📸 Testing image {i+1}: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not load {img_path}")
            continue
        
        # Run detection
        results = model(img, conf=0.1, iou=0.45)
        
        detections_found = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    conf = boxes.conf[j].item()
                    class_name = class_names[cls_id]
                    
                    detections_found.append(f"{class_name} ({conf:.2f})")
                    detection_summary[class_name] += 1
        
        if detections_found:
            print(f"   ✅ Detections: {', '.join(detections_found)}")
        else:
            print(f"   ❌ No detections found")
    
    print(f"\n📊 DETECTION SUMMARY:")
    print("=" * 60)
    for class_name, count in detection_summary.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {class_name}: {count} detections")
    
    # Key metrics we care about
    key_detections = {
        "Hash Marks": detection_summary["left_hash_mark"] + detection_summary["right_hash_mark"],
        "QB Position": detection_summary["qb_position"],
        "HUD": detection_summary["hud"],
        "Game State": detection_summary["preplay"] + detection_summary["playcall"]
    }
    
    print(f"\n🎯 KEY PERFORMANCE METRICS:")
    print("=" * 60)
    for metric, count in key_detections.items():
        status = "✅ GOOD" if count > 0 else "❌ NEEDS WORK"
        print(f"   {metric}: {count} detections - {status}")
    
    return detection_summary

def visualize_detections(model, image_path):
    """Visualize detections on a specific image."""
    class_names = ["hud", "qb_position", "left_hash_mark", "right_hash_mark", "preplay", "playcall"]
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load {image_path}")
        return
    
    # Run detection
    results = model(img, conf=0.1, iou=0.45)
    
    # Draw detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for j in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[j].item())
                conf = boxes.conf[j].item()
                
                # Draw bounding box
                color = colors[cls_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_names[cls_id]}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    output_path = f"detection_test_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, img)
    print(f"✅ Visualization saved as: {output_path}")

def main():
    """Main test function."""
    print("🏈 SpygateAI Improved Model Testing")
    print("=" * 60)
    print("Testing improvements in hashmarks and QB position detection")
    print("Plus new game state detection (preplay/playcall)")
    
    # Load the trained model
    model = load_trained_model()
    if model is None:
        print("❌ Could not load trained model!")
        return
    
    # Test model performance
    detection_summary = test_model_on_images(model, num_images=10)
    
    # Test on a specific image for visualization
    images_dir = Path("training_data/images")
    test_images = list(images_dir.glob("*.png"))
    if test_images:
        print(f"\n🎨 Creating visualization for: {test_images[0].name}")
        visualize_detections(model, test_images[0])
    
    # Overall assessment
    total_key_detections = (
        detection_summary["left_hash_mark"] + 
        detection_summary["right_hash_mark"] + 
        detection_summary["qb_position"] + 
        detection_summary["hud"]
    )
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print("=" * 60)
    if total_key_detections >= 15:
        print("   🌟 EXCELLENT! Model is detecting key elements well!")
    elif total_key_detections >= 8:
        print("   ✅ GOOD! Model shows solid improvement!")
    elif total_key_detections >= 3:
        print("   ⚠️  FAIR - Some detections, but needs more training data")
    else:
        print("   ❌ POOR - Model needs significant improvement")
    
    print(f"\nNext steps:")
    if detection_summary["left_hash_mark"] == 0 or detection_summary["right_hash_mark"] == 0:
        print("   📝 Add more hash mark annotations")
    if detection_summary["qb_position"] == 0:
        print("   📝 Add more QB position annotations")
    if detection_summary["preplay"] == 0 and detection_summary["playcall"] == 0:
        print("   📝 Add more game state annotations (preplay/playcall)")

if __name__ == "__main__":
    main() 
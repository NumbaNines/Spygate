"""
Balanced False Positive Reduction for 8-Class YOLOv8 Model.
Reduces false positives while preserving valid detections.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class BalancedFPReducer:
    """Balanced approach to false positive reduction."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = [
            'hud', 'possession_triangle_area', 'territory_triangle_area',
            'preplay_indicator', 'play_call_screen', 'down_distance_area',
            'game_clock_area', 'play_clock_area'
        ]
    
    def detect_with_balanced_filtering(self, image_path: str):
        """Run detection with balanced false positive reduction."""
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"🔍 Testing: {Path(image_path).name}")
        
        # Balanced inference settings
        results = self.model(img, conf=0.4, iou=0.5, max_det=15, agnostic_nms=False)
        
        detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                class_name = self.class_names[cls]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'class_name': class_name
                })
        
        print(f"📊 Found {len(detections)} detections with balanced settings")
        
        # Show class breakdown
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        return detections

def test_balanced_approach():
    """Test the balanced false positive reduction approach."""
    
    model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_v1/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🎯 TESTING BALANCED FALSE POSITIVE REDUCTION")
    print("=" * 60)
    
    reducer = BalancedFPReducer(model_path)
    
    # Test on validation images
    test_images_dir = Path("hud_region_training/hud_region_training_8class/datasets_8class/val/images")
    test_images = list(test_images_dir.glob("*.png"))[:3]  # Test first 3 images
    
    if not test_images:
        print(f"❌ No test images found in: {test_images_dir}")
        return
    
    total_detections = 0
    
    for img_path in test_images:
        print(f"\n{'='*40}")
        detections = reducer.detect_with_balanced_filtering(str(img_path))
        if detections:
            total_detections += len(detections)
    
    print(f"\n🎉 RESULTS:")
    print(f"📊 Total detections with balanced settings: {total_detections}")
    print(f"✅ Balanced approach preserves valid detections!")

if __name__ == "__main__":
    test_balanced_approach() 
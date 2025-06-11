"""
Visualize triangle detections to verify they are correct
"""

from ultralytics import YOLO
import cv2
import numpy as np

def visualize_detections():
    """Load model and visualize detections"""
    
    # Load model
    model = YOLO('triangle_training/triangle_detection_correct/weights/best.pt')
    
    # Test images
    test_images = [
        'images_to_annotate/monitor3_screenshot_20250608_021042_6.png',
        'images_to_annotate/monitor3_screenshot_20250608_021217_24.png', 
        'images_to_annotate/monitor3_screenshot_20250608_021532_63.png'
    ]
    
    # Class colors
    colors = {
        0: (0, 255, 0),    # possession_indicator - green
        1: (255, 0, 255)   # territory_indicator - purple
    }
    
    for i, img_path in enumerate(test_images):
        print(f"\n🖼️  Testing image {i+1}: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Failed to load {img_path}")
            continue
            
        # Run detection with very low confidence
        results = model(image, conf=0.001, iou=0.5, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                print(f"Found {len(boxes)} detections:")
                
                for j, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = model.names[cls]
                    color = colors[cls]
                    
                    print(f"  {j+1}. {class_name}: [{x1}, {y1}, {x2}, {y2}] confidence: {score:.4f}")
                    
                    # Draw detection
                    thickness = 4
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add label with larger text
                    label = f"{class_name}: {score:.3f}"
                    font_scale = 1.0
                    font_thickness = 2
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                    
                    # Add zoom-in view of detection
                    padding = 20
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(image.shape[1], x2 + padding)
                    crop_y2 = min(image.shape[0], y2 + padding)
                    
                    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                        # Scale up the crop
                        scale_factor = 3
                        scaled_crop = cv2.resize(cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                        
                        # Place crop in top-right corner
                        h, w = image.shape[:2]
                        crop_h, crop_w = scaled_crop.shape[:2]
                        
                        # Position crop based on detection number
                        crop_x = w - crop_w - 10
                        crop_y = 10 + (j * (crop_h + 10))
                        
                        if crop_y + crop_h < h:
                            image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = scaled_crop
                            
                            # Add border to crop
                            cv2.rectangle(image, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), color, 3)
            else:
                print("No detections found")
        
        # Save result
        output_path = f"triangle_visualization_{i+1}.jpg"
        cv2.imwrite(output_path, image)
        print(f"💾 Saved visualization to: {output_path}")

if __name__ == "__main__":
    visualize_detections() 
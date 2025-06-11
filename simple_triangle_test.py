"""
Simple test of triangle detection model with ultra-low confidence
"""

from ultralytics import YOLO
import cv2

# Load model
model = YOLO('triangle_training/triangle_detection_correct/weights/best.pt')

# Test image  
img_path = 'images_to_annotate/monitor3_screenshot_20250608_021042_6.png'

print(f"Testing: {img_path}")
print("Model info:", model.model.names if hasattr(model.model, 'names') else 'No names found')

# Test with ultra-low confidence
results = model(img_path, conf=0.001, iou=0.5, verbose=True)

print(f"Results: {len(results)} result objects")

if results:
    result = results[0]
    print(f"Result type: {type(result)}")
    
    if hasattr(result, 'boxes') and result.boxes is not None:
        print(f"Boxes found: {len(result.boxes)}")
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            print(f"Detection {i+1}: class={cls}, confidence={score:.4f}, box={box}")
    else:
        print("No boxes found in results")
        print(f"Result attributes: {dir(result)}")
        
# Also check model summary
print(f"\nModel summary:")
print(f"Classes: {model.names}")
print(f"Model device: {next(model.model.parameters()).device if hasattr(model, 'model') else 'Unknown'}") 
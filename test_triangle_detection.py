import cv2
import numpy as np
from pathlib import Path
from src.spygate.ml.yolov8_model import EnhancedYOLOv8
from src.spygate.core.hardware import HardwareDetector

def test_triangle_detection():
    """Test triangle detection on the extracted frame."""
    
    # Read the extracted frame
    frame = cv2.imread("test_frame.png")
    if frame is None:
        print("Error: Could not read test_frame.png")
        return
        
    print(f"Frame shape: {frame.shape}")
    
    # Create debug output directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Initialize YOLOv8 model directly with our custom trained model
    hardware = HardwareDetector()
    model = EnhancedYOLOv8(
        model_path="hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt",
        hardware_tier=hardware.detect_tier()
    )
    
    print("Running YOLO detection with our custom HUD model...")
    
    try:
        # Run detection
        detections = model.detect(frame)
        
        print(f"\n=== DETECTION RESULTS ===")
        print(f"Total detections: {len(detections)}")
        
        # Create visualization
        vis_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            print(f"Detection {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Bbox: {bbox}")
            
            # Draw detection on frame
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color mapping
            colors = {
                "hud": (0, 255, 0),  # Green
                "possession_triangle_area": (255, 0, 0),  # Blue  
                "territory_triangle_area": (0, 0, 255),  # Red
                "preplay_indicator": (255, 255, 0),  # Cyan
                "play_call_screen": (255, 0, 255),  # Magenta
            }
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # If it's a triangle area, extract and analyze the region
            if "triangle" in class_name:
                print(f"  Analyzing triangle region...")
                roi = frame[y1:y2, x1:x2]
                
                # Save the ROI for inspection
                roi_filename = f"debug_output/{class_name}_roi.png"
                cv2.imwrite(roi_filename, roi)
                print(f"  Saved ROI to: {roi_filename}")
                
                # Simple triangle analysis - convert to grayscale and find contours
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    print(f"  Largest contour area: {area}")
                    
                    # Draw contour on ROI
                    contour_vis = roi.copy()
                    cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.imwrite(f"debug_output/{class_name}_contour.png", contour_vis)
                    print(f"  Saved contour visualization")
        
        # Save the visualization
        cv2.imwrite("debug_output/detections_visualization.png", vis_frame)
        print(f"\nSaved detection visualization to: debug_output/detections_visualization.png")
        
        # Summary
        triangle_detections = [d for d in detections if "triangle" in d['class']]
        hud_detections = [d for d in detections if d['class'] == 'hud']
        
        print(f"\n=== SUMMARY ===")
        print(f"HUD detections: {len(hud_detections)}")
        print(f"Triangle detections: {len(triangle_detections)}")
        
        for triangle in triangle_detections:
            print(f"  - {triangle['class']}: {triangle['confidence']:.3f}")
            
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_triangle_detection() 
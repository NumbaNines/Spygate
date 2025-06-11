import cv2
import numpy as np
from ultralytics import YOLO

def test_hud_coordinates(video_path, model_path, test_frame_number=3240):
    """
    Interactive tool to test different HUD coordinate regions for team scores.
    """
    
    # Load the model
    model = YOLO(model_path)
    
    # Load video and go to test frame
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames at {fps} FPS")
    
    # Try multiple frames if the specified one doesn't work
    test_frames = [test_frame_number, 1800, 3600, 5400, 7200]  # Different time points
    
    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue
            
        print(f"\nTrying frame {frame_num}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            print(f"Successfully loaded frame {frame_num}")
            break
    else:
        print("Could not read any test frames")
        cap.release()
        return
    
    # Run detection to get HUD box
    results = model(frame, verbose=False)
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Find HUD boxes
            hud_boxes = [box for box in boxes if int(box.cls[0]) == 0]
            
            if hud_boxes:
                hud_box = hud_boxes[0]
                x1, y1, x2, y2 = hud_box.xyxy[0].cpu().numpy()
                hud_region = frame[int(y1):int(y2), int(x1):int(x2)]
                
                h, w = hud_region.shape[:2]
                print(f"HUD region size: {w}x{h}")
                
                # Test different coordinate regions
                test_regions = [
                    # Format: (name, x_start, x_end, y_start, y_end)
                    ("Current Down/Distance", 0.750, 0.900, 0.200, 0.800),  # Existing
                    ("Team Scores Wide", 0.100, 0.600, 0.100, 0.500),      # Initial guess
                    ("Team Scores Narrow", 0.150, 0.550, 0.150, 0.450),    # Narrower
                    ("Left Side Only", 0.100, 0.400, 0.100, 0.500),        # Focus left
                    ("Upper Center", 0.200, 0.600, 0.050, 0.350),          # Upper area
                    ("Full Upper", 0.050, 0.700, 0.050, 0.400),            # Wide upper
                ]
                
                # Create a composite image showing all regions
                composite = hud_region.copy()
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                
                for i, (name, x_start, x_end, y_start, y_end) in enumerate(test_regions):
                    x_start_px = int(w * x_start)
                    x_end_px = int(w * x_end)
                    y_start_px = int(h * y_start)
                    y_end_px = int(h * y_end)
                    
                    color = colors[i % len(colors)]
                    
                    # Draw rectangle
                    cv2.rectangle(composite, (x_start_px, y_start_px), (x_end_px, y_end_px), color, 2)
                    
                    # Add label
                    cv2.putText(composite, name, (x_start_px, y_start_px - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    print(f"\n{i+1}. {name}")
                    print(f"   Coordinates: ({x_start:.3f}-{x_end:.3f}, {y_start:.3f}-{y_end:.3f})")
                    print(f"   Pixels: x({x_start_px}-{x_end_px}), y({y_start_px}-{y_end_px})")
                
                # Save the composite image
                cv2.imwrite(f"hud_coordinate_test_frame_{frame_num}.png", composite)
                print(f"\nComposite image saved as: hud_coordinate_test_frame_{frame_num}.png")
                
                # Also save individual region crops
                for i, (name, x_start, x_end, y_start, y_end) in enumerate(test_regions):
                    x_start_px = int(w * x_start)
                    x_end_px = int(w * x_end)
                    y_start_px = int(h * y_start)
                    y_end_px = int(h * y_end)
                    
                    region_crop = hud_region[y_start_px:y_end_px, x_start_px:x_end_px]
                    filename = f"region_{i+1}_{name.replace(' ', '_').replace('/', '_')}.png"
                    cv2.imwrite(filename, region_crop)
                    print(f"   Saved crop: {filename}")
                
                print("\n🔍 INSTRUCTIONS:")
                print("1. Look at the composite image to see all coordinate regions overlaid")
                print("2. Check individual region crops to see what text is captured")
                print("3. Look for regions that clearly show team abbreviations and scores")
                print("4. Update the coordinates in detect_team_scores_and_possession() based on results")
                
    cap.release()

if __name__ == "__main__":
    # Test with your video and model
    video_path = "live_detections.mp4"  # Use available video file
    model_path = "../hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
    
    # Test frame where you know there should be team scores visible
    test_frame = 1800  # Start with 30 seconds in (1800 frames at 60fps)
    
    print(f"Testing HUD coordinates on frame {test_frame}...")
    test_hud_coordinates(video_path, model_path, test_frame) 
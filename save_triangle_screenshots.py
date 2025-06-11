#!/usr/bin/env python3
"""
Save triangle detection screenshots with bounding boxes.
Bypasses display issues by saving images to files.
"""

import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO
from pathlib import Path
import os

def save_triangle_screenshots():
    """Capture screenshots with triangle detection and save them."""
    
    # Create output directory
    output_dir = Path("triangle_screenshots")
    output_dir.mkdir(exist_ok=True)
    
    # Load the triangle model
    model_path = "runs/detect/spygate_triangles_20250610_123232/weights/best.pt"
    
    print(f"🔍 Loading triangle model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📋 Model classes: {model.names}")
    print(f"📁 Saving screenshots to: {output_dir}")
    
    # Setup screen capture
    with mss.mss() as sct:
        # Get monitor 1
        monitor = sct.monitors[1]
        
        print(f"📺 Monitor resolution: {monitor['width']}x{monitor['height']}")
        print(f"📸 Capturing 10 screenshots with triangle detection...")
        
        screenshots_saved = 0
        triangles_found = 0
        
        for i in range(50):  # Try up to 50 captures to get 10 with triangles
            try:
                # Capture screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Run triangle detection
                results = model(frame, conf=0.05, iou=0.45, verbose=False)
                
                display_frame = frame.copy()
                triangles_this_frame = 0
                detections_this_frame = 0
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        detections_this_frame = len(boxes)
                        
                        # Colors for different classes
                        colors = {
                            0: (0, 0, 255),        # hud - red
                            1: (0, 255, 0),        # qb_position - green
                            2: (255, 0, 0),        # left_hash_mark - blue
                            3: (0, 255, 255),      # right_hash_mark - cyan
                            4: (255, 0, 255),      # preplay - magenta
                            5: (255, 255, 0),      # playcall - yellow
                            6: (255, 165, 0),      # possession_indicator - orange
                            7: (128, 0, 128)       # territory_indicator - purple
                        }
                        
                        for box, score, cls in zip(boxes, scores, classes):
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = model.names.get(cls, f"class_{cls}")
                            color = colors.get(cls, (255, 255, 255))
                            
                            # Count triangles
                            if cls in [6, 7]:  # Triangle classes
                                triangles_this_frame += 1
                                
                                # Special triangle visualization
                                thickness = 6  # Extra thick for triangles
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                                
                                # Triangle labels with background
                                if cls == 6:
                                    label = f"POSSESSION ▶️ {score:.3f}"
                                else:
                                    label = f"TERRITORY ▼▲ {score:.3f}"
                                
                                # Create label background
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1.0
                                thickness_text = 2
                                label_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
                                
                                # Draw label background
                                cv2.rectangle(display_frame, 
                                            (x1, y1-35), 
                                            (x1+label_size[0]+10, y1), 
                                            color, -1)
                                
                                # Draw label text
                                cv2.putText(display_frame, label, 
                                          (x1+5, y1-10), 
                                          font, font_scale, (255, 255, 255), thickness_text)
                                
                                # Add triangle marker in corner
                                cv2.circle(display_frame, (x1+10, y1+10), 8, (255, 255, 255), -1)
                                cv2.putText(display_frame, "🔺", (x1+5, y1+15), font, 0.5, (0, 0, 0), 1)
                            
                            else:
                                # Regular detections
                                thickness = 3
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                                label = f"{class_name}: {score:.2f}"
                                cv2.putText(display_frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add info overlay
                info_bg_height = 150
                cv2.rectangle(display_frame, (10, 10), (500, info_bg_height), (0, 0, 0), -1)
                
                info_text = [
                    f"Screenshot: {i+1}",
                    f"Total detections: {detections_this_frame}",
                    f"🔺 Triangles: {triangles_this_frame}",
                    f"Saved screenshots: {screenshots_saved}",
                    f"Model: Triangle Detection v1.0"
                ]
                
                for j, text in enumerate(info_text):
                    y_pos = 35 + (j * 25)
                    cv2.putText(display_frame, text, (20, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save if triangles detected or if we want some examples anyway
                should_save = triangles_this_frame > 0 or screenshots_saved < 3
                
                if should_save and screenshots_saved < 10:
                    # Create filename
                    timestamp = int(time.time() * 1000)
                    if triangles_this_frame > 0:
                        filename = f"triangle_detection_{screenshots_saved+1:02d}_T{triangles_this_frame}_{timestamp}.png"
                        triangles_found += triangles_this_frame
                    else:
                        filename = f"no_triangles_{screenshots_saved+1:02d}_{timestamp}.png"
                    
                    filepath = output_dir / filename
                    
                    # Save the image
                    success = cv2.imwrite(str(filepath), display_frame)
                    
                    if success:
                        screenshots_saved += 1
                        print(f"📸 Saved: {filename}")
                        if triangles_this_frame > 0:
                            print(f"   🔺 Contains {triangles_this_frame} triangle(s)!")
                    else:
                        print(f"❌ Failed to save: {filename}")
                
                # Quick status update
                if i % 10 == 0:
                    print(f"🔄 Processed {i+1} frames, saved {screenshots_saved} screenshots...")
                
                # Stop if we have enough screenshots
                if screenshots_saved >= 10:
                    break
                
                # Small delay
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Error in frame {i+1}: {e}")
                continue
    
    print(f"\n🎉 SCREENSHOT CAPTURE COMPLETE!")
    print(f"=" * 50)
    print(f"📸 Screenshots saved: {screenshots_saved}")
    print(f"🔺 Total triangles found: {triangles_found}")
    print(f"📁 Location: {output_dir.absolute()}")
    
    # List saved files
    saved_files = list(output_dir.glob("*.png"))
    if saved_files:
        print(f"\n📋 Saved files:")
        for file in sorted(saved_files):
            file_size = file.stat().st_size / 1024 / 1024  # MB
            print(f"   {file.name} ({file_size:.1f} MB)")
    
    print(f"\n💡 Open the '{output_dir}' folder to view your triangle detection screenshots!")
    print(f"🔺 Look for orange and purple boxes around triangles!")

if __name__ == "__main__":
    save_triangle_screenshots() 
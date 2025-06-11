#!/usr/bin/env python3
"""
Console-only triangle detection test - no display functions.
Tests the new triangle model on live screen capture and prints results.
"""

import numpy as np
import mss
import time
from ultralytics import YOLO
from pathlib import Path

def console_triangle_test():
    """Console-only triangle detection test."""
    
    # Load the new triangle model
    model_path = "runs/detect/spygate_triangles_20250610_123232/weights/best.pt"
    
    print(f"🔍 Loading triangle model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📋 Model classes: {model.names}")
    
    # Setup screen capture
    print("🖥️  Starting screen capture (Monitor 1)...")
    print("⏳ Testing triangle detection for 30 seconds...")
    print("📊 Will report detections every 5 seconds and final summary")
    
    with mss.mss() as sct:
        # Get monitor 1
        monitor = sct.monitors[1]
        
        print(f"📺 Monitor resolution: {monitor['width']}x{monitor['height']}")
        print(f"🔄 Starting detection loop...")
        
        frame_count = 0
        triangle_detections = 0
        total_detections = 0
        start_time = time.time()
        last_report_time = start_time
        detection_details = []
        
        # Run for 30 seconds
        while time.time() - start_time < 30:
            try:
                # Capture screen
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                # Convert BGRA to RGB for YOLO
                frame = frame[:, :, :3]  # Remove alpha channel
                
                frame_count += 1
                
                # Run detection with low confidence for triangles
                results = model(frame, conf=0.05, iou=0.45, verbose=False)
                
                triangles_this_frame = 0
                detections_this_frame = 0
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        detections_this_frame = len(boxes)
                        total_detections += detections_this_frame
                        
                        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                            class_name = model.names.get(cls, f"class_{cls}")
                            
                            # Count triangles
                            if cls in [6, 7]:  # Triangle classes
                                triangles_this_frame += 1
                                triangle_detections += 1
                                
                                # Store triangle details
                                triangle_type = "POSSESSION" if cls == 6 else "TERRITORY"
                                detection_details.append({
                                    'frame': frame_count,
                                    'type': triangle_type,
                                    'confidence': score,
                                    'bbox': box
                                })
                
                # Report every 5 seconds
                if time.time() - last_report_time >= 5:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    triangle_rate = (triangle_detections / frame_count * 100) if frame_count > 0 else 0
                    
                    print(f"\\n📊 {elapsed:.0f}s Progress:")
                    print(f"   Frames processed: {frame_count}")
                    print(f"   FPS: {fps:.1f}")
                    print(f"   Total detections: {total_detections}")
                    print(f"   🔺 Triangle detections: {triangle_detections}")
                    print(f"   🔺 Triangle rate: {triangle_rate:.1f}%")
                    
                    last_report_time = time.time()
                
                # Print triangle detections as they happen
                if triangles_this_frame > 0:
                    elapsed = time.time() - start_time
                    print(f"🔺 {elapsed:.1f}s: {triangles_this_frame} triangles detected in frame {frame_count}!")
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"❌ Error in frame {frame_count}: {e}")
                continue
    
    # Final results
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    triangle_rate = (triangle_detections / frame_count * 100) if frame_count > 0 else 0
    
    print(f"\\n🎉 TEST COMPLETED!")
    print(f"=" * 50)
    print(f"📊 Total frames processed: {frame_count}")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📈 Average FPS: {fps:.1f}")
    print(f"🎯 Total detections: {total_detections}")
    print(f"🔺 Total triangles: {triangle_detections}")
    print(f"📊 Triangle detection rate: {triangle_rate:.1f}%")
    
    if triangle_detections > 0:
        print(f"\\n✅ SUCCESS! Triangles are being detected!")
        print(f"\\n🔺 Triangle Detection Details:")
        
        # Group by type
        possession_count = len([d for d in detection_details if d['type'] == 'POSSESSION'])
        territory_count = len([d for d in detection_details if d['type'] == 'TERRITORY'])
        
        print(f"   POSSESSION triangles: {possession_count}")
        print(f"   TERRITORY triangles: {territory_count}")
        
        # Show confidence ranges
        if detection_details:
            confidences = [d['confidence'] for d in detection_details]
            print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"   Average confidence: {np.mean(confidences):.3f}")
        
        # Show sample detections
        print(f"\\n📋 Sample detections:")
        for i, det in enumerate(detection_details[:5]):  # First 5
            print(f"   Frame {det['frame']}: {det['type']} (conf: {det['confidence']:.3f})")
        
        if len(detection_details) > 5:
            print(f"   ... and {len(detection_details) - 5} more")
            
    else:
        print(f"\\n❌ No triangles detected!")
        print(f"💡 Possible reasons:")
        print(f"   - No Madden game running")
        print(f"   - HUD not visible on screen")
        print(f"   - Wrong monitor selected")
        print(f"   - Model needs different confidence threshold")
        
        if total_detections > 0:
            print(f"\\n✅ Other HUD elements detected: {total_detections}")
            print(f"   The model is working, just no triangles in current view")

if __name__ == "__main__":
    console_triangle_test() 
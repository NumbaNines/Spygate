#!/usr/bin/env python3
"""
OpenCV File Visualization Tool for Triangle Detection
Saves visualization frames to files instead of displaying windows
Works around OpenCV GUI issues on Windows
"""

import cv2
import numpy as np
import mss
import time
import os
from pathlib import Path
from ultralytics import YOLO

class OpenCVFileVisualizer:
    def __init__(self, model_path, output_dir="opencv_frames"):
        """Initialize the file-based visualizer"""
        print("🔍 OpenCV File Visualization Tool")
        print("=" * 50)
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Clear previous frames
        for file in self.output_dir.glob("*.jpg"):
            file.unlink()
        
        # Load YOLO model
        print(f"✅ Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
        # Processing parameters
        self.frame_count = 0
        self.save_every_n_frames = 30  # Save every 30 frames (1 second at 30fps)
        
        print(f"💾 Saving visualization frames to: {self.output_dir}")
        print("🎥 Starting capture... Press Ctrl+C to stop")
    
    def detect_triangles_opencv(self, image, hud_region):
        """Detect triangles using OpenCV within HUD region"""
        # Extract HUD region
        x1, y1, x2, y2 = map(int, hud_region)
        hud_crop = image[y1:y2, x1:x2]
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2HSV)
        
        # Color ranges for triangle detection
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([25, 255, 255])
        
        purple_lower = np.array([120, 50, 50])
        purple_upper = np.array([140, 255, 255])
        
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        # Create masks
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(orange_mask, cv2.bitwise_or(purple_mask, white_mask))
        
        # Edge detection
        edges = cv2.Canny(combined_mask, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        triangle_detections = []
        debug_info = {
            'hud_crop': hud_crop,
            'hsv': hsv,
            'orange_mask': orange_mask,
            'purple_mask': purple_mask,
            'white_mask': white_mask,
            'combined_mask': combined_mask,
            'edges': edges,
            'contours_image': hud_crop.copy()
        }
        
        # Draw all contours on debug image
        cv2.drawContours(debug_info['contours_image'], contours, -1, (0, 255, 0), 1)
        
        # Process contours for triangle detection
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's triangle-like (3-4 vertices)
                if len(approx) >= 3 and len(approx) <= 4:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert back to original image coordinates
                    global_x = x1 + x
                    global_y = y1 + y
                    
                    # Calculate confidence based on triangle properties
                    aspect_ratio = w / h if h > 0 else 0
                    confidence = 0.7 if 0.5 < aspect_ratio < 2.0 else 0.4
                    
                    triangle_detections.append({
                        'bbox': [global_x, global_y, global_x + w, global_y + h],
                        'confidence': confidence,
                        'area': area,
                        'vertices': len(approx)
                    })
        
        return triangle_detections, debug_info
    
    def save_visualization_frame(self, screenshot, yolo_results, opencv_triangles, debug_info):
        """Save all visualization frames to files"""
        frame_dir = self.output_dir / f"frame_{self.frame_count:04d}"
        frame_dir.mkdir(exist_ok=True)
        
        # 1. Original screenshot
        cv2.imwrite(str(frame_dir / "1_original.jpg"), screenshot)
        
        # 2. HSV color space
        cv2.imwrite(str(frame_dir / "2_hsv.jpg"), debug_info['hsv'])
        
        # 3. Orange mask (possession indicator)
        cv2.imwrite(str(frame_dir / "3_orange_mask.jpg"), debug_info['orange_mask'])
        
        # 4. Purple mask (territory indicator)
        cv2.imwrite(str(frame_dir / "4_purple_mask.jpg"), debug_info['purple_mask'])
        
        # 5. White mask (alternative detection)
        cv2.imwrite(str(frame_dir / "5_white_mask.jpg"), debug_info['white_mask'])
        
        # 6. Edge detection
        cv2.imwrite(str(frame_dir / "6_edges.jpg"), debug_info['edges'])
        
        # 7. Contours found
        cv2.imwrite(str(frame_dir / "7_contours.jpg"), debug_info['contours_image'])
        
        # 8. Final detections comparison
        final_image = screenshot.copy()
        
        # Draw YOLO detections in red
        for result in yolo_results:
            for box in result.boxes:
                if box.conf.item() > 0.001:  # Very low threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    
                    if class_id in [1, 2]:  # Triangle classes
                        cv2.rectangle(final_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f"YOLO: {confidence:.3f}"
                        cv2.putText(final_image, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw OpenCV detections in green
        for detection in opencv_triangles:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"OpenCV: {confidence:.1f}"
            cv2.putText(final_image, label, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(str(frame_dir / "8_final_detections.jpg"), final_image)
        
        # Save summary info
        summary_path = frame_dir / "detection_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Frame {self.frame_count} Detection Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("YOLO Detections:\n")
            yolo_count = 0
            for result in yolo_results:
                for box in result.boxes:
                    if box.conf.item() > 0.001 and int(box.cls.item()) in [1, 2]:
                        yolo_count += 1
                        f.write(f"  - Confidence: {box.conf.item():.6f}\n")
            f.write(f"Total YOLO triangles: {yolo_count}\n\n")
            
            f.write("OpenCV Detections:\n")
            for i, detection in enumerate(opencv_triangles):
                f.write(f"  - Triangle {i+1}: {detection['confidence']:.1f} confidence, "
                       f"area: {detection['area']}, vertices: {detection['vertices']}\n")
            f.write(f"Total OpenCV triangles: {len(opencv_triangles)}\n")
        
        print(f"💾 Saved frame {self.frame_count} with {yolo_count} YOLO + {len(opencv_triangles)} OpenCV triangles")
    
    def run(self, duration_seconds=30):
        """Run visualization for specified duration"""
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                # Capture screenshot
                screenshot = np.array(self.sct.grab(self.monitor))
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # YOLO detection
                yolo_results = self.model(screenshot, verbose=False)
                
                # Find HUD for OpenCV processing
                hud_region = None
                for result in yolo_results:
                    for box in result.boxes:
                        if int(box.cls.item()) == 0:  # HUD class
                            hud_region = box.xyxy[0].cpu().numpy()
                            break
                    if hud_region is not None:
                        break
                
                # OpenCV triangle detection
                opencv_triangles = []
                debug_info = {}
                if hud_region is not None:
                    opencv_triangles, debug_info = self.detect_triangles_opencv(screenshot, hud_region)
                
                # Save frames every N iterations
                if self.frame_count % self.save_every_n_frames == 0 and debug_info:
                    self.save_visualization_frame(screenshot, yolo_results, opencv_triangles, debug_info)
                
                self.frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        print(f"\n✅ Captured {self.frame_count} frames")
        print(f"📁 Visualization frames saved to: {self.output_dir}")
        print(f"🔍 Check the frame folders for detailed analysis")

def main():
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    visualizer = OpenCVFileVisualizer(model_path)
    visualizer.run(duration_seconds=60)  # Run for 1 minute

if __name__ == "__main__":
    main() 
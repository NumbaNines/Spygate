#!/usr/bin/env python3
"""
OpenCV Live Visualization Tool
==============================
See exactly what OpenCV is detecting at every stage of processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mss
import time
from pathlib import Path

class OpenCVLiveVisualizer:
    """Live visualization of OpenCV detection pipeline."""
    
    def __init__(self, yolo_model_path):
        """Initialize the live visualizer."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # OpenCV processing parameters
        self.color_ranges = {
            'orange': ([8, 80, 80], [25, 255, 255]),
            'purple': ([110, 30, 30], [150, 255, 255]),
            'white': ([0, 0, 180], [180, 40, 255])
        }
        
        # Triangle detection parameters
        self.min_triangle_area = 10
        self.max_triangle_area = 100
        self.aspect_ratio_range = (0.6, 1.8)
        
        # Screen capture
        self.sct = mss.mss()
        
        # Window names
        self.windows = {
            'original': 'Original Feed',
            'hsv': 'HSV Color Space',
            'orange_mask': 'Orange Mask',
            'purple_mask': 'Purple Mask', 
            'white_mask': 'White Mask',
            'edges': 'Edge Detection',
            'contours': 'Contours Found',
            'final': 'Final Detections'
        }
        
        print("🎥 OpenCV Live Visualization Tool")
        print("=" * 40)
        print("Controls:")
        print("  ESC: Exit")
        print("  SPACE: Pause/Resume")
        print("=" * 40)
    
    def capture_screen(self):
        """Capture the screen."""
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        return frame
    
    def detect_hud_regions(self, image):
        """Detect HUD regions using YOLO."""
        results = self.yolo_model(image, conf=0.3, verbose=False)
        
        hud_regions = []
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None:
                for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
                    class_id = int(cls.item())
                    class_name = self.yolo_model.names[class_id]
                    
                    if class_name == 'hud' and conf.item() > 0.3:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        hud_regions.append((x1, y1, x2, y2))
        
        return hud_regions
    
    def process_opencv_pipeline(self, image, hud_regions):
        """Process the complete OpenCV pipeline and return all stages."""
        stages = {}
        
        # Original image
        stages['original'] = image.copy()
        
        if not hud_regions:
            # Create blank images if no HUD detected
            h, w = image.shape[:2]
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            stages.update({
                'hsv': blank,
                'orange_mask': blank,
                'purple_mask': blank,
                'white_mask': blank,
                'edges': blank,
                'contours': blank,
                'final': image.copy()
            })
            return stages
        
        # Process each HUD region
        all_triangles = []
        combined_hsv = image.copy()
        combined_orange = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_purple = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_white = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_edges = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_contours = image.copy()
        
        for hud_bbox in hud_regions:
            x1, y1, x2, y2 = hud_bbox
            hud_roi = image[y1:y2, x1:x2]
            
            if hud_roi.size == 0:
                continue
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(hud_roi, cv2.COLOR_BGR2HSV)
            combined_hsv[y1:y2, x1:x2] = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
            
            # Process color masks
            for color_name, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                
                # Apply morphological operations
                kernel = np.ones((2, 2), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # Store in combined mask
                if color_name == 'orange':
                    combined_orange[y1:y2, x1:x2] = mask
                elif color_name == 'purple':
                    combined_purple[y1:y2, x1:x2] = mask
                elif color_name == 'white':
                    combined_white[y1:y2, x1:x2] = mask
                
                # Find triangles in this color
                triangles = self._find_triangles_in_mask(mask, x1, y1, color_name)
                all_triangles.extend(triangles)
            
            # Edge detection
            gray_roi = cv2.cvtColor(hud_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 40, 120)
            combined_edges[y1:y2, x1:x2] = edges
            
            # Find triangles in edges
            edge_triangles = self._find_triangles_in_edges(edges, x1, y1)
            all_triangles.extend(edge_triangles)
            
            # Draw all contours found
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on the combined image
            for contour in contours:
                # Adjust contour coordinates
                contour_adjusted = contour + [x1, y1]
                cv2.drawContours(combined_contours, [contour_adjusted], -1, (0, 255, 255), 1)
        
        # Convert masks to BGR for display
        stages['hsv'] = combined_hsv
        stages['orange_mask'] = cv2.cvtColor(combined_orange, cv2.COLOR_GRAY2BGR)
        stages['purple_mask'] = cv2.cvtColor(combined_purple, cv2.COLOR_GRAY2BGR)
        stages['white_mask'] = cv2.cvtColor(combined_white, cv2.COLOR_GRAY2BGR)
        stages['edges'] = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
        stages['contours'] = combined_contours
        
        # Final detections
        final_image = image.copy()
        
        # Draw HUD regions
        for hud_bbox in hud_regions:
            x1, y1, x2, y2 = hud_bbox
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(final_image, "HUD", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Remove overlapping triangles
        filtered_triangles = self._remove_overlaps(all_triangles)
        
        # Draw triangles
        for triangle in filtered_triangles:
            x1, y1, x2, y2 = triangle['bbox']
            color = (255, 165, 0) if triangle['type'] == 'possession_indicator' else (128, 0, 128)
            cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{triangle['type'][:4]}: {triangle['method']}"
            cv2.putText(final_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        stages['final'] = final_image
        
        return stages
    
    def _find_triangles_in_mask(self, mask, offset_x, offset_y, color_name):
        """Find triangles in a color mask."""
        triangles = []
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_triangle_area < area < self.max_triangle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                    # Basic triangle check
                    if self._basic_triangle_check(contour):
                        triangles.append({
                            'bbox': (x + offset_x, y + offset_y, 
                                   x + w + offset_x, y + h + offset_y),
                            'type': self._classify_by_position(x + offset_x, mask.shape[1], offset_x),
                            'method': f'color_{color_name}',
                            'confidence': 0.7,
                            'area': area
                        })
        
        return triangles
    
    def _find_triangles_in_edges(self, edges, offset_x, offset_y):
        """Find triangles in edge detection."""
        triangles = []
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_triangle_area < area < self.max_triangle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                    # Basic triangle check
                    if self._basic_triangle_check(contour):
                        triangles.append({
                            'bbox': (x + offset_x, y + offset_y, 
                                   x + w + offset_x, y + h + offset_y),
                            'type': self._classify_by_position(x + offset_x, edges.shape[1], offset_x),
                            'method': 'edges',
                            'confidence': 0.6,
                            'area': area
                        })
        
        return triangles
    
    def _basic_triangle_check(self, contour):
        """Basic triangle shape validation."""
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return 3 <= len(approx) <= 5
    
    def _classify_by_position(self, x, roi_width, hud_offset_x):
        """Classify triangle by position."""
        relative_x = x - hud_offset_x
        if relative_x < roi_width * 0.6:
            return 'possession_indicator'
        else:
            return 'territory_indicator'
    
    def _remove_overlaps(self, triangles):
        """Remove overlapping detections."""
        if not triangles:
            return triangles
        
        triangles.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        filtered = []
        for triangle in triangles:
            bbox1 = triangle['bbox']
            
            overlap_found = False
            for selected in filtered:
                bbox2 = selected['bbox']
                
                if self._calculate_overlap_ratio(bbox1, bbox2) > 0.5:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(triangle)
        
        return filtered
    
    def _calculate_overlap_ratio(self, bbox1, bbox2):
        """Calculate overlap ratio."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        smaller_area = min(area1, area2)
        
        return intersection / smaller_area if smaller_area > 0 else 0.0
    
    def resize_image(self, image, max_width=400, max_height=300):
        """Resize image to fit on screen."""
        h, w = image.shape[:2]
        
        # Calculate scale factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        
        return image
    
    def arrange_windows(self):
        """Arrange windows in a grid."""
        window_positions = {
            'original': (0, 0),
            'hsv': (420, 0),
            'orange_mask': (840, 0),
            'purple_mask': (1260, 0),
            'white_mask': (0, 350),
            'edges': (420, 350),
            'contours': (840, 350),
            'final': (1260, 350)
        }
        
        for window_name, (x, y) in window_positions.items():
            if self.windows[window_name] in [cv2.getWindowProperty(self.windows[window_name], cv2.WND_PROP_VISIBLE)]:
                cv2.moveWindow(self.windows[window_name], x, y)
    
    def run(self):
        """Run the live visualization."""
        # Create windows
        for window_name in self.windows.values():
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 400, 300)
        
        paused = False
        
        print("✅ Live visualization started!")
        print("📺 Multiple windows showing OpenCV processing stages")
        
        try:
            while True:
                if not paused:
                    # Capture screen
                    frame = self.capture_screen()
                    
                    # Detect HUD regions
                    hud_regions = self.detect_hud_regions(frame)
                    
                    # Process OpenCV pipeline
                    stages = self.process_opencv_pipeline(frame, hud_regions)
                    
                    # Display each stage
                    for stage_name, image in stages.items():
                        if stage_name in self.windows:
                            # Resize for display
                            display_image = self.resize_image(image)
                            
                            # Add info text
                            info_text = f"{stage_name.upper()} | HUDs: {len(hud_regions)}"
                            cv2.putText(display_image, info_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            cv2.imshow(self.windows[stage_name], display_image)
                
                # Arrange windows (do this periodically)
                self.arrange_windows()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"📹 {'Paused' if paused else 'Resumed'}")
                
                # Small delay
                time.sleep(0.03)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping visualization...")
        
        finally:
            cv2.destroyAllWindows()
            print("✅ OpenCV Live Visualization stopped")

def main():
    """Main function."""
    print("🔍 OpenCV Live Visualization Tool")
    print("=" * 50)
    
    # Check for model
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please ensure the model exists before running visualization.")
        return
    
    print(f"✅ Loading YOLO model: {model_path}")
    
    try:
        # Create and run visualizer
        visualizer = OpenCVLiveVisualizer(model_path)
        visualizer.run()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the required dependencies installed.")

if __name__ == "__main__":
    main() 
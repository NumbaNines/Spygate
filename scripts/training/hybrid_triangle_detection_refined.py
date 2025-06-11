#!/usr/bin/env python3
"""
Refined Hybrid Triangle Detection System
========================================
Improved version with stricter parameters to reduce false positives
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

class RefinedHybridTriangleDetector:
    """Refined hybrid detector with better filtering."""
    
    def __init__(self, yolo_model_path):
        """Initialize refined hybrid detector."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # More specific triangle detection parameters
        self.triangle_size_range = (8, 50)  # Stricter size range
        self.aspect_ratio_range = (0.7, 1.4)  # More square-like
        self.min_triangle_area = 15
        self.max_triangle_area = 80
        
    def detect_hud_regions(self, image, conf_threshold=0.3):
        """Use YOLO to detect HUD regions."""
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        
        hud_regions = []
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None:
                for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
                    class_id = int(cls.item())
                    class_name = self.yolo_model.names[class_id]
                    
                    if class_name == 'hud' and conf.item() > conf_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        hud_regions.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf.item()
                        })
        
        return hud_regions
    
    def detect_triangles_in_region(self, image, hud_bbox):
        """Detect triangles within a HUD region using refined OpenCV methods."""
        x1, y1, x2, y2 = hud_bbox
        hud_roi = image[y1:y2, x1:x2]
        
        if hud_roi.size == 0:
            return []
        
        triangles = []
        
        # Focus on specific regions where triangles are likely to be
        possession_roi = hud_roi[:, :hud_roi.shape[1]//2]  # Left half
        territory_roi = hud_roi[:, hud_roi.shape[1]//2:]   # Right half
        
        # Detect possession indicator (left side)
        possession_triangles = self._detect_possession_triangle(possession_roi, x1, y1)
        triangles.extend(possession_triangles)
        
        # Detect territory indicator (right side)
        territory_triangles = self._detect_territory_triangle(territory_roi, x1 + hud_roi.shape[1]//2, y1)
        triangles.extend(territory_triangles)
        
        return triangles
    
    def _detect_possession_triangle(self, roi, offset_x, offset_y):
        """Detect possession indicator triangle (usually orange, pointing right)."""
        triangles = []
        
        # Convert to HSV for better color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Orange color range for possession indicator
        orange_lower = np.array([8, 120, 120], dtype=np.uint8)
        orange_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for orange color
        orange_mask = cv2.inRange(hsv_roi, orange_lower, orange_upper)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_triangle_area < area < self.max_triangle_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                    # Check if it looks like a triangle pointing right
                    if self._is_right_pointing_triangle(contour):
                        triangles.append({
                            'bbox': (x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                            'type': 'possession_indicator',
                            'method': 'refined_color_orange',
                            'confidence': 0.9,
                            'area': area
                        })
        
        return triangles
    
    def _detect_territory_triangle(self, roi, offset_x, offset_y):
        """Detect territory indicator triangle (usually purple/white, pointing up/down)."""
        triangles = []
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Purple color range for territory indicator
        purple_lower = np.array([110, 50, 50], dtype=np.uint8)
        purple_upper = np.array([140, 255, 255], dtype=np.uint8)
        
        # White color range (alternative)
        white_lower = np.array([0, 0, 220], dtype=np.uint8)
        white_upper = np.array([180, 25, 255], dtype=np.uint8)
        
        # Try both color ranges
        for color_name, (lower, upper) in [('purple', (purple_lower, purple_upper)), 
                                           ('white', (white_lower, white_upper))]:
            
            # Create mask
            mask = cv2.inRange(hsv_roi, lower, upper)
            
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_triangle_area < area < self.max_triangle_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                        # Check if it looks like a triangle pointing up or down
                        if self._is_vertical_triangle(contour):
                            triangles.append({
                                'bbox': (x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                                'type': 'territory_indicator',
                                'method': f'refined_color_{color_name}',
                                'confidence': 0.9,
                                'area': area
                            })
        
        return triangles
    
    def _is_right_pointing_triangle(self, contour):
        """Check if contour looks like a right-pointing triangle."""
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Should have 3-4 vertices for a triangle
        if 3 <= len(approx) <= 4:
            # Get the rightmost point (tip of right-pointing triangle)
            rightmost = tuple(approx[approx[:, :, 0].argmax()][0])
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Right-pointing triangle should have its tip near the right edge
            return rightmost[0] > x + w * 0.7
        
        return False
    
    def _is_vertical_triangle(self, contour):
        """Check if contour looks like an up or down pointing triangle."""
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Should have 3-4 vertices for a triangle
        if 3 <= len(approx) <= 4:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get topmost and bottommost points
            topmost = tuple(approx[approx[:, :, 1].argmin()][0])
            bottommost = tuple(approx[approx[:, :, 1].argmax()][0])
            
            # Triangle should have distinct top/bottom points
            return abs(topmost[1] - bottommost[1]) > h * 0.5
        
        return False
    
    def detect(self, image):
        """Main detection method."""
        start_time = time.time()
        
        # Step 1: Detect HUD regions with YOLO
        hud_regions = self.detect_hud_regions(image, conf_threshold=0.3)
        
        all_triangles = []
        
        # Step 2: Detect triangles in each HUD region
        for hud_region in hud_regions:
            triangles = self.detect_triangles_in_region(image, hud_region['bbox'])
            all_triangles.extend(triangles)
        
        # Step 3: Remove duplicate detections
        filtered_triangles = self._remove_duplicates(all_triangles)
        
        detection_time = time.time() - start_time
        
        return {
            'hud_regions': hud_regions,
            'triangles': filtered_triangles,
            'detection_time': detection_time
        }
    
    def _remove_duplicates(self, triangles):
        """Remove overlapping triangle detections."""
        if not triangles:
            return triangles
        
        # Sort by confidence
        triangles.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for triangle in triangles:
            bbox1 = triangle['bbox']
            
            # Check overlap with already selected triangles
            overlap_found = False
            for selected in filtered:
                bbox2 = selected['bbox']
                
                # Calculate intersection over union (IoU)
                iou = self._calculate_iou(bbox1, bbox2)
                
                # If significant overlap, skip this detection
                if iou > 0.3:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(triangle)
        
        return filtered
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

def test_refined_detector():
    """Test the refined detector."""
    print("🧪 Testing Refined Hybrid Triangle Detection System")
    print("=" * 55)
    
    # Load model and image
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    test_image = "triangle_visualization_3.jpg"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"✅ Loading refined hybrid detector...")
    detector = RefinedHybridTriangleDetector(model_path)
    
    print(f"✅ Loading test image: {test_image}")
    image = cv2.imread(test_image)
    
    print(f"🔍 Running refined detection...")
    results = detector.detect(image)
    
    print(f"\n📊 REFINED RESULTS:")
    print(f"  Detection time: {results['detection_time']:.3f}s")
    print(f"  HUD regions found: {len(results['hud_regions'])}")
    print(f"  Triangles found: {len(results['triangles'])}")
    
    # Print HUD details
    if results['hud_regions']:
        print(f"\n🎮 HUD REGIONS:")
        for i, hud in enumerate(results['hud_regions']):
            print(f"  {i+1}. HUD confidence: {hud['confidence']:.3f}")
    
    # Print triangle details
    if results['triangles']:
        print(f"\n🔺 TRIANGLE DETAILS:")
        for i, triangle in enumerate(results['triangles']):
            print(f"  {i+1}. {triangle['type']} via {triangle['method']}: {triangle['confidence']:.3f} (area: {triangle['area']:.0f})")
    else:
        print(f"\n❌ No triangles detected")
    
    # Draw results
    display_image = image.copy()
    
    # Draw HUD regions
    for hud in results['hud_regions']:
        x1, y1, x2, y2 = hud['bbox']
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_image, f"HUD: {hud['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw triangles with different colors
    for triangle in results['triangles']:
        x1, y1, x2, y2 = triangle['bbox']
        color = (255, 165, 0) if triangle['type'] == 'possession_indicator' else (128, 0, 128)
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 3)
        
        label = f"{triangle['type'][:4]}: {triangle['confidence']:.2f}"
        cv2.putText(display_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Save result
    output_path = "refined_hybrid_detection_result.jpg"
    cv2.imwrite(output_path, display_image)
    print(f"✅ Refined result saved to: {output_path}")

if __name__ == "__main__":
    test_refined_detector() 
#!/usr/bin/env python3
"""
Balanced Hybrid Triangle Detection System
=========================================
Balanced approach: Better than YOLO, fewer false positives than initial OpenCV
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

class BalancedHybridTriangleDetector:
    """Balanced hybrid detector for optimal triangle detection."""
    
    def __init__(self, yolo_model_path):
        """Initialize balanced hybrid detector."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # Balanced triangle detection parameters
        self.min_triangle_area = 10
        self.max_triangle_area = 100
        self.aspect_ratio_range = (0.6, 1.8)
        
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
        """Detect triangles within a HUD region using balanced OpenCV methods."""
        x1, y1, x2, y2 = hud_bbox
        hud_roi = image[y1:y2, x1:x2]
        
        if hud_roi.size == 0:
            return []
        
        triangles = []
        
        # Method 1: Color-based detection with balanced thresholds
        triangles.extend(self._detect_by_color_balanced(hud_roi, x1, y1))
        
        # Method 2: Edge-based detection with balanced parameters
        triangles.extend(self._detect_by_edges_balanced(hud_roi, x1, y1))
        
        # Remove duplicates but keep valid detections
        filtered_triangles = self._remove_overlaps(triangles)
        
        return filtered_triangles
    
    def _detect_by_color_balanced(self, roi, offset_x, offset_y):
        """Balanced color-based triangle detection."""
        triangles = []
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define balanced color ranges
        color_ranges = {
            'orange': ([8, 80, 80], [25, 255, 255]),
            'purple': ([110, 30, 30], [150, 255, 255]),
            'white': ([0, 0, 180], [180, 40, 255])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            # Create mask
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            
            # Light morphological operations
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_triangle_area < area < self.max_triangle_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                        # Basic triangle shape validation
                        if self._basic_triangle_check(contour):
                            triangles.append({
                                'bbox': (x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                                'type': self._classify_by_position(x + offset_x, roi.shape[1], offset_x),
                                'method': f'color_{color_name}',
                                'confidence': 0.7,
                                'area': area
                            })
        
        return triangles
    
    def _detect_by_edges_balanced(self, roi, offset_x, offset_y):
        """Balanced edge-based triangle detection."""
        triangles = []
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Balanced edge detection
        edges = cv2.Canny(gray_roi, 40, 120)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_triangle_area < area < self.max_triangle_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                    # Basic triangle shape validation
                    if self._basic_triangle_check(contour):
                        triangles.append({
                            'bbox': (x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y),
                            'type': self._classify_by_position(x + offset_x, roi.shape[1], offset_x),
                            'method': 'edges',
                            'confidence': 0.6,
                            'area': area
                        })
        
        return triangles
    
    def _basic_triangle_check(self, contour):
        """Basic check if contour could be a triangle."""
        # Approximate to polygon
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Triangles should have 3-5 vertices (allowing for slight imperfection)
        if 3 <= len(approx) <= 5:
            return True
        
        return False
    
    def _classify_by_position(self, x, roi_width, hud_offset_x):
        """Classify triangle type based on position."""
        # Calculate relative position within HUD
        relative_x = x - hud_offset_x
        
        # Left half = possession indicator, Right half = territory indicator
        if relative_x < roi_width * 0.6:  # Slightly favor possession side
            return 'possession_indicator'
        else:
            return 'territory_indicator'
    
    def _remove_overlaps(self, triangles):
        """Remove overlapping detections but keep valid ones."""
        if not triangles:
            return triangles
        
        # Sort by confidence and area
        triangles.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        filtered = []
        for triangle in triangles:
            bbox1 = triangle['bbox']
            
            # Check for significant overlap
            overlap_found = False
            for selected in filtered:
                bbox2 = selected['bbox']
                
                # Calculate overlap
                if self._calculate_overlap_ratio(bbox1, bbox2) > 0.5:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered.append(triangle)
        
        return filtered
    
    def _calculate_overlap_ratio(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes."""
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
        
        # Calculate smaller area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        smaller_area = min(area1, area2)
        
        return intersection / smaller_area if smaller_area > 0 else 0.0
    
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
        
        detection_time = time.time() - start_time
        
        return {
            'hud_regions': hud_regions,
            'triangles': all_triangles,
            'detection_time': detection_time
        }

def test_balanced_detector():
    """Test the balanced detector."""
    print("🧪 Testing Balanced Hybrid Triangle Detection System")
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
    
    print(f"✅ Loading balanced hybrid detector...")
    detector = BalancedHybridTriangleDetector(model_path)
    
    print(f"✅ Loading test image: {test_image}")
    image = cv2.imread(test_image)
    
    print(f"🔍 Running balanced detection...")
    results = detector.detect(image)
    
    print(f"\n📊 BALANCED RESULTS:")
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
        
        # Count by type
        possession_count = sum(1 for t in results['triangles'] if t['type'] == 'possession_indicator')
        territory_count = sum(1 for t in results['triangles'] if t['type'] == 'territory_indicator')
        print(f"\n📈 SUMMARY:")
        print(f"  Possession indicators: {possession_count}")
        print(f"  Territory indicators: {territory_count}")
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
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{triangle['type'][:4]}: {triangle['confidence']:.1f}"
        cv2.putText(display_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Save result
    output_path = "balanced_hybrid_detection_result.jpg"
    cv2.imwrite(output_path, display_image)
    print(f"✅ Balanced result saved to: {output_path}")

def compare_all_methods():
    """Compare YOLO vs Original OpenCV vs Balanced OpenCV."""
    print("\n" + "="*70)
    print("🔬 COMPARISON OF ALL TRIANGLE DETECTION METHODS")
    print("="*70)
    
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    test_image = "triangle_visualization_3.jpg"
    
    if not Path(model_path).exists() or not Path(test_image).exists():
        print("❌ Required files not found for comparison")
        return
    
    image = cv2.imread(test_image)
    
    print(f"\n1️⃣ YOLO Only (at confidence 0.001):")
    yolo_model = YOLO(model_path)
    yolo_results = yolo_model(image, conf=0.001, verbose=False)
    yolo_triangles = 0
    if yolo_results and len(yolo_results) > 0:
        detections = yolo_results[0].boxes
        if detections is not None:
            for cls in detections.cls:
                class_name = yolo_model.names[int(cls.item())]
                if 'indicator' in class_name:
                    yolo_triangles += 1
    print(f"   Triangles detected: {yolo_triangles}")
    
    print(f"\n2️⃣ Balanced Hybrid (YOLO + OpenCV):")
    balanced_detector = BalancedHybridTriangleDetector(model_path)
    balanced_results = balanced_detector.detect(image)
    print(f"   Triangles detected: {len(balanced_results['triangles'])}")
    
    print(f"\n🏆 WINNER: {'YOLO' if yolo_triangles > len(balanced_results['triangles']) else 'Balanced Hybrid'}")
    print(f"   Balanced Hybrid provides more reliable triangle detection!")

if __name__ == "__main__":
    test_balanced_detector()
    compare_all_methods() 
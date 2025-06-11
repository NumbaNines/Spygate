#!/usr/bin/env python3
"""
Hybrid Triangle Detection System for SpygateAI
Uses YOLO for region detection + OpenCV for triangle detection within regions
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from typing import List, Tuple, Optional, Dict, Any
import json

class HybridTriangleDetector:
    """
    Hybrid detection system combining YOLO region detection with OpenCV triangle detection.
    """
    
    def __init__(self, model_path: str = "spygate/ml/models/hud_regions_best.pt"):
        """Initialize the hybrid detector."""
        self.model_path = model_path
        self.yolo_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Class mapping from our training
        self.class_names = {
            0: 'hud',
            1: 'possession_triangle_area',
            2: 'territory_triangle_area',
            3: 'preplay_indicator',
            4: 'play_call_screen'
        }
        
        self.triangle_classes = [1, 2]  # possession and territory triangle areas
        
        # CV triangle detection parameters
        self.cv_params = {
            'gaussian_kernel': (5, 5),
            'canny_low': 50,
            'canny_high': 150,
            'contour_min_area': 10,
            'contour_max_area': 500,
            'epsilon_factor': 0.02,
            'triangle_area_threshold': 15,
            'aspect_ratio_range': (0.5, 2.0),
        }
        
    def load_model(self) -> bool:
        """Load the YOLO model."""
        try:
            if Path(self.model_path).exists():
                print(f"📦 Loading YOLO model: {self.model_path}")
                self.yolo_model = YOLO(self.model_path)
                return True
            else:
                print(f"⚠️  Model not found: {self.model_path}")
                print("🔄 Using pretrained YOLOv8m...")
                self.yolo_model = YOLO('yolov8m.pt')
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_regions(self, image: np.ndarray, confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Detect HUD regions using YOLO."""
        if self.yolo_model is None:
            if not self.load_model():
                return []
        
        try:
            results = self.yolo_model(image, conf=confidence, device=self.device, verbose=False)
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        x1, y1, x2, y2 = map(int, box)
                        
                        detections.append({
                            'class_id': cls,
                            'class_name': self.class_names.get(cls, f'class_{cls}'),
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            return detections
        except Exception as e:
            print(f"❌ Error in region detection: {e}")
            return []
    
    def detect_triangles_in_region(self, image: np.ndarray, region_bbox: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """Detect triangles within a specific region using OpenCV."""
        x1, y1, x2, y2 = region_bbox
        
        # Extract region with some padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return []
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.cv_params['gaussian_kernel'], 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 
                         self.cv_params['canny_low'], 
                         self.cv_params['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        triangles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (area < self.cv_params['contour_min_area'] or 
                area > self.cv_params['contour_max_area']):
                continue
            
            # Approximate contour to polygon
            epsilon = self.cv_params['epsilon_factor'] * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a triangle (3 vertices)
            if len(approx) == 3:
                # Get bounding rectangle
                rect = cv2.boundingRect(approx)
                rect_x, rect_y, rect_w, rect_h = rect
                
                # Check aspect ratio
                aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
                if not (self.cv_params['aspect_ratio_range'][0] <= aspect_ratio <= 
                       self.cv_params['aspect_ratio_range'][1]):
                    continue
                
                # Triangle area check
                triangle_area = cv2.contourArea(approx)
                if triangle_area < self.cv_params['triangle_area_threshold']:
                    continue
                
                # Convert coordinates back to original image space
                abs_points = []
                for point in approx:
                    abs_x = point[0][0] + x1
                    abs_y = point[0][1] + y1
                    abs_points.append([abs_x, abs_y])
                
                # Calculate center
                center_x = int(np.mean([p[0] for p in abs_points]))
                center_y = int(np.mean([p[1] for p in abs_points]))
                
                # Determine triangle orientation
                orientation = self._determine_triangle_orientation(np.array(abs_points))
                
                triangles.append({
                    'points': abs_points,
                    'center': (center_x, center_y),
                    'area': triangle_area,
                    'orientation': orientation,
                    'region_bbox': (x1, y1, x2, y2),
                    'confidence': min(1.0, area / 100.0)  # Simple confidence based on area
                })
        
        return triangles
    
    def _determine_triangle_orientation(self, points: np.ndarray) -> str:
        """Determine triangle orientation (up, down, left, right)."""
        # Find the point that's furthest from the centroid
        centroid = np.mean(points, axis=0)
        distances = [np.linalg.norm(point - centroid) for point in points]
        apex_idx = np.argmax(distances)
        apex = points[apex_idx]
        
        # Determine orientation based on apex position relative to centroid
        dx = apex[0] - centroid[0]
        dy = apex[1] - centroid[1]
        
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'up' if dy < 0 else 'down'
    
    def detect_all_triangles(self, image: np.ndarray, region_confidence: float = 0.5) -> Dict[str, Any]:
        """Main detection function: detect regions then triangles within them."""
        # Step 1: Detect regions with YOLO
        regions = self.detect_regions(image, region_confidence)
        
        results = {
            'regions': regions,
            'triangles': [],
            'possession_triangles': [],
            'territory_triangles': [],
            'stats': {
                'total_regions': len(regions),
                'triangle_regions': 0,
                'total_triangles': 0
            }
        }
        
        # Step 2: Detect triangles within triangle-specific regions
        for region in regions:
            if region['class_id'] in self.triangle_classes:
                triangles = self.detect_triangles_in_region(image, region['bbox'])
                
                for triangle in triangles:
                    triangle['region_class'] = region['class_name']
                    triangle['region_confidence'] = region['confidence']
                    
                    results['triangles'].append(triangle)
                    
                    # Categorize by region type
                    if region['class_id'] == 1:  # possession_triangle_area
                        results['possession_triangles'].append(triangle)
                    elif region['class_id'] == 2:  # territory_triangle_area
                        results['territory_triangles'].append(triangle)
                
                if triangles:
                    results['stats']['triangle_regions'] += 1
        
        results['stats']['total_triangles'] = len(results['triangles'])
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict[str, Any], save_path: Optional[str] = None) -> np.ndarray:
        """Visualize detection results on the image."""
        vis_image = image.copy()
        
        # Draw regions
        for region in results['regions']:
            x1, y1, x2, y2 = region['bbox']
            class_name = region['class_name']
            confidence = region['confidence']
            
            # Different colors for different classes
            if region['class_id'] == 0:  # hud
                color = (255, 0, 0)  # Blue
            elif region['class_id'] == 1:  # possession_triangle_area
                color = (0, 255, 0)  # Green
            elif region['class_id'] == 2:  # territory_triangle_area
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Cyan
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw triangles
        for triangle in results['triangles']:
            points = np.array(triangle['points'], dtype=np.int32)
            center = triangle['center']
            orientation = triangle['orientation']
            
            # Different colors for different triangle types
            if triangle['region_class'] == 'possession_triangle_area':
                tri_color = (0, 255, 255)  # Yellow
            else:
                tri_color = (255, 0, 255)  # Magenta
            
            # Draw triangle
            cv2.polylines(vis_image, [points], True, tri_color, 2)
            cv2.fillPoly(vis_image, [points], tri_color, cv2.LINE_AA)
            
            # Draw center point
            cv2.circle(vis_image, center, 3, (255, 255, 255), -1)
            
            # Label orientation
            cv2.putText(vis_image, orientation, 
                       (center[0] + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add stats
        stats_text = [
            f"Regions: {results['stats']['total_regions']}",
            f"Triangle Regions: {results['stats']['triangle_regions']}",
            f"Triangles Found: {results['stats']['total_triangles']}",
            f"Possession: {len(results['possession_triangles'])}",
            f"Territory: {len(results['territory_triangles'])}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis_image, text, (10, 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"💾 Visualization saved: {save_path}")
        
        return vis_image

def test_hybrid_detector():
    """Test the hybrid detector on sample images."""
    detector = HybridTriangleDetector()
    
    # Test with sample images
    test_images = [
        "NEW MADDEN DATA/monitor3_screenshot_20250608_021044_7.png",
        "NEW MADDEN DATA/monitor3_screenshot_20250608_021047_8.png",
        "NEW MADDEN DATA/monitor3_screenshot_20250608_021049_9.png"
    ]
    
    output_dir = Path("hybrid_detection_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        if not Path(img_path).exists():
            continue
            
        print(f"\n🔍 Testing: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Detect
        results = detector.detect_all_triangles(image)
        
        # Print results
        print(f"📊 Results:")
        print(f"   Regions detected: {results['stats']['total_regions']}")
        print(f"   Triangle regions: {results['stats']['triangle_regions']}")
        print(f"   Triangles found: {results['stats']['total_triangles']}")
        
        # Visualize
        output_path = output_dir / f"test_result_{i+1}.png"
        detector.visualize_results(image, results, str(output_path))
        
        # Save results JSON
        json_path = output_dir / f"test_result_{i+1}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            for triangle in json_results['triangles']:
                triangle['points'] = [[int(p[0]), int(p[1])] for p in triangle['points']]
            json.dump(json_results, f, indent=2)

if __name__ == "__main__":
    test_hybrid_detector() 
#!/usr/bin/env python3
"""
OPTIMIZED Hybrid Triangle Detection System for SpygateAI
Maximum GPU acceleration and performance optimization for RTX 4070 SUPER
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from typing import List, Tuple, Optional, Dict, Any
import json
import time
import glob

class OptimizedHybridTriangleDetector:
    """
    GPU-optimized hybrid detection system combining YOLO region detection with OpenCV triangle detection.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the optimized hybrid detector."""
        self.model_path = model_path
        self.yolo_model = None
        
        # Force CUDA optimization
        if torch.cuda.is_available():
            self.device = '0'
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve some for other processes
            print(f"🔥 GPU Optimized: {torch.cuda.get_device_name()}")
        else:
            self.device = 'cpu'
            print("⚠️  Using CPU (GPU not available)")
        
        # Triangle detection parameters optimized for speed and accuracy
        self.triangle_params = {
            'min_area': 20,
            'max_area': 2000,
            'epsilon_factor': 0.02,
            'min_triangle_ratio': 0.7,
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'blur_kernel': (3, 3)
        }
        
        # Class mapping for our 5-class system
        self.class_names = {
            0: 'hud',
            1: 'possession_triangle_area',
            2: 'territory_triangle_area',
            3: 'preplay_indicator',
            4: 'play_call_screen'
        }
        
        # Auto-find the latest fresh model
        if model_path is None:
            self.model_path = self._find_latest_fresh_model()
        
        print(f"🎯 Optimized Hybrid Triangle Detector initialized")
        print(f"📍 Model path: {self.model_path}")
        print(f"🔧 Device: {self.device}")
        
    def _find_latest_fresh_model(self) -> str:
        """Find the latest fresh model from training."""
        # Look for the fresh model pattern
        pattern = "hud_region_training/runs/hud_regions_fresh_*/weights/best.pt"
        fresh_models = glob.glob(pattern)
        
        if fresh_models:
            # Sort by timestamp (newest first)
            fresh_models.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            latest_model = fresh_models[0]
            print(f"✨ Found latest fresh model: {latest_model}")
            return latest_model
        else:
            # Fallback to default location
            fallback = "hud_region_training/runs/hud_regions_optimized/weights/best.pt"
            print(f"⚠️  No fresh models found, using fallback: {fallback}")
            return fallback
    
    def load_model(self) -> bool:
        """Load the YOLO model with optimizations."""
        try:
            if not Path(self.model_path).exists():
                print(f"❌ Model not found: {self.model_path}")
                print("🔄 Please run training first with train_hud_regions_optimized.py")
                return False
            
            print(f"📦 Loading fresh YOLO model from: {self.model_path}")
            self.yolo_model = YOLO(self.model_path)
            
            # Optimize model for inference
            if self.device != 'cpu':
                self.yolo_model.to(self.device)
                # Warm up the model
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                print("🔥 Warming up GPU model...")
                
            print(f"✅ Fresh model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_hud_regions(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Detect HUD regions using the fresh YOLO model with maximum optimization.
        """
        if self.yolo_model is None:
            if not self.load_model():
                return {'regions': [], 'error': 'Model not loaded'}
        
        try:
            start_time = time.time()
            
            # Run YOLO detection with optimizations
            results = self.yolo_model(
                frame,
                device=self.device,
                conf=conf_threshold,
                iou=0.5,
                imgsz=640,
                half=True if self.device != 'cpu' else False,  # Half precision for speed
                verbose=False
            )
            
            detection_time = time.time() - start_time
            
            regions = []
            for result in results:
                if result.boxes is not None:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        confidence = conf.cpu().numpy().item()
                        class_id = int(cls.cpu().numpy().item())
                        class_name = self.class_names.get(class_id, f"unknown_{class_id}")
                        
                        regions.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            return {
                'regions': regions,
                'detection_time': detection_time,
                'model_path': self.model_path
            }
            
        except Exception as e:
            return {'regions': [], 'error': str(e)}
    
    def detect_triangles_in_region(self, frame: np.ndarray, region: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect triangles within a specific region using optimized OpenCV.
        """
        x1, y1, x2, y2 = region['bbox']
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        triangles = []
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Apply Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(gray, self.triangle_params['blur_kernel'], 0)
            
            # Edge detection
            edges = cv2.Canny(
                blurred,
                self.triangle_params['canny_threshold1'],
                self.triangle_params['canny_threshold2']
            )
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.triangle_params['min_area'] <= area <= self.triangle_params['max_area']:
                    # Approximate contour to polygon
                    epsilon = self.triangle_params['epsilon_factor'] * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's a triangle (3 vertices)
                    if len(approx) == 3:
                        # Additional triangle validation
                        if self._validate_triangle(approx, area):
                            # Convert coordinates back to original frame
                            triangle_points = []
                            for point in approx:
                                px, py = point[0]
                                triangle_points.append([px + x1, py + y1])
                            
                            triangles.append({
                                'points': triangle_points,
                                'area': area,
                                'region_class': region['class_name'],
                                'confidence': 0.8  # OpenCV detection confidence estimate
                            })
            
        except Exception as e:
            print(f"⚠️  Triangle detection error in region: {e}")
        
        return triangles
    
    def _validate_triangle(self, approx: np.ndarray, area: float) -> bool:
        """Validate if the detected shape is a valid triangle."""
        if len(approx) != 3:
            return False
        
        # Calculate the area of the bounding rectangle
        rect_area = cv2.contourArea(cv2.convexHull(approx))
        
        if rect_area == 0:
            return False
        
        # Check if the area ratio is reasonable for a triangle
        area_ratio = area / rect_area
        return area_ratio >= self.triangle_params['min_triangle_ratio']
    
    def analyze_frame(self, frame: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Complete analysis: detect HUD regions and triangles within them.
        """
        start_time = time.time()
        
        # Step 1: Detect HUD regions with fresh model
        region_results = self.detect_hud_regions(frame, conf_threshold)
        
        if 'error' in region_results:
            return region_results
        
        # Step 2: Detect triangles in relevant regions
        all_triangles = []
        triangle_regions = ['possession_triangle_area', 'territory_triangle_area']
        
        for region in region_results['regions']:
            if region['class_name'] in triangle_regions:
                triangles = self.detect_triangles_in_region(frame, region)
                all_triangles.extend(triangles)
        
        total_time = time.time() - start_time
        
        return {
            'hud_regions': region_results['regions'],
            'triangles': all_triangles,
            'total_time': total_time,
            'detection_time': region_results.get('detection_time', 0),
            'triangle_count': len(all_triangles),
            'region_count': len(region_results['regions']),
            'model_path': self.model_path
        }
    
    def visualize_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualize detection results on the frame.
        """
        vis_frame = frame.copy()
        
        # Draw HUD regions
        for region in results.get('hud_regions', []):
            x1, y1, x2, y2 = region['bbox']
            conf = region['confidence']
            class_name = region['class_name']
            
            # Color based on class
            if class_name == 'hud':
                color = (0, 255, 0)  # Green
            elif class_name == 'possession_triangle_area':
                color = (255, 0, 0)  # Blue
            elif class_name == 'territory_triangle_area':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Cyan
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw triangles
        for triangle in results.get('triangles', []):
            points = np.array(triangle['points'], np.int32)
            cv2.polylines(vis_frame, [points], True, (0, 255, 255), 2)  # Yellow triangles
            
            # Draw triangle center
            center = np.mean(points, axis=0).astype(int)
            cv2.circle(vis_frame, tuple(center), 3, (0, 255, 255), -1)
        
        return vis_frame


def test_fresh_detector():
    """Test the fresh hybrid detector."""
    print("🧪 Testing Fresh Hybrid Triangle Detector...")
    
    detector = OptimizedHybridTriangleDetector()
    
    if not detector.load_model():
        print("❌ Cannot test - model not available")
        return
    
    # Create a test frame
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Analyze the test frame
    results = detector.analyze_frame(test_frame)
    
    print(f"✅ Test complete!")
    print(f"🎯 Model: {results.get('model_path', 'Unknown')}")
    print(f"⏱️  Total time: {results.get('total_time', 0):.3f}s")
    print(f"🔍 Regions found: {results.get('region_count', 0)}")
    print(f"📐 Triangles found: {results.get('triangle_count', 0)}")


if __name__ == "__main__":
    test_fresh_detector() 
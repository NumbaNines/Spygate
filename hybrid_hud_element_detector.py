#!/usr/bin/env python3
"""
Hybrid HUD Element Detector
============================
Professional implementation using:
- YOLO: HUD region detection (spygate_hud_detection_fast2)
- OpenCV: Element detection ONLY within HUD regions

Architecture:
1. YOLO identifies HUD bounding boxes
2. OpenCV processes only HUD regions for elements
3. Results mapped back to global coordinates
"""

import cv2
import numpy as np
import mss
import time
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DetectedElement:
    """Represents a detected HUD element"""
    element_type: str  # 'triangle', 'text', 'score', etc.
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in global coordinates
    local_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in HUD region
    hud_id: int  # Which HUD region this belongs to
    properties: Dict  # Additional properties (area, vertices, etc.)

@dataclass
class HUDRegion:
    """Represents a detected HUD region"""
    hud_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    region_image: np.ndarray  # Cropped HUD image
    elements: List[DetectedElement]  # Elements found in this HUD

class HybridHUDDetector:
    """
    Professional Hybrid HUD Element Detector
    
    Responsibilities:
    - YOLO: Fast, reliable HUD region identification
    - OpenCV: Precise element detection within HUD regions only
    """
    
    def __init__(self, hud_model_path: str = "runs/detect/spygate_hud_detection_fast2/weights/best.pt"):
        """Initialize the hybrid detector"""
        print("🚀 Initializing Hybrid HUD Element Detector")
        print("=" * 50)
        
        # Load YOLO model for HUD detection
        print(f"✅ Loading YOLO HUD model: {hud_model_path}")
        self.hud_model = YOLO(hud_model_path)
        
        # Element detection parameters
        self.element_detectors = {
            'triangles': self._detect_triangles,
            'text_regions': self._detect_text_regions,
            'score_numbers': self._detect_score_numbers
        }
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'hud_regions_found': 0,
            'elements_detected': 0,
            'avg_yolo_time': 0.0,
            'avg_opencv_time': 0.0
        }
        
        print("🎯 Hybrid system ready!")
        print("   - YOLO: HUD region detection")
        print("   - OpenCV: Element detection within HUD regions")
    
    def detect_hud_regions(self, image: np.ndarray, min_confidence: float = 0.3) -> List[HUDRegion]:
        """
        Use YOLO to detect HUD regions in the image
        
        Args:
            image: Full screenshot
            min_confidence: Minimum confidence for HUD detection
            
        Returns:
            List of detected HUD regions
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.hud_model(image, verbose=False)
        
        yolo_time = time.time() - start_time
        self.stats['avg_yolo_time'] = (self.stats['avg_yolo_time'] + yolo_time) / 2
        
        hud_regions = []
        hud_id = 0
        
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                
                if confidence >= min_confidence:
                    # Extract HUD region
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Crop HUD region from full image
                    region_image = image[y1:y2, x1:x2].copy()
                    
                    # Create HUD region object
                    hud_region = HUDRegion(
                        hud_id=hud_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        region_image=region_image,
                        elements=[]
                    )
                    
                    hud_regions.append(hud_region)
                    hud_id += 1
        
        self.stats['hud_regions_found'] += len(hud_regions)
        return hud_regions
    
    def _detect_triangles(self, hud_region: HUDRegion) -> List[DetectedElement]:
        """
        Detect triangular indicators within HUD region using OpenCV
        
        Args:
            hud_region: HUD region to process
            
        Returns:
            List of detected triangle elements
        """
        region_img = hud_region.region_image
        elements = []
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different triangle types
        color_ranges = {
            'possession_indicator': {
                'lower': np.array([5, 100, 100]),   # Orange
                'upper': np.array([25, 255, 255]),
                'type': 'possession_triangle'
            },
            'territory_indicator': {
                'lower': np.array([120, 50, 50]),   # Purple
                'upper': np.array([140, 255, 255]),
                'type': 'territory_triangle'
            },
            'generic_indicator': {
                'lower': np.array([0, 0, 200]),     # White/bright
                'upper': np.array([180, 30, 255]),
                'type': 'generic_triangle'
            }
        }
        
        for color_name, color_info in color_ranges.items():
            # Create color mask
            mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (reasonable triangle size)
                if 30 < area < 3000:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if triangle-like (3-4 vertices)
                    if 3 <= len(approx) <= 4:
                        # Get bounding box in local coordinates
                        x, y, w, h = cv2.boundingRect(contour)
                        local_bbox = (x, y, x + w, y + h)
                        
                        # Convert to global coordinates
                        hud_x1, hud_y1, _, _ = hud_region.bbox
                        global_bbox = (
                            hud_x1 + x,
                            hud_y1 + y,
                            hud_x1 + x + w,
                            hud_y1 + y + h
                        )
                        
                        # Calculate confidence based on triangle properties
                        aspect_ratio = w / h if h > 0 else 0
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        # Higher confidence for good triangle properties
                        confidence = 0.8 if (0.5 < aspect_ratio < 2.0 and circularity > 0.3) else 0.6
                        
                        element = DetectedElement(
                            element_type=color_info['type'],
                            confidence=confidence,
                            bbox=global_bbox,
                            local_bbox=local_bbox,
                            hud_id=hud_region.hud_id,
                            properties={
                                'area': area,
                                'vertices': len(approx),
                                'aspect_ratio': aspect_ratio,
                                'circularity': circularity,
                                'color_type': color_name
                            }
                        )
                        
                        elements.append(element)
        
        return elements
    
    def _detect_text_regions(self, hud_region: HUDRegion) -> List[DetectedElement]:
        """
        Detect text regions within HUD using OpenCV
        
        Args:
            hud_region: HUD region to process
            
        Returns:
            List of detected text elements
        """
        region_img = hud_region.region_image
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Text detection using MSER (Maximally Stable Extremal Regions)
        mser = cv2.MSER_create()
        
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            # Get bounding box for text region
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if 1.5 < aspect_ratio < 8.0:  # Reasonable text aspect ratio
                local_bbox = (x, y, x + w, y + h)
                
                # Convert to global coordinates
                hud_x1, hud_y1, _, _ = hud_region.bbox
                global_bbox = (
                    hud_x1 + x,
                    hud_y1 + y,
                    hud_x1 + x + w,
                    hud_y1 + y + h
                )
                
                element = DetectedElement(
                    element_type='text_region',
                    confidence=0.7,
                    bbox=global_bbox,
                    local_bbox=local_bbox,
                    hud_id=hud_region.hud_id,
                    properties={
                        'area': w * h,
                        'aspect_ratio': aspect_ratio,
                        'width': w,
                        'height': h
                    }
                )
                
                elements.append(element)
        
        return elements
    
    def _detect_score_numbers(self, hud_region: HUDRegion) -> List[DetectedElement]:
        """
        Detect score/number regions within HUD using OpenCV
        
        Args:
            hud_region: HUD region to process
            
        Returns:
            List of detected score elements
        """
        region_img = hud_region.region_image
        elements = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright text (scores are usually bright)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter for reasonable score size
            if 100 < area < 1500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Scores are typically compact and roughly square
                if 0.3 < aspect_ratio < 3.0:
                    local_bbox = (x, y, x + w, y + h)
                    
                    # Convert to global coordinates
                    hud_x1, hud_y1, _, _ = hud_region.bbox
                    global_bbox = (
                        hud_x1 + x,
                        hud_y1 + y,
                        hud_x1 + x + w,
                        hud_y1 + y + h
                    )
                    
                    element = DetectedElement(
                        element_type='score_number',
                        confidence=0.75,
                        bbox=global_bbox,
                        local_bbox=local_bbox,
                        hud_id=hud_region.hud_id,
                        properties={
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'brightness_score': cv2.mean(gray[y:y+h, x:x+w])[0]
                        }
                    )
                    
                    elements.append(element)
        
        return elements
    
    def process_hud_regions(self, hud_regions: List[HUDRegion]) -> List[HUDRegion]:
        """
        Process each HUD region with OpenCV element detectors
        
        Args:
            hud_regions: List of HUD regions from YOLO
            
        Returns:
            HUD regions with detected elements
        """
        start_time = time.time()
        
        for hud_region in hud_regions:
            # Run all element detectors on this HUD region
            for detector_name, detector_func in self.element_detectors.items():
                elements = detector_func(hud_region)
                hud_region.elements.extend(elements)
                self.stats['elements_detected'] += len(elements)
        
        opencv_time = time.time() - start_time
        self.stats['avg_opencv_time'] = (self.stats['avg_opencv_time'] + opencv_time) / 2
        
        return hud_regions
    
    def detect_elements(self, image: np.ndarray) -> Tuple[List[HUDRegion], Dict]:
        """
        Main detection method: YOLO finds HUDs, OpenCV finds elements
        
        Args:
            image: Full screenshot
            
        Returns:
            Tuple of (HUD regions with elements, detection statistics)
        """
        self.stats['total_frames'] += 1
        
        # Step 1: YOLO detects HUD regions
        hud_regions = self.detect_hud_regions(image)
        
        if not hud_regions:
            return [], self.get_stats()
        
        # Step 2: OpenCV processes ONLY HUD regions for elements
        hud_regions = self.process_hud_regions(hud_regions)
        
        return hud_regions, self.get_stats()
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_frames': self.stats['total_frames'],
            'hud_regions_found': self.stats['hud_regions_found'],
            'elements_detected': self.stats['elements_detected'],
            'avg_yolo_time_ms': self.stats['avg_yolo_time'] * 1000,
            'avg_opencv_time_ms': self.stats['avg_opencv_time'] * 1000,
            'avg_hud_per_frame': self.stats['hud_regions_found'] / max(1, self.stats['total_frames']),
            'avg_elements_per_frame': self.stats['elements_detected'] / max(1, self.stats['total_frames'])
        }
    
    def visualize_detections(self, image: np.ndarray, hud_regions: List[HUDRegion]) -> np.ndarray:
        """
        Visualize all detections on the image
        
        Args:
            image: Original image
            hud_regions: Detected HUD regions with elements
            
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        for hud_region in hud_regions:
            # Draw HUD region boundary (YOLO detection)
            x1, y1, x2, y2 = hud_region.bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow for HUD
            cv2.putText(vis_image, f"HUD {hud_region.hud_id}: {hud_region.confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw elements within HUD (OpenCV detections)
            for element in hud_region.elements:
                ex1, ey1, ex2, ey2 = element.bbox
                
                # Color code by element type
                colors = {
                    'possession_triangle': (0, 165, 255),  # Orange
                    'territory_triangle': (128, 0, 128),   # Purple
                    'generic_triangle': (255, 255, 255),   # White
                    'text_region': (0, 255, 0),            # Green
                    'score_number': (0, 0, 255)            # Red
                }
                
                color = colors.get(element.element_type, (128, 128, 128))
                cv2.rectangle(vis_image, (ex1, ey1), (ex2, ey2), color, 2)
                
                # Label
                label = f"{element.element_type}: {element.confidence:.2f}"
                cv2.putText(vis_image, label, (ex1, ey1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_image

def main():
    """Demo the hybrid detector"""
    print("🎮 Hybrid HUD Element Detector Demo")
    print("=" * 40)
    
    # Initialize detector
    detector = HybridHUDDetector()
    
    # Screen capture
    sct = mss.mss()
    monitor = sct.monitors[1]
    
    print("🎥 Capturing and processing frames...")
    print("Press Ctrl+C to stop")
    
    try:
        for frame_num in range(10):  # Process 10 frames
            # Capture screenshot
            screenshot = np.array(sct.grab(monitor))
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Detect elements
            hud_regions, stats = detector.detect_elements(screenshot)
            
            print(f"\nFrame {frame_num + 1}:")
            print(f"  HUD regions: {len(hud_regions)}")
            
            total_elements = sum(len(region.elements) for region in hud_regions)
            print(f"  Total elements: {total_elements}")
            
            # Show element breakdown
            element_counts = {}
            for region in hud_regions:
                for element in region.elements:
                    element_counts[element.element_type] = element_counts.get(element.element_type, 0) + 1
            
            for element_type, count in element_counts.items():
                print(f"    {element_type}: {count}")
            
            time.sleep(0.5)  # Brief pause between frames
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    # Print final statistics
    final_stats = detector.get_stats()
    print(f"\n📊 Final Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

if __name__ == "__main__":
    main() 
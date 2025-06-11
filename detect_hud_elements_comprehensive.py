#!/usr/bin/env python3
"""
Comprehensive HUD Elements Detection - Lower threshold to catch both possession and territory areas
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

class ComprehensiveHUDDetector:
    def __init__(self, model_path="hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"):
        """Initialize comprehensive HUD element detector."""
        self.model_path = model_path
        
        # Load the trained model
        print(f"🚀 Loading HUD element detection model: {model_path}")
        self.model = YOLO(model_path)
        
        # Class definitions from training
        self.classes = [
            "hud",                      # 0: Main HUD bar region
            "possession_triangle_area", # 1: Left triangle area (possession indicator)  
            "territory_triangle_area",  # 2: Right triangle area (territory indicator)
            "preplay_indicator",       # 3: Bottom left pre-play indicator
            "play_call_screen"         # 4: Play call screen overlay
        ]
        
        self.class_colors = {
            0: (255, 255, 0),    # Cyan - Main HUD
            1: (0, 255, 0),      # Green - Possession area
            2: (0, 0, 255),      # Red - Territory area  
            3: (255, 0, 255),    # Magenta - Pre-play
            4: (0, 165, 255)     # Orange - Play call
        }
        
        print(f"✅ Model loaded with {len(self.classes)} classes")
        
    def detect_all_elements(self, frame, min_confidence=0.1):
        """Detect HUD elements with very low confidence to catch everything."""
        
        print(f"\n🔍 Running detection with minimum confidence: {min_confidence}")
        
        # Run YOLO detection with low confidence
        results = self.model(frame, conf=min_confidence, verbose=False)
        
        all_detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        x1, y1, x2, y2 = box
                        
                        detection = {
                            'class_name': class_name,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        }
                        
                        all_detections.append(detection)
                        
                        print(f"✅ {class_name}: confidence={conf:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        
        return all_detections
    
    def filter_detections_by_confidence(self, all_detections, thresholds=None):
        """Filter detections by different confidence thresholds for each class."""
        
        if thresholds is None:
            # Default thresholds - lower for triangle areas since they might be harder to detect
            thresholds = {
                'hud': 0.3,
                'possession_triangle_area': 0.2,
                'territory_triangle_area': 0.2,  # Lower threshold for territory
                'preplay_indicator': 0.3,
                'play_call_screen': 0.3
            }
        
        filtered_detections = {}
        
        print(f"\n📊 Filtering detections with thresholds: {thresholds}")
        
        for detection in all_detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            threshold = thresholds.get(class_name, 0.3)
            
            if confidence >= threshold:
                # Keep the best detection for each class
                if class_name not in filtered_detections or confidence > filtered_detections[class_name]['confidence']:
                    filtered_detections[class_name] = detection
                    print(f"✅ ACCEPTED {class_name}: confidence={confidence:.3f} (threshold={threshold})")
            else:
                print(f"❌ REJECTED {class_name}: confidence={confidence:.3f} < threshold={threshold}")
        
        return filtered_detections
    
    def convert_to_normalized_coordinates(self, bbox, frame_width, frame_height):
        """Convert pixel bbox to normalized coordinates."""
        x1, y1, x2, y2 = bbox
        
        normalized = [
            x1 / frame_width,   # x_start
            x2 / frame_width,   # x_end  
            y1 / frame_height,  # y_start
            y2 / frame_height   # y_end
        ]
        
        return normalized
    
    def visualize_all_detections(self, frame, all_detections, filtered_detections, save_path="comprehensive_hud_detections.jpg"):
        """Visualize both all detections and filtered detections."""
        
        height, width = frame.shape[:2]
        
        # Create side-by-side visualization
        vis_width = width * 2
        vis_frame = np.zeros((height, vis_width, 3), dtype=np.uint8)
        
        # Left side: All detections
        left_frame = frame.copy()
        for detection in all_detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Draw with transparency based on confidence
            alpha = min(conf * 2, 1.0)  # Scale up low confidences for visibility
            overlay = left_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.addWeighted(overlay, alpha, left_frame, 1-alpha, 0, left_frame)
            
            # Label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(left_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Right side: Filtered detections
        right_frame = frame.copy()
        for class_name, detection in filtered_detections.items():
            bbox = detection['bbox']
            conf = detection['confidence']
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = bbox
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Draw bright, solid boxes for accepted detections
            cv2.rectangle(right_frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(right_frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(right_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Combine images
        vis_frame[:, :width] = left_frame
        vis_frame[:, width:] = right_frame
        
        # Add titles
        cv2.putText(vis_frame, "ALL DETECTIONS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, "FILTERED DETECTIONS", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(save_path, vis_frame)
        print(f"💾 Comprehensive visualization saved: {save_path}")
        
        return vis_frame
    
    def analyze_comprehensive(self, frame_path):
        """Comprehensive analysis to find both possession and territory areas."""
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"❌ Could not load frame: {frame_path}")
            return None
            
        height, width = frame.shape[:2]
        print(f"📐 Frame dimensions: {width}x{height}")
        
        # Get all detections with very low threshold
        all_detections = self.detect_all_elements(frame, min_confidence=0.05)
        
        if not all_detections:
            print(f"❌ No HUD elements detected at all!")
            return None
        
        print(f"\n📊 Found {len(all_detections)} total detections")
        
        # Filter detections by class-specific thresholds
        filtered_detections = self.filter_detections_by_confidence(all_detections)
        
        print(f"📊 Accepted {len(filtered_detections)} detections after filtering")
        
        # Create comprehensive visualization
        self.visualize_all_detections(frame, all_detections, filtered_detections)
        
        # Extract coordinate information for key elements
        coordinate_info = {}
        
        print(f"\n📍 COORDINATE ANALYSIS:")
        
        # Check for both triangle areas
        triangle_areas = ['possession_triangle_area', 'territory_triangle_area']
        
        for area_name in triangle_areas:
            if area_name in filtered_detections:
                detection = filtered_detections[area_name]
                bbox = detection['bbox']
                normalized = self.convert_to_normalized_coordinates(bbox, width, height)
                
                coordinate_info[area_name] = {
                    'pixels': bbox,
                    'normalized': normalized,
                    'confidence': detection['confidence'],
                    'format_for_code': f"{normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}"
                }
                
                print(f"\n🔺 {area_name.replace('_', ' ').title()}:")
                print(f"   Confidence: {detection['confidence']:.3f}")
                print(f"   Pixels: {bbox}")
                print(f"   Normalized: {normalized}")
                print(f"   Code format: {normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}")
            else:
                print(f"\n❌ {area_name.replace('_', ' ').title()}: NOT DETECTED")
        
        # Check for main HUD
        if 'hud' in filtered_detections:
            detection = filtered_detections['hud']
            bbox = detection['bbox']
            normalized = self.convert_to_normalized_coordinates(bbox, width, height)
            
            coordinate_info['hud'] = {
                'pixels': bbox,
                'normalized': normalized,
                'confidence': detection['confidence'],
                'format_for_code': f"{normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}"
            }
            
            print(f"\n📍 Main HUD:")
            print(f"   Confidence: {detection['confidence']:.3f}")
            print(f"   Pixels: {bbox}")
            print(f"   Normalized: {normalized}")
        
        # Save comprehensive summary
        self.save_comprehensive_summary(coordinate_info, all_detections, filtered_detections)
        
        return coordinate_info, all_detections, filtered_detections
    
    def save_comprehensive_summary(self, coordinate_info, all_detections, filtered_detections):
        """Save comprehensive analysis summary."""
        
        summary_lines = ["# COMPREHENSIVE HUD ELEMENT ANALYSIS\n\n"]
        
        summary_lines.append(f"# DETECTION SUMMARY\n")
        summary_lines.append(f"# Total detections found: {len(all_detections)}\n")
        summary_lines.append(f"# Accepted after filtering: {len(filtered_detections)}\n\n")
        
        # All detections
        summary_lines.append("# ALL DETECTIONS (with low confidence threshold)\n")
        for detection in all_detections:
            class_name = detection['class_name']
            conf = detection['confidence']
            bbox = detection['bbox']
            summary_lines.append(f"# {class_name}: confidence={conf:.3f}, bbox={bbox}\n")
        summary_lines.append("\n")
        
        # Accepted detections with coordinates
        summary_lines.append("# ACCEPTED DETECTIONS - READY TO USE\n")
        for element_name, data in coordinate_info.items():
            normalized = data['normalized']
            confidence = data['confidence']
            summary_lines.append(f"# {element_name.upper()} (confidence: {confidence:.3f})\n")
            summary_lines.append(f"{element_name}_coordinates = [{normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}]  # x_start, x_end, y_start, y_end\n\n")
        
        # Reference
        summary_lines.append("# EXISTING DOWN/DISTANCE (for reference)\n")
        summary_lines.append("down_distance_coordinates = [0.750, 0.900, 0.200, 0.800]  # x_start, x_end, y_start, y_end\n\n")
        
        with open("comprehensive_hud_analysis.txt", "w") as f:
            f.writelines(summary_lines)
        
        print(f"\n📄 Comprehensive analysis saved: comprehensive_hud_analysis.txt")

def main():
    print("🎯 COMPREHENSIVE HUD ELEMENT DETECTION")
    print("=" * 60)
    print("🔍 Looking for BOTH possession and territory triangle areas")
    print("📊 Using low confidence thresholds to catch all elements")
    
    # Initialize detector
    detector = ComprehensiveHUDDetector()
    
    # Analyze the HUD frame
    frame_path = "found_and_frame_3000.png"
    
    print(f"\n🖼️ Analyzing frame: {frame_path}")
    
    result = detector.analyze_comprehensive(frame_path)
    
    if result:
        coordinate_info, all_detections, filtered_detections = result
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Total detections: {len(all_detections)}")
        print(f"📊 Accepted detections: {len(filtered_detections)}")
        
        # Check if we got both triangle areas
        triangle_areas = ['possession_triangle_area', 'territory_triangle_area']
        found_triangles = [area for area in triangle_areas if area in coordinate_info]
        
        print(f"\n🔺 TRIANGLE AREAS STATUS:")
        print(f"   Found: {len(found_triangles)}/2 triangle areas")
        for area in found_triangles:
            conf = coordinate_info[area]['confidence']
            print(f"   ✅ {area}: confidence={conf:.3f}")
        
        missing_triangles = [area for area in triangle_areas if area not in coordinate_info]
        for area in missing_triangles:
            print(f"   ❌ {area}: NOT DETECTED")
        
        print(f"\n📁 Files created:")
        print(f"   📸 comprehensive_hud_detections.jpg - Visual comparison")
        print(f"   📄 comprehensive_hud_analysis.txt - Detailed analysis")
        
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    main() 
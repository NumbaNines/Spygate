#!/usr/bin/env python3
"""
Detect HUD Elements using trained YOLO model
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class HUDElementDetector:
    def __init__(self, model_path="hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"):
        """Initialize HUD element detector with trained model."""
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
        
    def detect_elements(self, frame, confidence_threshold=0.5):
        """Detect HUD elements in the frame."""
        
        # Run YOLO detection
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        
        detected_elements = {}
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    if conf >= confidence_threshold and class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        x1, y1, x2, y2 = box
                        
                        detected_elements[class_name] = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': int(class_id)
                        }
                        
                        print(f"✅ Detected {class_name}: confidence={conf:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
        
        return detected_elements
    
    def convert_to_normalized_coordinates(self, bbox, frame_width, frame_height):
        """Convert pixel bbox to normalized coordinates in format: x_start, x_end, y_start, y_end"""
        x1, y1, x2, y2 = bbox
        
        normalized = [
            x1 / frame_width,   # x_start
            x2 / frame_width,   # x_end  
            y1 / frame_height,  # y_start
            y2 / frame_height   # y_end
        ]
        
        return normalized
    
    def visualize_detections(self, frame, detected_elements, save_path="hud_detections_visualization.jpg"):
        """Visualize detected HUD elements."""
        
        vis_frame = frame.copy()
        
        for class_name, data in detected_elements.items():
            bbox = data['bbox']
            conf = data['confidence']
            class_id = data['class_id']
            
            x1, y1, x2, y2 = bbox
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        cv2.imwrite(save_path, vis_frame)
        print(f"💾 Visualization saved: {save_path}")
        
        return vis_frame
    
    def analyze_frame_for_coordinates(self, frame_path):
        """Analyze a frame and extract coordinate information for OCR regions."""
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"❌ Could not load frame: {frame_path}")
            return None
            
        height, width = frame.shape[:2]
        print(f"📐 Frame dimensions: {width}x{height}")
        
        # Detect HUD elements
        print(f"\n🔍 Detecting HUD elements...")
        detected_elements = self.detect_elements(frame, confidence_threshold=0.3)
        
        if not detected_elements:
            print(f"❌ No HUD elements detected!")
            return None
        
        # Extract relevant coordinate information
        coordinate_info = {}
        
        # Process HUD main area
        if 'hud' in detected_elements:
            hud_bbox = detected_elements['hud']['bbox']
            hud_normalized = self.convert_to_normalized_coordinates(hud_bbox, width, height)
            coordinate_info['hud'] = {
                'pixels': hud_bbox,
                'normalized': hud_normalized,
                'format_for_code': f"{hud_normalized[0]:.3f}, {hud_normalized[1]:.3f}, {hud_normalized[2]:.3f}, {hud_normalized[3]:.3f}"
            }
            print(f"\n📍 HUD Region:")
            print(f"   Pixels: {hud_bbox}")
            print(f"   Normalized: {hud_normalized}")
        
        # Process possession triangle area
        if 'possession_triangle_area' in detected_elements:
            poss_bbox = detected_elements['possession_triangle_area']['bbox']
            poss_normalized = self.convert_to_normalized_coordinates(poss_bbox, width, height)
            coordinate_info['possession_triangle'] = {
                'pixels': poss_bbox,
                'normalized': poss_normalized,
                'format_for_code': f"{poss_normalized[0]:.3f}, {poss_normalized[1]:.3f}, {poss_normalized[2]:.3f}, {poss_normalized[3]:.3f}"
            }
            print(f"\n🔺 Possession Triangle Area:")
            print(f"   Pixels: {poss_bbox}")
            print(f"   Normalized: {poss_normalized}")
        
        # Process territory triangle area  
        if 'territory_triangle_area' in detected_elements:
            terr_bbox = detected_elements['territory_triangle_area']['bbox']
            terr_normalized = self.convert_to_normalized_coordinates(terr_bbox, width, height)
            coordinate_info['territory_triangle'] = {
                'pixels': terr_bbox,
                'normalized': terr_normalized,
                'format_for_code': f"{terr_normalized[0]:.3f}, {terr_normalized[1]:.3f}, {terr_normalized[2]:.3f}, {terr_normalized[3]:.3f}"
            }
            print(f"\n🗺️ Territory Triangle Area:")
            print(f"   Pixels: {terr_bbox}")
            print(f"   Normalized: {terr_normalized}")
        
        # Create visualization
        self.visualize_detections(frame, detected_elements)
        
        # Save coordinate summary
        self.save_coordinate_summary(coordinate_info)
        
        return coordinate_info
    
    def save_coordinate_summary(self, coordinate_info):
        """Save coordinate summary to file."""
        
        summary_lines = ["# HUD Element Coordinates (detected via YOLO)\n\n"]
        
        for element_name, data in coordinate_info.items():
            normalized = data['normalized']
            summary_lines.append(f"# {element_name.upper()}\n")
            summary_lines.append(f"{element_name}_coordinates = [{normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}]  # x_start, x_end, y_start, y_end\n\n")
        
        # Add comparison with existing down/distance
        summary_lines.append("# EXISTING DOWN/DISTANCE (for reference)\n")
        summary_lines.append("down_distance_coordinates = [0.750, 0.900, 0.200, 0.800]  # x_start, x_end, y_start, y_end\n\n")
        
        with open("yolo_detected_coordinates.txt", "w") as f:
            f.writelines(summary_lines)
        
        print(f"\n📄 Coordinate summary saved: yolo_detected_coordinates.txt")

def main():
    """Main function to detect HUD elements and extract coordinates."""
    
    print("🎯 HUD ELEMENT DETECTION USING YOLO")
    print("=" * 50)
    
    # Initialize detector
    detector = HUDElementDetector()
    
    # Analyze the HUD frame
    frame_path = "found_and_frame_3000.png"
    
    print(f"\n🖼️ Analyzing frame: {frame_path}")
    
    coordinate_info = detector.analyze_frame_for_coordinates(frame_path)
    
    if coordinate_info:
        print(f"\n✅ SUCCESS! Detected {len(coordinate_info)} HUD elements")
        print(f"\n📋 COORDINATE SUMMARY:")
        
        for element_name, data in coordinate_info.items():
            normalized = data['normalized']
            print(f"   {element_name}: {normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}")
        
        print(f"\n🎉 Now you can use these coordinates for team scores and possession detection!")
        print(f"📁 Check 'hud_detections_visualization.jpg' to see the detected regions")
        print(f"📁 Check 'yolo_detected_coordinates.txt' for copy-paste coordinates")
        
    else:
        print(f"\n❌ Failed to detect HUD elements. Check the model and frame.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Convert triangle annotations from JSON (LabelMe) format to YOLO TXT format.
"""

import json
import os
from pathlib import Path

def convert_json_to_yolo_triangles():
    """Convert JSON triangle annotations to YOLO TXT format."""
    print("🔄 Converting JSON triangles to YOLO TXT format...")
    
    # Class mapping
    class_mapping = {
        "hud": 0,
        "qb_position": 1, 
        "left_hash_mark": 2,
        "right_hash_mark": 3,
        "preplay": 4,
        "playcall": 5,
        "possession_indicator": 6,  # Triangle class!
        "territory_indicator": 7   # Triangle class!
    }
    
    labels_dir = Path("training_data/labels")
    converted_files = 0
    triangles_added = 0
    
    # Process each JSON file
    for json_file in labels_dir.glob("*.json"):
        txt_file = json_file.with_suffix(".txt")
        
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
            continue
        
        # Get image dimensions
        img_width = data.get('imageWidth', 1920)
        img_height = data.get('imageHeight', 1080)
        
        # Read existing TXT annotations if they exist
        existing_annotations = []
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                existing_annotations = [line.strip() for line in f.readlines() if line.strip()]
        
        # Look for triangle annotations in JSON
        new_annotations = []
        found_triangles = False
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            
            # Only process triangle classes
            if label in ['possession_indicator', 'territory_indicator']:
                class_id = class_mapping[label]
                points = shape.get('points', [])
                
                if len(points) == 2:  # Rectangle format
                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # Calculate center and dimensions
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = abs(x2 - x1) / img_width
                    height = abs(y2 - y1) / img_height
                    
                    # Format as YOLO annotation
                    yolo_annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    new_annotations.append(yolo_annotation)
                    found_triangles = True
                    triangles_added += 1
        
        # Write updated annotations if triangles were found
        if found_triangles:
            all_annotations = existing_annotations + new_annotations
            
            with open(txt_file, 'w') as f:
                for annotation in all_annotations:
                    f.write(annotation + '\n')
            
            print(f"✅ {json_file.name}: Added {len(new_annotations)} triangle annotations")
            converted_files += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"   📁 Files updated: {converted_files}")
    print(f"   🔺 Triangle annotations added: {triangles_added}")
    
    # Update classes.txt to include all 8 classes
    classes_file = Path("training_data/classes.txt")
    all_classes = [
        "hud",
        "qb_position", 
        "left_hash_mark",
        "right_hash_mark",
        "preplay",
        "playcall",
        "possession_indicator",
        "territory_indicator"
    ]
    
    with open(classes_file, 'w') as f:
        for class_name in all_classes:
            f.write(class_name + '\n')
    
    print(f"✅ Updated classes.txt with all 8 classes")
    
    return converted_files, triangles_added

if __name__ == "__main__":
    convert_json_to_yolo_triangles() 
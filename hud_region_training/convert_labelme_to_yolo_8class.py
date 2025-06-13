"""
Convert labelme JSON annotations to YOLO format for 8-class HUD detection.
"""

import json
import os
from pathlib import Path
import argparse

# 8-class mapping
CLASS_MAPPING = {
    "hud": 0,
    "possession_triangle_area": 1,
    "territory_triangle_area": 2,
    "preplay_indicator": 3,
    "play_call_screen": 4,
    "down_distance_area": 5,
    "game_clock_area": 6,
    "play_clock_area": 7
}

def convert_labelme_to_yolo(json_file, output_dir):
    """Convert a single labelme JSON file to YOLO format."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # Prepare YOLO annotations
    yolo_annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASS_MAPPING:
            print(f"Warning: Unknown class '{label}' in {json_file}")
            continue
        
        class_id = CLASS_MAPPING[label]
        points = shape['points']
        
        # Convert rectangle points to YOLO format
        if shape['shape_type'] == 'rectangle' and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Calculate center and dimensions
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        elif shape['shape_type'] == 'polygon':
            # Convert polygon to bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Write YOLO annotation file
    base_name = Path(json_file).stem
    output_file = Path(output_dir) / f"{base_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return len(yolo_annotations)

def convert_directory(input_dir, output_dir):
    """Convert all labelme JSON files in a directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    total_annotations = 0
    converted_files = 0
    
    for json_file in json_files:
        try:
            annotations_count = convert_labelme_to_yolo(json_file, output_path)
            total_annotations += annotations_count
            converted_files += 1
            print(f"✅ Converted {json_file.name}: {annotations_count} annotations")
        except Exception as e:
            print(f"❌ Error converting {json_file.name}: {e}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"Files converted: {converted_files}/{len(json_files)}")
    print(f"Total annotations: {total_annotations}")

def main():
    parser = argparse.ArgumentParser(description="Convert labelme annotations to YOLO format")
    parser.add_argument("--input", "-i", required=True, help="Input directory with labelme JSON files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for YOLO txt files")
    
    args = parser.parse_args()
    
    print(f"Converting labelme annotations from {args.input} to {args.output}")
    print(f"Using 8-class mapping: {CLASS_MAPPING}")
    
    convert_directory(args.input, args.output)

if __name__ == "__main__":
    main() 
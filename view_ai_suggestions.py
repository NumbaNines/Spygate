#!/usr/bin/env python3
"""Quick visualization tool to see AI annotation suggestions overlaid on images."""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def draw_ai_suggestions(image_path, json_path, output_path=None):
    """Draw AI suggestion bounding boxes on an image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Load annotations
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Could not load JSON: {json_path} - {e}")
        return
    
    # Draw bounding boxes
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            
            label = shape['label']
            confidence = shape.get('description', '')
            
            # Choose color based on class
            colors = {
                'hud': (0, 255, 0),  # Green
                'qb_position': (255, 0, 0),  # Blue
                'left_hash_mark': (0, 255, 255),  # Yellow
                'right_hash_mark': (0, 255, 255),  # Yellow
                'possession_indicator': (255, 0, 255),  # Magenta
                'territory_indicator': (255, 255, 0),  # Cyan
            }
            color = colors.get(label, (255, 255, 255))  # White default
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label} {confidence}"
            cv2.putText(image, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"✅ Saved visualization: {output_path}")
    else:
        # Display
        cv2.imshow(f"AI Suggestions - {image_path.name}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize AI annotation suggestions")
    parser.add_argument("--input", default="converted_labels", help="Directory with JSON files")
    parser.add_argument("--images", default="converted_images", help="Directory with images")
    parser.add_argument("--output", help="Output directory for visualizations")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to show")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    images_dir = Path(args.images)
    output_dir = Path(args.output) if args.output else None
    
    if output_dir:
        output_dir.mkdir(exist_ok=True)
    
    # Get JSON files
    json_files = list(input_dir.glob("*.json"))[:args.sample]
    
    print(f"🔍 Visualizing {len(json_files)} AI suggestions...")
    
    for json_file in json_files:
        # Find corresponding image
        image_name = json_file.stem + ".png"
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        # Create output path if saving
        output_path = None
        if output_dir:
            output_path = output_dir / f"viz_{image_name}"
        
        print(f"📷 Processing: {image_name}")
        draw_ai_suggestions(image_path, json_file, output_path)
        
        # If not saving, show one at a time
        if not output_dir:
            print("Press any key to continue to next image...")

if __name__ == "__main__":
    main() 
"""
Quick script to check labelme annotations and visualize them.

This script reads the labelme JSON files and shows what triangles were annotated.
"""

import json
import cv2
import numpy as np
from pathlib import Path

def check_annotations(annotation_dir="labelme_annotations"):
    """Check what annotations were created"""
    annotation_path = Path(annotation_dir)
    json_files = list(annotation_path.glob("*.json"))
    
    if not json_files:
        print("No annotation files found!")
        return
    
    print(f"Found {len(json_files)} annotation files:")
    print("=" * 50)
    
    total_triangles = 0
    
    for json_file in json_files:
        print(f"\nFile: {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            image_path = data.get('imagePath', 'Unknown')
            shapes = data.get('shapes', [])
            
            print(f"Image: {image_path}")
            print(f"Annotations: {len(shapes)}")
            
            possession_count = 0
            territory_count = 0
            
            for i, shape in enumerate(shapes):
                label = shape.get('label', 'Unknown')
                shape_type = shape.get('shape_type', 'Unknown')
                points = shape.get('points', [])
                
                if label == 'possession_indicator':
                    possession_count += 1
                elif label == 'territory_indicator':
                    territory_count += 1
                
                print(f"  {i+1}. Label: {label}, Type: {shape_type}, Points: {len(points)}")
                
                # Show bounding box coordinates
                if points and len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    print(f"     BBox: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
            
            print(f"Summary: {possession_count} possession, {territory_count} territory triangles")
            total_triangles += len(shapes)
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    print("\n" + "=" * 50)
    print(f"TOTAL: {total_triangles} triangles annotated across {len(json_files)} images")

def visualize_annotation(json_file, output_dir="annotation_preview"):
    """Create a visualization of one annotation"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get image data
        image_data = data.get('imageData')
        if image_data:
            # Decode base64 image
            import base64
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # Try to load from file path
            image_path = Path(json_file).parent.parent / "images_to_annotate" / data.get('imagePath', '')
            if image_path.exists():
                image = cv2.imread(str(image_path))
            else:
                print(f"Could not load image for {json_file}")
                return None
        
        if image is None:
            print(f"Failed to load image for {json_file}")
            return None
        
        # Draw annotations
        shapes = data.get('shapes', [])
        for shape in shapes:
            label = shape.get('label', 'Unknown')
            points = shape.get('points', [])
            
            if points and len(points) >= 2:
                # Get bounding box
                x_coords = [int(p[0]) for p in points]
                y_coords = [int(p[1]) for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Choose color based on label
                if label == 'possession_indicator':
                    color = (255, 0, 0)  # Blue
                elif label == 'territory_indicator':
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
                
                # Draw label
                cv2.putText(image, label, (x_min, y_min-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Save visualization
        json_name = Path(json_file).stem
        output_file = output_path / f"{json_name}_annotated.jpg"
        cv2.imwrite(str(output_file), image)
        
        print(f"Saved visualization: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"Error visualizing {json_file}: {e}")
        return None

def main():
    print("Checking Labelme Triangle Annotations")
    print("=" * 40)
    
    # Check all annotations
    check_annotations()
    
    # Create visualizations for all annotation files
    print("\nCreating visualizations...")
    annotation_files = list(Path("labelme_annotations").glob("*.json"))
    
    for json_file in annotation_files:
        visualize_annotation(json_file)
    
    print(f"\nVisualization images saved to 'annotation_preview/' directory")
    print("Check these images to verify your triangle annotations look correct!")

if __name__ == "__main__":
    main() 
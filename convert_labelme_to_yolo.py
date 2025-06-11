#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to YOLO format for SpygateAI HUD detection.
Converts NEW MADDEN DATA folder to hud_region_training/dataset
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import random

# Class mapping for our 5-class system
CLASS_MAPPING = {
    'hud': 0,
    'possession_triangle_area': 1,
    'territory_triangle_area': 2,
    'preplay_indicator': 3,
    'play_call_screen': 4
}

def convert_labelme_to_yolo(labelme_json_path, image_path, output_txt_path):
    """Convert a single LabelMe JSON to YOLO format."""
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    yolo_annotations = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASS_MAPPING:
            print(f"Warning: Unknown label '{label}' in {labelme_json_path}")
            continue
            
        class_id = CLASS_MAPPING[label]
        points = shape['points']
        
        # Convert polygon to bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        center_x = (x_min + x_max) / 2 / img_width
        center_y = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Write YOLO annotation file
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return len(yolo_annotations)

def main():
    """Main conversion process."""
    input_dir = Path("NEW MADDEN DATA")
    output_dir = Path("hud_region_training/dataset")
    
    # Create output directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_dir.glob(ext)))
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split 80/20 train/val
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    total_annotations = 0
    processed = 0
    
    # Process training set
    for img_path in train_images:
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            continue
            
        # Copy image
        dst_img = train_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Convert annotations
        dst_txt = train_label_dir / img_path.with_suffix('.txt').name
        annotations = convert_labelme_to_yolo(json_path, img_path, dst_txt)
        total_annotations += annotations
        processed += 1
        
        if processed % 50 == 0:
            print(f"Processed {processed} files...")
    
    # Process validation set
    for img_path in val_images:
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            continue
            
        # Copy image
        dst_img = val_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Convert annotations
        dst_txt = val_label_dir / img_path.with_suffix('.txt').name
        annotations = convert_labelme_to_yolo(json_path, img_path, dst_txt)
        total_annotations += annotations
        processed += 1
    
    print(f"Conversion complete!")
    print(f"Processed {processed} images")
    print(f"Total annotations: {total_annotations}")
    print(f"Train images: {len(list(train_img_dir.glob('*')))}")
    print(f"Val images: {len(list(val_img_dir.glob('*')))}")
    
    # Update dataset.yaml
    yaml_content = f"""# SpygateAI HUD Region Detection Dataset
path: ./hud_region_training/dataset
train: images/train
val: images/val

# Classes
nc: 5  # number of classes
names: ['hud', 'possession_triangle_area', 'territory_triangle_area', 'preplay_indicator', 'play_call_screen']

# Class mapping:
# 0: hud - Main HUD bar region
# 1: possession_triangle_area - Left triangle area (between team names)
# 2: territory_triangle_area - Right triangle area (next to yard marker)
# 3: preplay_indicator - Pre-play indicator elements
# 4: play_call_screen - Play selection screen elements
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Updated dataset.yaml at {yaml_path}")

if __name__ == "__main__":
    main()

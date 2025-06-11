#!/usr/bin/env python3
"""
Verify training data annotations - check what we actually labeled as triangles.
This will help us understand if the model is learning from correct data.
"""

import cv2
import numpy as np
from pathlib import Path
import os

def load_yolo_annotations(label_path, img_width, img_height):
    """Load YOLO format annotations and convert to pixel coordinates."""
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to corner coordinates
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            annotations.append({
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'center': [x_center, y_center],
                'size': [width, height]
            })
    
    return annotations

def verify_training_annotations():
    """Visualize training annotations to verify triangle labels."""
    
    # Create output directory
    output_dir = Path("annotation_verification")
    output_dir.mkdir(exist_ok=True)
    
    # Class names
    class_names = {
        0: "hud",
        1: "qb_position", 
        2: "left_hash_mark",
        3: "right_hash_mark",
        4: "preplay",
        5: "playcall",
        6: "possession_indicator",  # LEFT triangle
        7: "territory_indicator"    # RIGHT triangle
    }
    
    # Colors for visualization
    colors = {
        0: (0, 0, 255),        # hud - red
        1: (0, 255, 0),        # qb_position - green
        2: (255, 0, 0),        # left_hash_mark - blue
        3: (0, 255, 255),      # right_hash_mark - cyan
        4: (255, 0, 255),      # preplay - magenta
        5: (255, 255, 0),      # playcall - yellow
        6: (255, 165, 0),      # possession_indicator - orange
        7: (128, 0, 128)       # territory_indicator - purple
    }
    
    # Check training data paths
    train_images_dir = Path("training_data/train/images")
    train_labels_dir = Path("training_data/train/labels")
    
    if not train_images_dir.exists():
        print(f"❌ Training images directory not found: {train_images_dir}")
        return
    
    if not train_labels_dir.exists():
        print(f"❌ Training labels directory not found: {train_labels_dir}")
        return
    
    # Get list of images
    image_files = list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.jpg"))
    print(f"🔍 Found {len(image_files)} training images")
    
    triangle_images_found = 0
    total_triangles = 0
    verified_images = 0
    
    # Process first 20 images to check annotations
    for i, img_path in enumerate(image_files[:20]):
        print(f"\n📋 Checking image {i+1}: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load image: {img_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Load corresponding label file
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        annotations = load_yolo_annotations(label_path, img_width, img_height)
        
        if not annotations:
            print(f"   ⚠️  No annotations found")
            continue
        
        # Check for triangles in this image
        triangles_in_image = []
        all_classes_in_image = []
        
        display_image = image.copy()
        
        for ann in annotations:
            class_id = ann['class_id']
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox
            
            all_classes_in_image.append(class_id)
            
            # Check if this is a triangle class
            if class_id in [6, 7]:  # Triangle classes
                triangles_in_image.append(class_id)
                total_triangles += 1
                
                # Draw triangle with special highlighting
                color = colors[class_id]
                thickness = 8  # Extra thick for triangles
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
                
                # Add special triangle label
                class_name = class_names[class_id]
                if class_id == 6:
                    label = f"🔺 POSSESSION (LEFT): {class_name}"
                else:
                    label = f"🔺 TERRITORY (RIGHT): {class_name}"
                
                # Label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness_text = 2
                label_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
                
                cv2.rectangle(display_image, 
                            (x1, y1-40), 
                            (x1+label_size[0]+10, y1), 
                            color, -1)
                
                cv2.putText(display_image, label, 
                          (x1+5, y1-15), 
                          font, font_scale, (255, 255, 255), thickness_text)
                
                # Show triangle coordinates and size
                center_x, center_y = ann['center']
                width, height = ann['size']
                coord_text = f"Center: ({center_x:.0f},{center_y:.0f}) Size: {width:.0f}x{height:.0f}"
                cv2.putText(display_image, coord_text, 
                          (x1, y2+25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            else:
                # Draw regular annotations
                color = colors.get(class_id, (128, 128, 128))
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                label = class_names.get(class_id, f"class_{class_id}")
                cv2.putText(display_image, label, (x1, y1-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Print analysis
        unique_classes = list(set(all_classes_in_image))
        class_names_in_image = [class_names.get(c, f"class_{c}") for c in unique_classes]
        
        print(f"   📊 Classes found: {class_names_in_image}")
        
        if triangles_in_image:
            triangle_images_found += 1
            triangle_names = [class_names[t] for t in triangles_in_image]
            print(f"   🔺 TRIANGLES FOUND: {triangle_names}")
            
            # Add info overlay to image
            info_bg_height = 100
            cv2.rectangle(display_image, (10, 10), (600, info_bg_height), (0, 0, 0), -1)
            
            info_text = [
                f"Image: {img_path.name}",
                f"Triangles: {triangle_names}",
                f"All classes: {len(unique_classes)} total"
            ]
            
            for j, text in enumerate(info_text):
                y_pos = 35 + (j * 25)
                cv2.putText(display_image, text, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save this image for verification
            output_path = output_dir / f"triangles_{verified_images+1:02d}_{img_path.stem}.png"
            cv2.imwrite(str(output_path), display_image)
            print(f"   💾 Saved verification image: {output_path.name}")
            verified_images += 1
            
        else:
            print(f"   ⚪ No triangles in this image")
        
        # Stop after finding enough triangle examples
        if verified_images >= 10:
            break
    
    # Summary
    print(f"\n" + "="*60)
    print(f"🔍 TRAINING DATA ANNOTATION VERIFICATION")
    print(f"="*60)
    print(f"📊 Images checked: {min(20, len(image_files))}")
    print(f"🔺 Images with triangles: {triangle_images_found}")
    print(f"🎯 Total triangles found: {total_triangles}")
    print(f"💾 Verification images saved: {verified_images}")
    print(f"📁 Location: {output_dir.absolute()}")
    
    if triangle_images_found == 0:
        print(f"\n❌ WARNING: NO TRIANGLES FOUND IN TRAINING DATA!")
        print(f"   This explains why the model has false positives.")
        print(f"   The training data might not have triangle annotations.")
    else:
        print(f"\n✅ Found triangles in training data")
        print(f"📋 Check the verification images to see if these are the CORRECT triangles")
        print(f"🔺 Look for:")
        print(f"   - possession_indicator: LEFT side triangle (shows ball possession)")
        print(f"   - territory_indicator: RIGHT side triangle (shows field territory)")
    
    return triangle_images_found, total_triangles

if __name__ == "__main__":
    verify_training_annotations() 
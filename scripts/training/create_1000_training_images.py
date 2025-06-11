#!/usr/bin/env python3
"""
Create 1000 more training images from existing training data with advanced augmentations.
This will improve triangle detection by providing more diverse training examples.
"""

import os
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

def load_yolo_label(label_path):
    """Load YOLO format label file."""
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append([class_id, x_center, y_center, width, height])
    
    return annotations

def save_yolo_label(annotations, label_path):
    """Save annotations in YOLO format."""
    with open(label_path, 'w') as f:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def yolo_to_bbox(annotation, img_width, img_height):
    """Convert YOLO format to bounding box format for albumentations."""
    class_id, x_center, y_center, width, height = annotation
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to corner coordinates
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2
    
    # Normalize to [0, 1] for albumentations
    x_min_norm = max(0, x_min / img_width)
    y_min_norm = max(0, y_min / img_height)
    x_max_norm = min(1, x_max / img_width)
    y_max_norm = min(1, y_max / img_height)
    
    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm, class_id]

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box format back to YOLO format."""
    x_min_norm, y_min_norm, x_max_norm, y_max_norm, class_id = bbox
    
    # Convert to center coordinates
    x_center = (x_min_norm + x_max_norm) / 2
    y_center = (y_min_norm + y_max_norm) / 2
    width = x_max_norm - x_min_norm
    height = y_max_norm - y_min_norm
    
    return [int(class_id), x_center, y_center, width, height]

def create_augmentation_pipeline():
    """Create comprehensive augmentation pipeline for HUD images."""
    
    # Define multiple augmentation pipelines with different intensities
    pipelines = []
    
    # Pipeline 1: Light augmentations (most common)
    light_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    
    # Pipeline 2: Medium augmentations
    medium_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
        ], p=0.9),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.05), intensity=(0.2, 0.5), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    
    # Pipeline 3: Heavy augmentations (less common but important for robustness)
    heavy_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(var_limit=(15.0, 35.0), p=1.0),
            A.ISONoise(color_shift=(0.03, 0.08), intensity=(0.3, 0.7), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.4),
        A.Perspective(scale=(0.03, 0.08), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    
    # Pipeline 4: Geometric transformations (careful with triangles)
    geometric_transform = A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=2, p=1.0),
            A.Affine(scale=(0.98, 1.02), translate_percent=(-0.02, 0.02), rotate=(-1, 1), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=1.0),
        ], p=0.7),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
    
    # Return pipelines with weights (how often to use each)
    return [
        (light_transform, 0.5),      # 50% of images get light augmentation
        (medium_transform, 0.3),     # 30% get medium augmentation
        (heavy_transform, 0.15),     # 15% get heavy augmentation  
        (geometric_transform, 0.05)  # 5% get geometric transformation
    ]

def augment_image_and_labels(image_path, label_path, pipelines):
    """Augment single image and its labels."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    # Convert BGR to RGB for albumentations
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image_rgb.shape[:2]
    
    # Load labels
    annotations = load_yolo_label(label_path)
    if not annotations:
        print(f"No annotations found for: {label_path}")
        return None, None
    
    # Convert YOLO format to albumentations bbox format
    bboxes = []
    class_labels = []
    for ann in annotations:
        bbox = yolo_to_bbox(ann, img_width, img_height)
        bboxes.append(bbox[:4])  # x_min, y_min, x_max, y_max
        class_labels.append(bbox[4])  # class_id
    
    # Select random augmentation pipeline
    cumulative_weights = []
    total_weight = 0
    for _, weight in pipelines:
        total_weight += weight
        cumulative_weights.append(total_weight)
    
    rand_val = random.random() * total_weight
    selected_pipeline = pipelines[0][0]  # default
    for i, cum_weight in enumerate(cumulative_weights):
        if rand_val <= cum_weight:
            selected_pipeline = pipelines[i][0]
            break
    
    # Apply augmentation
    try:
        augmented = selected_pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
        
        if not augmented['bboxes']:
            print(f"Augmentation removed all bboxes for: {image_path}")
            return None, None
            
        # Convert back to BGR for saving
        augmented_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        # Convert bboxes back to YOLO format
        augmented_annotations = []
        for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
            yolo_bbox = bbox_to_yolo(list(bbox) + [class_id], img_width, img_height)
            augmented_annotations.append(yolo_bbox)
        
        return augmented_image, augmented_annotations
        
    except Exception as e:
        print(f"Augmentation failed for {image_path}: {e}")
        return None, None

def main():
    """Main function to create 1000 augmented training images."""
    
    print("🚀 Creating 1000 new training images with advanced augmentations")
    print("This will significantly improve triangle detection performance!")
    
    # Paths
    source_images_dir = Path("training_data/images")
    source_labels_dir = Path("training_data/labels")
    
    if not source_images_dir.exists() or not source_labels_dir.exists():
        print("❌ Training data directories not found!")
        return
    
    # Get existing image files
    image_files = list(source_images_dir.glob("*.png")) + list(source_images_dir.glob("*.jpg"))
    print(f"📂 Found {len(image_files)} existing training images")
    
    if len(image_files) == 0:
        print("❌ No training images found!")
        return
    
    # Create augmentation pipelines
    pipelines = create_augmentation_pipeline()
    print(f"🔧 Created {len(pipelines)} augmentation pipelines")
    
    # Track statistics
    successful_augmentations = 0
    failed_augmentations = 0
    triangle_detections = 0
    
    print("\n🎯 Starting augmentation process...")
    
    for i in range(1000):
        # Select random source image
        source_image_path = random.choice(image_files)
        source_label_path = source_labels_dir / (source_image_path.stem + ".txt")
        
        if not source_label_path.exists():
            failed_augmentations += 1
            continue
        
        # Generate new filename
        new_filename = f"augmented_{i+1:04d}_{source_image_path.stem}"
        new_image_path = source_images_dir / f"{new_filename}.png"
        new_label_path = source_labels_dir / f"{new_filename}.txt"
        
        # Skip if already exists
        if new_image_path.exists():
            print(f"⚠️  Skipping {new_filename}, already exists")
            continue
        
        # Augment image and labels
        augmented_image, augmented_labels = augment_image_and_labels(
            source_image_path, source_label_path, pipelines
        )
        
        if augmented_image is not None and augmented_labels is not None:
            # Save augmented image
            cv2.imwrite(str(new_image_path), augmented_image)
            
            # Save augmented labels
            save_yolo_label(augmented_labels, new_label_path)
            
            # Count triangles
            triangle_count = sum(1 for label in augmented_labels if label[0] in [6, 7])  # possession_indicator, territory_indicator
            triangle_detections += triangle_count
            
            successful_augmentations += 1
            
            if (i + 1) % 100 == 0:
                print(f"✅ Created {i+1}/1000 augmented images ({triangle_detections} triangles so far)")
        else:
            failed_augmentations += 1
            print(f"❌ Failed to augment image {i+1}")
    
    print(f"\n🎉 Augmentation complete!")
    print(f"✅ Successfully created: {successful_augmentations} images")
    print(f"❌ Failed augmentations: {failed_augmentations}")
    print(f"🔺 Total triangle annotations: {triangle_detections}")
    print(f"📊 Average triangles per image: {triangle_detections/successful_augmentations:.2f}")
    
    # Final count
    final_image_count = len(list(source_images_dir.glob("*.png")) + list(source_images_dir.glob("*.jpg")))
    final_label_count = len(list(source_labels_dir.glob("*.txt")))
    
    print(f"\n📈 Final training set:")
    print(f"📷 Total images: {final_image_count}")
    print(f"🏷️  Total labels: {final_label_count}")
    
    print(f"\n🚀 Ready to retrain the model with {final_image_count} images!")
    print("Run: python train_with_triangles_final.py")

if __name__ == "__main__":
    # Install required package if needed
    try:
        import albumentations
    except ImportError:
        print("Installing albumentations for advanced augmentations...")
        import subprocess
        subprocess.check_call(["pip", "install", "albumentations"])
        import albumentations as A
    
    main() 
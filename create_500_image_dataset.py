#!/usr/bin/env python3
"""Create a curated dataset of 500 high-quality images for improved confidence training."""

import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from collections import defaultdict

def analyze_image_quality(image_path, label_path):
    """Analyze image and annotation quality for ranking."""
    score = 0
    
    # Check if image exists and loads properly
    if not image_path.exists():
        return 0
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return 0
        
        height, width = image.shape[:2]
        
        # Image quality factors
        # 1. Image size (prefer standard sizes)
        if width >= 1920 and height >= 1080:
            score += 10
        
        # 2. Check if annotation exists
        if not label_path.exists():
            return score * 0.1  # Heavily penalize missing annotations
        
        # 3. Analyze annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        annotation_score = 0
        triangle_count = 0
        hud_count = 0
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                
                # Prefer larger bounding boxes (easier to detect)
                box_area = w * h
                if box_area > 0.0005:  # At least 0.05% of image
                    annotation_score += 5
                
                # Count classes
                if cls == 0:  # HUD
                    hud_count += 1
                elif cls in [1, 2]:  # Triangles
                    triangle_count += 1
                    if box_area > 0.0002:  # Decent size triangles
                        annotation_score += 10
        
        # Prefer images with both HUD and triangles
        if hud_count > 0 and triangle_count > 0:
            annotation_score += 20
        
        score += annotation_score
        
        # 4. Image clarity (simple brightness/contrast check)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std = np.std(gray)
        if std > 50:  # Good contrast
            score += 5
        
        return score
        
    except Exception as e:
        return 0

def create_curated_dataset():
    """Create a curated dataset of 500 best images."""
    print("🎯 Creating Curated 500-Image Dataset")
    print("=" * 50)
    
    # Source directories
    src_images = Path("yolo_triangle_dataset/train/images")
    src_labels = Path("yolo_triangle_dataset/train/labels")
    
    if not src_images.exists() or not src_labels.exists():
        print("❌ Source dataset not found!")
        return
    
    # Get all images
    image_files = list(src_images.glob("*.png"))
    print(f"📊 Found {len(image_files)} total images")
    
    # Score each image
    print("🔍 Analyzing image quality...")
    scored_images = []
    
    for i, img_path in enumerate(image_files):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(image_files)} images...")
        
        # Find corresponding label
        label_path = src_labels / (img_path.stem + ".txt")
        score = analyze_image_quality(img_path, label_path)
        
        if score > 0:
            scored_images.append((score, img_path, label_path))
    
    # Sort by score (highest first)
    scored_images.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📈 Scored {len(scored_images)} valid images")
    print(f"🏆 Top score: {scored_images[0][0]}")
    print(f"📉 Lowest score: {scored_images[-1][0]}")
    
    # Select top 500
    top_500 = scored_images[:500]
    print(f"✅ Selected top 500 images for training")
    
    # Create curated dataset directories
    curated_dir = Path("yolo_triangle_dataset_500")
    curated_train = curated_dir / "train"
    curated_val = curated_dir / "val"
    
    for split_dir in [curated_train, curated_val]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Split 500 images: 400 train, 100 val
    train_images = top_500[:400]
    val_images = top_500[400:500]
    
    print("📁 Copying curated images...")
    
    # Copy training images
    for score, img_path, label_path in train_images:
        shutil.copy2(img_path, curated_train / "images" / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, curated_train / "labels" / label_path.name)
    
    # Copy validation images  
    for score, img_path, label_path in val_images:
        shutil.copy2(img_path, curated_val / "images" / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, curated_val / "labels" / label_path.name)
    
    # Create dataset YAML
    yaml_content = f"""train: train
val: val
nc: 3
names:
  0: hud
  1: possession_indicator
  2: territory_indicator
"""
    
    with open(curated_dir / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created curated dataset:")
    print(f"   📁 {curated_dir}")
    print(f"   🎯 Training: 400 images")
    print(f"   🔍 Validation: 100 images")
    print(f"   📊 Quality-ranked selection from {len(image_files)} total")
    
    return curated_dir

def main():
    """Main function."""
    curated_path = create_curated_dataset()
    
    if curated_path:
        print(f"\n🚀 Next step: Train with curated dataset")
        print(f"💡 Update train_improved_triangle_model.py to use:")
        print(f"   data=\"{curated_path}/dataset.yaml\"")

if __name__ == "__main__":
    main() 
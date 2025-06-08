#!/usr/bin/env python3
"""
SpygateAI Dataset Organization Script
This script organizes annotated gameplay footage screenshots and YOLO format labels
into a structured dataset for machine learning training.
"""

import os
import shutil
import random
from pathlib import Path
import logging

def organize_dataset():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    base_dir = Path("resized_1920x1080")
    dataset_dir = Path("test_dataset")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Create necessary directories if they don't exist
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files (excluding special files)
    annotation_files = [f for f in base_dir.glob("*.txt") 
                       if f.name.startswith("resized_monitor3_screenshot_")]
    
    # Create pairs of annotation files and their corresponding images
    valid_pairs = []
    for ann_file in annotation_files:
        img_name = ann_file.stem + ".png"
        img_path = base_dir / img_name
        
        if img_path.exists():
            valid_pairs.append((img_path, ann_file))
    
    # Shuffle the pairs
    random.shuffle(valid_pairs)
    
    # Split into train (70%), val (20%), test (10%)
    total = len(valid_pairs)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:train_size + val_size]
    test_pairs = valid_pairs[train_size + val_size:]
    
    # Copy files to their respective directories
    for pairs, split in [(train_pairs, "train"), (val_pairs, "val"), (test_pairs, "test")]:
        for img_path, ann_path in pairs:
            # Copy image
            shutil.copy2(img_path, images_dir / split / img_path.name)
            # Copy annotation
            shutil.copy2(ann_path, labels_dir / split / ann_path.name)
    
    # Copy classes.txt to dataset root
    shutil.copy2(base_dir / "classes.txt", dataset_dir / "classes.txt")
    
    print("\nDataset organization complete:")
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")
    print(f"Total: {len(valid_pairs)} image-annotation pairs")

if __name__ == "__main__":
    organize_dataset() 
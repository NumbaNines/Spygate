#!/usr/bin/env python3
"""
Test Fresh HUD Region Detection Model - Visualization
Shows what the newly trained model detects on unannotated images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob
import os

def load_fresh_model():
    """Load the freshly trained model."""
    model_path = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    model = YOLO(model_path)
    print(f"✅ Loaded fresh model from {model_path}")
    return model

def visualize_detections(image_path, model, output_dir="test_results"):
    """Run inference and create visualization."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(image_path, conf=0.25)  # Lower confidence to see more detections
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image_rgb)
    ax1.set_title("Original Image", fontsize=16)
    ax1.axis('off')
    
    # Detected regions
    ax2.imshow(image_rgb)
    ax2.set_title("Fresh Model Detections", fontsize=16)
    ax2.axis('off')
    
    # Class names and colors
    class_names = {
        0: 'hud',
        1: 'possession_triangle_area', 
        2: 'territory_triangle_area',
        3: 'preplay_indicator',
        4: 'play_call_screen'
    }
    
    colors = {
        0: 'red',      # hud
        1: 'yellow',   # possession_triangle_area
        2: 'cyan',     # territory_triangle_area  
        3: 'green',    # preplay_indicator
        4: 'magenta'   # play_call_screen
    }
    
    detection_count = 0
    
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3,
                    edgecolor=colors.get(cls, 'white'),
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                # Add label
                label = f"{class_names.get(cls, 'unknown')}: {conf:.2f}"
                ax2.text(x1, y1-10, label, 
                        color=colors.get(cls, 'white'),
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                detection_count += 1
                print(f"   📍 {class_names.get(cls, 'unknown')}: {conf:.3f} at ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
    
    # Add summary text
    summary_text = f"Detections Found: {detection_count}\nModel: Fresh HUD Regions (mAP50: 98.6%)"
    ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Save visualization
    filename = Path(image_path).stem
    output_path = f"{output_dir}/detection_{filename}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")
    
    plt.close()
    return detection_count

def find_test_images():
    """Find some test images to use."""
    # Look for images in various locations
    search_paths = [
        "NEW MADDEN DATA/*.png",
        "NEW MADDEN DATA/*.jpg", 
        "*.png",
        "*.jpg",
        "spygate/test_data/*.png",
        "spygate/test_data/*.jpg"
    ]
    
    all_images = []
    for pattern in search_paths:
        images = glob.glob(pattern)
        all_images.extend(images)
    
    # Remove duplicates and limit to first 5
    unique_images = list(set(all_images))
    return unique_images[:5]

def main():
    """Test the fresh model on unannotated images."""
    print("🎯 Testing Fresh HUD Region Detection Model")
    print("=" * 50)
    
    # Load model
    model = load_fresh_model()
    if not model:
        return
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("❌ No test images found!")
        print("Please ensure there are PNG/JPG files in the current directory or NEW MADDEN DATA folder")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    total_detections = 0
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🔍 Testing image {i}/{len(test_images)}: {Path(image_path).name}")
        detections = visualize_detections(image_path, model)
        if detections is not None:
            total_detections += detections
    
    print(f"\n🎉 Testing Complete!")
    print(f"📊 Total detections across all images: {total_detections}")
    print(f"📁 Visualizations saved in: test_results/")
    print(f"🎯 Model Performance: mAP50 98.6%, Precision 97.9%, Recall 95.7%")

if __name__ == "__main__":
    main() 
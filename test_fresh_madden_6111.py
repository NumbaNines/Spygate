#!/usr/bin/env python3

"""
Test Fresh HUD Region Detection Model on Completely Unannotated Images
Tests the newly trained model on images from 'madden 6111' folder - these are 100% fresh!
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
import time
import random

def load_fresh_model():
    """Load the freshly trained model."""
    model_path = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    model = YOLO(model_path)
    print(f"✅ Loaded fresh model from {model_path}")
    return model

def visualize_detections(image_path, model, output_dir, show_confidence=True):
    """Create side-by-side visualization of original vs detections."""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    start_time = time.time()
    results = model(image_path, conf=0.25, verbose=False)  # Lower confidence for more detections
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image_rgb)
    ax1.set_title(f"Original: {Path(image_path).name}", fontsize=14, pad=20)
    ax1.axis('off')
    
    # Image with detections
    ax2.imshow(image_rgb)
    
    # Class colors
    class_colors = {
        0: 'red',           # hud
        1: 'yellow',        # possession_triangle_area
        2: 'cyan',          # territory_triangle_area
        3: 'green',         # preplay_indicator
        4: 'magenta'        # play_call_screen
    }
    
    class_names = {
        0: 'hud',
        1: 'possession_triangle_area', 
        2: 'territory_triangle_area',
        3: 'preplay_indicator',
        4: 'play_call_screen'
    }
    
    detection_count = 0
    detection_info = []
    
    # Process detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=3,
                    edgecolor=class_colors.get(cls, 'white'),
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                # Add label with confidence
                label = f"{class_names.get(cls, f'class_{cls}')}"
                if show_confidence:
                    label += f"\n{conf:.3f}"
                
                ax2.text(x1, y1-10, label, 
                        fontsize=10, 
                        color=class_colors.get(cls, 'white'),
                        weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='black', 
                                alpha=0.7))
                
                detection_count += 1
                detection_info.append({
                    'class': class_names.get(cls, f'class_{cls}'),
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
    
    ax2.set_title(f"Fresh Model Detections: {detection_count} found\nInference: {inference_time:.1f}ms", 
                  fontsize=14, pad=20)
    ax2.axis('off')
    
    # Save the visualization
    output_file = output_dir / f"fresh_test_{Path(image_path).stem}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Processed {Path(image_path).name}: {detection_count} detections in {inference_time:.1f}ms")
    
    return {
        'image': str(image_path),
        'detections': detection_count,
        'inference_time_ms': inference_time,
        'detection_details': detection_info
    }

def main():
    """Test the fresh model on madden 6111 images."""
    
    print("🎯 Testing Fresh HUD Region Model on Completely Unannotated Images")
    print("="*70)
    
    # Load the fresh model
    model = load_fresh_model()
    if model is None:
        return
    
    # Setup paths
    input_dir = Path("madden 6111")
    output_dir = Path("fresh_madden_6111_results")
    output_dir.mkdir(exist_ok=True)
    
    # Get all PNG images
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ No PNG files found in {input_dir}")
        return
    
    # Randomly select 50 images (or all if less than 50)
    num_to_test = min(50, len(image_files))
    selected_images = random.sample(image_files, num_to_test)
    
    print(f"📊 Found {len(image_files)} total images")
    print(f"🎲 Randomly selected {num_to_test} images for testing")
    print(f"💾 Results will be saved to: {output_dir}")
    print()
    
    # Process images
    results = []
    total_detections = 0
    total_time = 0
    
    for i, image_path in enumerate(selected_images, 1):
        print(f"[{i:2d}/{num_to_test}] Testing {image_path.name}...")
        
        result = visualize_detections(image_path, model, output_dir)
        if result:
            results.append(result)
            total_detections += result['detections']
            total_time += result['inference_time_ms']
    
    # Print summary
    print()
    print("🎉 FRESH MODEL TESTING COMPLETE!")
    print("="*50)
    print(f"✅ Images Tested: {len(results)}")
    print(f"✅ Total Detections: {total_detections}")
    print(f"✅ Average Detections per Image: {total_detections/len(results):.1f}")
    print(f"✅ Average Inference Time: {total_time/len(results):.1f}ms")
    print(f"✅ Visualizations saved in: {output_dir}")
    
    # Class breakdown
    class_counts = {}
    confidence_scores = []
    
    for result in results:
        for detection in result['detection_details']:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(detection['confidence'])
    
    print()
    print("📊 Detection Breakdown by Class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"   {class_name}: {count} detections")
    
    if confidence_scores:
        print()
        print(f"🎯 Confidence Statistics:")
        print(f"   Average: {np.mean(confidence_scores):.3f}")
        print(f"   Min: {np.min(confidence_scores):.3f}")
        print(f"   Max: {np.max(confidence_scores):.3f}")
        print(f"   Std Dev: {np.std(confidence_scores):.3f}")
    
    print()
    print("🔍 These are COMPLETELY FRESH images the model has NEVER seen!")
    print("🏆 This is the true test of our hybrid HUD detection system!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Inspect test images and templates to understand detection issues
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def show_image_comparison():
    """Show side-by-side comparison of test images and templates"""
    
    # Test cases
    test_cases = [
        ("templates/raw_gameplay/1st_10.png", "down_templates_real/1ST.png", "1ST vs 1st_10"),
        ("templates/raw_gameplay/2nd_7.png", "down_templates_real/2ND.png", "2ND vs 2nd_7"),
        ("templates/raw_gameplay/4th_goal.png", "down_templates_real/4TH_GOAL.png", "4TH_GOAL vs 4th_goal"),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(12, 4*len(test_cases)))
    
    for i, (test_path, template_path, title) in enumerate(test_cases):
        # Load test image
        test_img = cv2.imread(test_path)
        if test_img is not None:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = np.zeros((50, 125, 3), dtype=np.uint8)
            
        # Load template
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is not None:
            template_img_rgb = cv2.cvtColor(template_img, cv2.COLOR_GRAY2RGB)
        else:
            template_img_rgb = np.zeros((98, 192, 3), dtype=np.uint8)
        
        # Show test image
        axes[i, 0].imshow(test_img_rgb)
        axes[i, 0].set_title(f"Test: {Path(test_path).name} ({test_img_rgb.shape[1]}x{test_img_rgb.shape[0]})")
        axes[i, 0].axis('off')
        
        # Show template
        axes[i, 1].imshow(template_img_rgb)
        axes[i, 1].set_title(f"Template: {Path(template_path).name} ({template_img_rgb.shape[1]}x{template_img_rgb.shape[0]})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_vs_template_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved as: test_vs_template_comparison.png")

def analyze_image_content():
    """Analyze the actual content of test images"""
    print("="*60)
    print("TEST IMAGE CONTENT ANALYSIS")
    print("="*60)
    
    test_files = [
        "templates/raw_gameplay/1st_10.png",
        "templates/raw_gameplay/2nd_7.png", 
        "templates/raw_gameplay/3rd_goal.png",
        "templates/raw_gameplay/4th_goal.png"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            img = cv2.imread(test_file)
            if img is not None:
                print(f"\n{Path(test_file).name}:")
                print(f"  Shape: {img.shape}")
                print(f"  Size: {img.shape[1]}x{img.shape[0]}")
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(f"  Pixel range: {gray.min()} - {gray.max()}")
                print(f"  Mean intensity: {gray.mean():.1f}")
                
                # Check if it's mostly white/empty
                white_pixels = np.sum(gray > 240)
                total_pixels = gray.shape[0] * gray.shape[1]
                white_percentage = (white_pixels / total_pixels) * 100
                print(f"  White pixels: {white_percentage:.1f}%")
                
                # Save a larger version for inspection
                debug_path = f"debug_inspect_{Path(test_file).stem}.png"
                # Resize for better visibility
                resized = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(debug_path, resized)
                print(f"  Debug image saved: {debug_path}")
            else:
                print(f"\n{Path(test_file).name}: Failed to load")
        else:
            print(f"\n{Path(test_file).name}: File not found")

def check_template_content():
    """Check template content"""
    print("\n" + "="*60)
    print("TEMPLATE CONTENT ANALYSIS")
    print("="*60)
    
    template_dir = Path("down_templates_real")
    
    for template_file in template_dir.glob("*.png"):
        template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            print(f"\n{template_file.name}:")
            print(f"  Shape: {template.shape}")
            print(f"  Pixel range: {template.min()} - {template.max()}")
            print(f"  Mean intensity: {template.mean():.1f}")
            
            # Save a larger version for inspection
            debug_path = f"debug_template_{template_file.stem}.png"
            # Resize for better visibility
            resized = cv2.resize(template, (template.shape[1]*2, template.shape[0]*2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(debug_path, resized)
            print(f"  Debug image saved: {debug_path}")

if __name__ == "__main__":
    analyze_image_content()
    check_template_content()
    
    # Try to create the comparison plot
    try:
        show_image_comparison()
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60) 
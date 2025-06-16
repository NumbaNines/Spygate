#!/usr/bin/env python3
"""
View the real Madden templates we just created.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def view_templates():
    """Display all the real Madden templates."""
    print("👀 Viewing Real Madden Templates")
    print("=" * 40)
    
    template_dir = Path("down_templates_real")
    
    if not template_dir.exists():
        print("❌ Template directory not found!")
        return
    
    template_files = sorted(template_dir.glob("*.png"))
    
    if not template_files:
        print("❌ No template files found!")
        return
    
    print(f"📋 Found {len(template_files)} templates")
    print()
    
    # Create a figure to show all templates
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Real Madden Down Templates', fontsize=16)
    
    for i, template_file in enumerate(template_files):
        if i >= 9:  # Only show first 9
            break
            
        img = cv2.imread(str(template_file))
        if img is None:
            print(f"❌ Failed to load {template_file.name}")
            continue
            
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate position in grid
        row = i // 3
        col = i % 3
        
        # Display the template
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(f"{template_file.name}\n{img.shape[1]}x{img.shape[0]}")
        axes[row, col].axis('off')
        
        print(f"📸 {template_file.name}: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Hide any unused subplots
    for i in range(len(template_files), 9):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('down_templates_preview.png', dpi=150, bbox_inches='tight')
    print()
    print("✅ Templates preview saved as 'down_templates_preview.png'")
    plt.show()

if __name__ == "__main__":
    view_templates() 
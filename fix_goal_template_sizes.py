#!/usr/bin/env python3
"""
Fix GOAL template sizes by cropping them to just the down part.
"""

import cv2
import numpy as np
from pathlib import Path


def fix_goal_templates():
    """Fix GOAL template sizes by cropping to just the down number."""
    print("🔧 Fixing GOAL Template Sizes")
    print("=" * 40)
    
    template_dir = Path("down_templates_real")
    
    # GOAL templates that need fixing
    goal_templates = ["1ST_GOAL.png", "2ND_GOAL.png"]
    
    for template_name in goal_templates:
        template_path = template_dir / template_name
        
        if not template_path.exists():
            print(f"⚠️ {template_name} not found")
            continue
            
        # Load the oversized GOAL template
        img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Failed to load {template_name}")
            continue
            
        print(f"📏 {template_name}: {img.shape[1]}x{img.shape[0]} (original)")
        
        # Crop to just the down number part (left side)
        # The down number should be in the leftmost ~80 pixels
        if img.shape[1] > 100:  # Only crop if it's oversized
            # Crop to 80x30 to match normal templates
            target_height = 30
            target_width = 80
            
            # Take from the left side where the down number is
            cropped = img[:target_height, :target_width]
            
            # Save the fixed template
            cv2.imwrite(str(template_path), cropped)
            print(f"✅ Fixed {template_name}: {cropped.shape[1]}x{cropped.shape[0]} (cropped)")
        else:
            print(f"✅ {template_name}: Already correct size")


def verify_all_templates():
    """Verify all templates are now the same size."""
    print("\n📏 Template Size Verification:")
    print("=" * 35)
    
    template_dir = Path("down_templates_real")
    
    sizes = {}
    for template_file in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            size = f"{img.shape[1]}x{img.shape[0]}"
            sizes[template_file.name] = size
            print(f"📋 {template_file.name}: {size}")
    
    # Check if all sizes are consistent
    unique_sizes = set(sizes.values())
    if len(unique_sizes) == 1:
        print(f"\n✅ All templates are consistent size: {list(unique_sizes)[0]}")
        return True
    else:
        print(f"\n❌ Inconsistent sizes found: {unique_sizes}")
        return False


def test_fixed_templates():
    """Test the fixed templates."""
    print("\n🧪 Testing Fixed Templates:")
    print("=" * 30)
    
    from down_template_detector import DownTemplateDetector
    
    # Reload detector with fixed templates
    detector = DownTemplateDetector()
    print(f"✅ Loaded {len(detector.templates)} templates")
    
    # Test with a GOAL screenshot
    test_file = "templates/raw_gameplay/3rd_goal.png"
    if Path(test_file).exists():
        frame = cv2.imread(test_file)
        height, width = frame.shape[:2]
        bbox = (0, 0, width, height)
        
        # Test detection
        result = detector.detect_down_in_yolo_region(frame, bbox, True)  # is_goal=True
        
        if result:
            print(f"🎯 GOAL Detection: {result.down} (conf: {result.confidence:.3f})")
            print(f"   Template: {result.template_name}")
 
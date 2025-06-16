#!/usr/bin/env python3
"""
Fix the real GOAL templates by cropping them to proper size.
"""

import cv2
from pathlib import Path

def fix_goal_templates():
    """Crop the oversized real GOAL templates."""
    print("🔧 Fixing Real GOAL Templates")
    print("=" * 40)
    
    template_dir = Path("down_templates_real")
    
    # Fix 1ST_GOAL (602x94 -> ~80x30)
    goal_1st = template_dir / "1ST_GOAL.png"
    if goal_1st.exists():
        img = cv2.imread(str(goal_1st))
        if img is not None:
            print(f"📏 1ST_GOAL original: {img.shape[1]}x{img.shape[0]}")
            
            # Crop to just the down number part (left side)
            # Take leftmost 80 pixels and top 30 pixels
            cropped = img[:30, :80]
            
            cv2.imwrite(str(goal_1st), cropped)
            print(f"✅ 1ST_GOAL cropped to: {cropped.shape[1]}x{cropped.shape[0]}")
    
    # Fix 2ND_GOAL (661x91 -> ~80x30)  
    goal_2nd = template_dir / "2ND_GOAL.png"
    if goal_2nd.exists():
        img = cv2.imread(str(goal_2nd))
        if img is not None:
            print(f"📏 2ND_GOAL original: {img.shape[1]}x{img.shape[0]}")
            
            # Crop to just the down number part (left side)
            cropped = img[:30, :80]
            
            cv2.imwrite(str(goal_2nd), cropped)
            print(f"✅ 2ND_GOAL cropped to: {cropped.shape[1]}x{cropped.shape[0]}")

def verify_templates():
    """Verify all templates are now consistent size."""
    print("\n📏 Template Verification:")
    print("=" * 30)
    
    template_dir = Path("down_templates_real")
    
    for template_file in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(template_file))
        if img is not None:
            size = f"{img.shape[1]}x{img.shape[0]}"
            status = "✅" if img.shape[1] <= 100 else "⚠️"
            print(f"{status} {template_file.name}: {size}")

if __name__ == "__main__":
    fix_goal_templates()
    verify_templates() 
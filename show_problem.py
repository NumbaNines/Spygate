#!/usr/bin/env python3
"""Show the exact problem with template sizes."""

import cv2

def show_the_problem():
    """Show exactly what's wrong."""
    print("🔍 THE PROBLEM:")
    print("=" * 50)
    
    # Load real screenshot
    real = cv2.imread('debug_real_original.png')
    template = cv2.imread('debug_template_1ST.png')
    
    if real is not None and template is not None:
        print(f"📱 Real Madden screenshot: {real.shape[1]}x{real.shape[0]} pixels")
        print(f"🎯 Our template:           {template.shape[1]}x{template.shape[0]} pixels")
        print()
        print("❌ THE ISSUE:")
        print(f"   - Real screenshot is TINY: {real.shape[1]}x{real.shape[0]}")
        print(f"   - Our template is HUGE: {template.shape[1]}x{template.shape[0]}")
        print(f"   - Template is {template.shape[1]/real.shape[1]:.1f}x wider!")
        print(f"   - Template is {template.shape[0]/real.shape[0]:.1f}x taller!")
        print()
        print("💡 SOLUTION NEEDED:")
        print("   - Create templates FROM the real screenshots")
        print("   - Or use the real screenshots AS templates")
        print("   - Current templates are from synthetic text, not real Madden")
    else:
        print("❌ Debug images not found")

if __name__ == "__main__":
    show_the_problem() 
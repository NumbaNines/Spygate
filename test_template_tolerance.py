#!/usr/bin/env python3
"""
Test template matching tolerance for pixel shifts.
Check if GOAL text positioning affects down detection accuracy.
"""

from pathlib import Path

import cv2
import numpy as np


def test_template_shift_tolerance():
    """Test how pixel shifts affect template matching confidence."""

    templates_dir = Path("down_templates_real")
    if not templates_dir.exists():
        print("❌ No templates found! Run template creator first.")
        return

    # Load a template
    template_path = templates_dir / "1ST.png"
    if not template_path.exists():
        print("❌ 1ST template not found!")
        return

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("❌ Failed to load template!")
        return

    print(f"🔍 Testing template: {template.shape[1]}x{template.shape[0]}px")

    # Create test image with template at center
    test_size = 400
    test_image = np.zeros((test_size, test_size), dtype=np.uint8)

    # Place template at center
    center_x = test_size // 2 - template.shape[1] // 2
    center_y = test_size // 2 - template.shape[0] // 2

    test_image[center_y : center_y + template.shape[0], center_x : center_x + template.shape[1]] = (
        template
    )

    # Test different shift amounts
    shifts = [0, 1, 2, 3, 4, 5, 10, 15, 20]
    results = []

    for shift in shifts:
        # Create shifted version
        shifted_image = np.zeros_like(test_image)
        shifted_x = center_x - shift  # Shift left (like GOAL would do)

        if shifted_x >= 0 and shifted_x + template.shape[1] < test_size:
            shifted_image[
                center_y : center_y + template.shape[0], shifted_x : shifted_x + template.shape[1]
            ] = template

            # Template match
            result = cv2.matchTemplate(shifted_image, template, cv2.TM_CCOEFF_NORMED)
            max_val = np.max(result)

            results.append((shift, max_val))
            print(f"📍 Shift {shift:2d} pixels left: {max_val:.3f} confidence")
        else:
            results.append((shift, 0.0))
            print(f"📍 Shift {shift:2d} pixels left: out of bounds")

    # Analysis
    print("\n📊 Analysis:")
    baseline = results[0][1]  # 0 shift confidence

    for shift, confidence in results[1:]:
        if confidence > 0:
            loss = (baseline - confidence) / baseline * 100
            status = "✅ Good" if loss < 10 else "⚠️ Reduced" if loss < 30 else "❌ Poor"
            print(f"   {shift:2d}px shift: {loss:5.1f}% confidence loss - {status}")

    # Recommendation
    print("\n💡 Recommendation:")
    acceptable_shifts = [s for s, c in results if c > baseline * 0.8]  # 80% of original
    if len(acceptable_shifts) > 5:
        print("✅ Templates should work fine with GOAL text shifts")
    elif len(acceptable_shifts) > 2:
        print("⚠️ Templates may work but consider GOAL-specific versions")
    else:
        print("❌ Need separate GOAL templates - shifts cause too much confidence loss")

    return results


def check_goal_screenshots():
    """Check if we have any GOAL situation screenshots."""
    screenshots_dir = Path("down templates")

    print("\n🔍 Checking for GOAL situation screenshots...")

    if screenshots_dir.exists():
        screenshots = list(screenshots_dir.glob("*.png")) + list(screenshots_dir.glob("*.jpg"))
        print(f"📸 Found {len(screenshots)} screenshots:")
        for screenshot in screenshots:
            print(f"   - {screenshot.name}")

        print("\n💡 To test GOAL situations:")
        print("1. Look for screenshots with 'GOAL' text in down area")
        print("2. Use template creator to crop GOAL versions")
        print("3. Compare template positions between normal and GOAL")
    else:
        print("❌ No screenshots folder found")


if __name__ == "__main__":
    print("🧪 SpygateAI Template Shift Tolerance Test")
    print("=" * 50)

    test_template_shift_tolerance()
    check_goal_screenshots()

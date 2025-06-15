"""
Quick Fix: Relaxed Triangle Validation Parameters
================================================

Adjust the geometric validation thresholds to be less strict
and allow real triangles to pass through.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def update_triangle_detector_thresholds():
    """
    Update the TriangleOrientationDetector with relaxed thresholds.
    """

    triangle_detector_path = Path("src/spygate/ml/triangle_orientation_detector.py")

    if not triangle_detector_path.exists():
        print(f"❌ Could not find {triangle_detector_path}")
        return False

    print("🔧 UPDATING TRIANGLE VALIDATION THRESHOLDS")
    print("=" * 50)

    # Read the current file
    with open(triangle_detector_path) as f:
        content = f.read()

    # Define the threshold updates
    updates = [
        # Relax convexity requirement
        ("MIN_CONVEXITY = 0.85", "MIN_CONVEXITY = 0.70"),
        # Increase vertex tolerance
        ("MAX_VERTICES = 6", "MAX_VERTICES = 8"),
        # Relax aspect ratio ranges
        ("POSSESSION_ASPECT_RANGE = (0.8, 1.2)", "POSSESSION_ASPECT_RANGE = (0.6, 1.5)"),
        ("TERRITORY_ASPECT_RANGE = (0.8, 1.2)", "TERRITORY_ASPECT_RANGE = (0.6, 1.5)"),
        # Lower minimum area requirements
        ("MIN_TRIANGLE_AREA = 100", "MIN_TRIANGLE_AREA = 50"),
        # Relax symmetry tolerance
        ("SYMMETRY_TOLERANCE = 0.3", "SYMMETRY_TOLERANCE = 0.5"),
    ]

    # Apply updates
    updated_content = content
    changes_made = []

    for old_value, new_value in updates:
        if old_value in updated_content:
            updated_content = updated_content.replace(old_value, new_value)
            changes_made.append(f"✅ {old_value} → {new_value}")
        else:
            print(f"⚠️  Could not find: {old_value}")

    # Show changes
    print("\n📝 CHANGES MADE:")
    for change in changes_made:
        print(f"   {change}")

    if changes_made:
        # Backup original file
        backup_path = triangle_detector_path.with_suffix(".py.backup")
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"\n💾 Backup saved: {backup_path}")

        # Write updated file
        with open(triangle_detector_path, "w") as f:
            f.write(updated_content)
        print(f"✅ Updated: {triangle_detector_path}")

        return True
    else:
        print("❌ No changes were made - thresholds may already be updated")
        return False


def test_relaxed_validation():
    """
    Test the relaxed validation on the current frame.
    """
    print("\n🧪 TESTING RELAXED VALIDATION")
    print("=" * 35)

    try:
        # Import after updating the file
        import cv2
        import numpy as np

        from src.spygate.ml.triangle_orientation_detector import TriangleOrientationDetector

        # Load test frame
        frame = cv2.imread("extracted_frame.jpg")
        if frame is None:
            print("❌ Could not load extracted_frame.jpg")
            return

        # Initialize detector with relaxed parameters
        debug_dir = Path("debug_relaxed_validation")
        detector = TriangleOrientationDetector(debug_output_dir=debug_dir)

        # Test on territory area (where the "2" was)
        territory_x1, territory_y1, territory_x2, territory_y2 = 1053, 650, 1106, 689
        territory_roi = frame[territory_y1:territory_y2, territory_x1:territory_x2]

        # Find contours
        gray = cv2.cvtColor(territory_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} contours in territory area")

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 30:  # Filter small contours
                result = detector.analyze_territory_triangle(contour, territory_roi)
                print(
                    f"Contour {i+1}: Area={area:.1f}, Valid={result.is_valid}, Reason={result.validation_reason}"
                )

        print("\n✅ Relaxed validation test complete!")

    except Exception as e:
        print(f"❌ Error testing relaxed validation: {e}")


if __name__ == "__main__":
    print("🚀 QUICK FIX: RELAXED TRIANGLE VALIDATION")
    print("=" * 50)
    print()

    # Update thresholds
    success = update_triangle_detector_thresholds()

    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Test the relaxed validation")
        print("2. Run your triangle detection again")
        print("3. Check if real triangles are now detected")
        print("4. Fine-tune thresholds if needed")
        print()
        print("💡 If this fixes the issue, we avoid the complexity of CNN!")

        # Test the changes
        test_relaxed_validation()
    else:
        print("\n❌ Could not update thresholds automatically")
        print("Manual update may be needed")

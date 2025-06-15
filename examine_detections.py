import cv2
import numpy as np


def examine_detections():
    """Examine what was detected as triangle areas."""

    print("=== EXAMINING TRIANGLE DETECTIONS ===")

    # Load the ROI images
    possession_roi = cv2.imread("debug_output/possession_triangle_area_roi.png")
    territory_roi = cv2.imread("debug_output/territory_triangle_area_roi.png")

    if possession_roi is not None:
        print(f"\nPossession Triangle ROI:")
        print(f"  Shape: {possession_roi.shape}")
        print(f"  This should contain the left triangle (→) between team abbreviations")

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(possession_roi, cv2.COLOR_BGR2GRAY)

        # Check if it contains text/numbers
        unique_values = len(np.unique(gray))
        print(f"  Unique pixel values: {unique_values}")

        # Look for bright/white pixels (text)
        bright_pixels = np.sum(gray > 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels
        print(f"  Bright pixel ratio: {bright_ratio:.3f}")

    if territory_roi is not None:
        print(f"\nTerritory Triangle ROI:")
        print(f"  Shape: {territory_roi.shape}")
        print(f"  This should contain the right triangle (▲/▼) next to yard line")

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(territory_roi, cv2.COLOR_BGR2GRAY)

        # Check if it contains text/numbers
        unique_values = len(np.unique(gray))
        print(f"  Unique pixel values: {unique_values}")

        # Look for bright/white pixels (text)
        bright_pixels = np.sum(gray > 200)
        total_pixels = gray.shape[0] * gray.shape[1]
        bright_ratio = bright_pixels / total_pixels
        print(f"  Bright pixel ratio: {bright_ratio:.3f}")

        # Check if this looks like it contains a number "4"
        if bright_ratio > 0.1:  # If there's significant white text
            print(
                f"  *** This region likely contains text/numbers (possibly the '4' you mentioned) ***"
            )

    # Load the main detection visualization
    detection_vis = cv2.imread("debug_output/detections_visualization.png")
    if detection_vis is not None:
        print(f"\nMain detection visualization loaded: {detection_vis.shape}")
        print("Check debug_output/detections_visualization.png to see all detections")

    print(f"\n=== ANALYSIS COMPLETE ===")
    print("The model detected regions it thinks contain triangles.")
    print("If it detected a '4' as a triangle, this suggests:")
    print("1. The training data may have included numbers near triangle areas")
    print("2. The model learned to associate that region with triangles")
    print("3. We may need to refine the training data or detection logic")


if __name__ == "__main__":
    examine_detections()

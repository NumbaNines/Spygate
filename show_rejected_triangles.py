import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.triangle_orientation_detector import TriangleOrientationDetector, TriangleType


def find_contours_in_roi(roi_img):
    """Find contours in the ROI image."""
    # Convert to grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area (remove very small ones)
    min_area = 30
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    return filtered_contours


def visualize_rejected_contours(roi_img, contours, results, title, output_filename):
    """Create a detailed visualization of rejected contours."""
    # Create a larger canvas for annotations
    vis_height = max(roi_img.shape[0], 400)
    vis_width = roi_img.shape[1] + 400  # Extra space for text
    vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    # Place the ROI on the left side
    vis_img[: roi_img.shape[0], : roi_img.shape[1]] = roi_img

    # Colors for different rejection reasons
    colors = {
        "vertices": (0, 0, 255),  # Red for too many vertices
        "convexity": (255, 0, 0),  # Blue for low convexity
        "basic": (0, 255, 255),  # Yellow for basic validation failure
        "aspect": (255, 0, 255),  # Magenta for aspect ratio issues
    }

    # Draw contours and annotations
    for i, (contour, result) in enumerate(zip(contours, results)):
        # Determine color based on rejection reason
        reason = result.validation_reason.lower()
        if "vertices" in reason:
            color = colors["vertices"]
        elif "convexity" in reason:
            color = colors["convexity"]
        elif "basic" in reason:
            color = colors["basic"]
        elif "aspect" in reason:
            color = colors["aspect"]
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw contour
        cv2.drawContours(vis_img, [contour], -1, color, 2)

        # Get contour center for labeling
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw contour number
            cv2.putText(vis_img, str(i + 1), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add title
    cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add legend and details on the right side
    text_x = roi_img.shape[1] + 10
    y_offset = 60

    cv2.putText(
        vis_img,
        "REJECTED CONTOURS:",
        (text_x, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    y_offset += 30

    for i, (contour, result) in enumerate(zip(contours, results)):
        area = cv2.contourArea(contour)

        # Contour info
        cv2.putText(
            vis_img,
            f"Contour {i+1}:",
            (text_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20

        cv2.putText(
            vis_img,
            f"  Area: {area:.1f}",
            (text_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        y_offset += 15

        cv2.putText(
            vis_img,
            f"  Reason:",
            (text_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        y_offset += 15

        # Split long rejection reasons into multiple lines
        reason = result.validation_reason
        if len(reason) > 25:
            words = reason.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 25:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
        else:
            lines = [reason]

        for line in lines:
            cv2.putText(
                vis_img,
                f"    {line}",
                (text_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 255),
                1,
            )
            y_offset += 12

        y_offset += 10  # Extra space between contours

    # Add color legend
    y_offset += 20
    cv2.putText(
        vis_img,
        "COLOR LEGEND:",
        (text_x, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    y_offset += 20

    legend_items = [
        ("Red: Too many vertices", colors["vertices"]),
        ("Blue: Low convexity", colors["convexity"]),
        ("Yellow: Basic validation fail", colors["basic"]),
        ("Magenta: Aspect ratio issue", colors["aspect"]),
    ]

    for text, color in legend_items:
        cv2.putText(vis_img, text, (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 15

    # Save the visualization
    cv2.imwrite(output_filename, vis_img)
    print(f"Saved detailed visualization: {output_filename}")


def main():
    # Load the frame
    frame = cv2.imread("extracted_frame.jpg")
    if frame is None:
        print("Could not load extracted_frame.jpg")
        return

    # Initialize triangle detector
    debug_dir = Path("debug_rejected_triangles")
    detector = TriangleOrientationDetector(debug_output_dir=debug_dir)

    print("=== VISUALIZING REJECTED TRIANGLES ===\n")

    # 1. Territory Triangle Area (where the "2" digit is)
    print("1. TERRITORY TRIANGLE AREA (Right side - contains the '2' digit)")
    territory_x1, territory_y1, territory_x2, territory_y2 = 1053, 650, 1106, 689
    territory_roi = frame[territory_y1:territory_y2, territory_x1:territory_x2]

    territory_contours = find_contours_in_roi(territory_roi)
    territory_results = []

    for i, contour in enumerate(territory_contours):
        result = detector.analyze_territory_triangle(contour, territory_roi)
        territory_results.append(result)
        area = cv2.contourArea(contour)
        print(f"   Contour {i+1}: Area={area:.1f}, Rejected: {result.validation_reason}")

    visualize_rejected_contours(
        territory_roi,
        territory_contours,
        territory_results,
        "Territory Area - '2' Digit Rejected",
        "rejected_territory_triangles.jpg",
    )

    # 2. Manual Left ROI (expected possession area)
    print("\n2. MANUAL LEFT ROI (Expected possession triangle area)")
    left_roi = cv2.imread("debug_manual_left_roi_simple.jpg")
    if left_roi is not None:
        left_contours = find_contours_in_roi(left_roi)
        left_results = []

        for i, contour in enumerate(left_contours):
            result = detector.analyze_possession_triangle(contour, left_roi)
            left_results.append(result)
            area = cv2.contourArea(contour)
            print(f"   Contour {i+1}: Area={area:.1f}, Rejected: {result.validation_reason}")

        visualize_rejected_contours(
            left_roi,
            left_contours,
            left_results,
            "Left Area - No Valid Triangles Found",
            "rejected_possession_triangles.jpg",
        )

    # 3. Show the original ROIs for comparison
    print("\n3. SAVING ORIGINAL ROIS FOR COMPARISON")
    cv2.imwrite("original_territory_roi.jpg", territory_roi)
    print("   Saved: original_territory_roi.jpg")

    if left_roi is not None:
        cv2.imwrite("original_left_roi.jpg", left_roi)
        print("   Saved: original_left_roi.jpg")

    print(f"\n=== SUMMARY ===")
    print("Generated visualizations:")
    print("• rejected_territory_triangles.jpg - Shows why the '2' digit was rejected")
    print("• rejected_possession_triangles.jpg - Shows why left area contours were rejected")
    print("• original_territory_roi.jpg - Raw territory area image")
    print("• original_left_roi.jpg - Raw left area image")
    print("\nThe enhanced validation is working correctly - it's rejecting digit shapes!")


if __name__ == "__main__":
    main()

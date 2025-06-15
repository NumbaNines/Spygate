#!/usr/bin/env python3
"""
Advanced Triangle Template Extractor with Image Selection
Allows precise polygon selection with zoom and point editing capabilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.ml.template_triangle_detector import YOLOIntegratedTriangleDetector
from src.spygate.ml.triangle_orientation_detector import Direction, TriangleType


class AdvancedTriangleExtractor:
    """Advanced tool for extracting triangle templates with zoom and polygon selection."""

    def __init__(self, image_path: str = None):
        """Initialize the advanced triangle template extractor."""

        # Initialize game analyzer and detector
        game_analyzer = EnhancedGameAnalyzer()
        self.detector = YOLOIntegratedTriangleDetector(game_analyzer)

        # Load image
        if image_path:
            self.image_path = image_path
        else:
            # Ask user to specify image path
            print("📸 Available images:")
            print("   - comprehensive_hud_detections.jpg")
            print("   - correct_coordinate_regions.jpg")
            print("   - debug_manual_left_roi_simple.jpg")
            print("   - Or specify your own image path")
            print()
            self.image_path = input("Enter image filename or path: ").strip()

        # Load the image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"❌ Could not load image: {self.image_path}")
            sys.exit(1)

        print(f"✅ Loaded image: {self.image_path}")
        print(f"📐 Image size: {self.original_image.shape[1]}x{self.original_image.shape[0]}")

        # Display image for template extraction
        self.display_image = self.original_image.copy()

        # Polygon selection state
        self.polygon_points = []
        self.selected_point_idx = -1
        self.dragging = False
        self.drag_start = None
        self.current_mask = None

        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.max_zoom = 10.0
        self.min_zoom = 0.1

        # Window setup
        self.window_name = "Advanced Triangle Template Extractor"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Templates directory
        self.templates_dir = Path("templates/triangles")
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Template info
        self.template_name = ""
        self.triangle_type = None
        self.direction = None

        print("\n🎯 ADVANCED TRIANGLE TEMPLATE EXTRACTION")
        print("=" * 70)
        print("🔍 Features:")
        print("   • Zoom in/out with mouse wheel for pixel-perfect selection")
        print("   • Click around triangle edges to create precise polygon")
        print("   • Right-click to complete triangle selection")
        print("   • Automatic mask creation for clean templates")
        print("📍 What to extract:")
        print("   1. POSSESSION TRIANGLES (◄ ►):")
        print("      - Located between team scores on LEFT side of HUD")
        print("      - Points to team that HAS the ball")
        print("   2. TERRITORY TRIANGLES (▲ ▼):")
        print("      - Located on FAR RIGHT side of HUD")
        print("      - Shows field territory context")
        print("💡 Pro Tips:")
        print("   - Start with 'f' to fit image to window")
        print("   - Zoom in close to the triangle for precision")
        print("   - Click around the triangle edges in order")
        print("   - Include just the triangle shape, minimal background")

        # Zoom and display state
        self.zoom_center = None
        self.display_offset = (0, 0)

        # Window size
        self.window_width = 1200
        self.window_height = 800

        self.update_display()

        print("📋 Controls:")
        print("   🖱️  LEFT CLICK: Add polygon point OR select/drag existing point")
        print("   🖱️  RIGHT CLICK: Complete polygon selection")
        print("   ⚡ MOUSE WHEEL: Zoom in/out")
        print("   📐 SPACE: Save selected triangle template")
        print("   🔄 'r': Reset current selection")
        print("   🔍 'f': Fit image to window")
        print("   🗑️  DELETE: Remove selected point")
        print("   ❌ 'q': Quit")
        print()
        print("🎯 Triangle Selection Process:")
        print("   1. Zoom in on the triangle area")
        print("   2. Click around the triangle edges (use as many points as needed)")
        print("   3. Click existing points to select/drag them for fine-tuning")
        print("   4. Right-click to close the polygon")
        print("   5. Press SPACE to save the template")
        print()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon selection, editing, and zoom."""

        # Convert display coordinates to image coordinates
        img_x, img_y = self.display_to_image_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_valid_image_coord(img_x, img_y):
                # Check if clicking near an existing point
                clicked_point_index = self.find_nearest_point(img_x, img_y)

                if clicked_point_index >= 0:
                    # Select existing point for dragging
                    self.selected_point_idx = clicked_point_index
                    self.dragging = True
                    self.drag_start = (img_x, img_y)
                    print(
                        f"🎯 Selected point {clicked_point_index + 1}: ({self.polygon_points[clicked_point_index][0]}, {self.polygon_points[clicked_point_index][1]})"
                    )
                else:
                    # Add new point
                    self.polygon_points.append((img_x, img_y))
                    self.selected_point_idx = len(self.polygon_points) - 1
                    print(
                        f"📍 Added point {len(self.polygon_points)}: ({img_x}, {img_y}) - Total points: {len(self.polygon_points)}"
                    )

                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            if self.dragging:
                self.dragging = False
                self.drag_start = None
                print(
                    f"✅ Moved point {self.selected_point_idx + 1} to: ({self.polygon_points[self.selected_point_idx][0]}, {self.polygon_points[self.selected_point_idx][1]})"
                )

        elif event == cv2.EVENT_MOUSEMOVE:
            # Drag selected point
            if (
                self.dragging
                and self.selected_point_idx >= 0
                and self.is_valid_image_coord(img_x, img_y)
            ):
                self.polygon_points[self.selected_point_idx] = (img_x, img_y)
                self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Complete polygon selection
            if len(self.polygon_points) >= 3:
                self.complete_polygon_selection()
            else:
                print("❌ Need at least 3 points to form a polygon!")

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            if flags > 0:  # Scroll up - zoom in
                self.zoom_factor = min(self.zoom_factor * 1.2, self.max_zoom)
            else:  # Scroll down - zoom out
                self.zoom_factor = max(self.zoom_factor / 1.2, self.min_zoom)

            # Set zoom center to mouse position
            self.zoom_center = (img_x, img_y)
            self.update_display()
            print(f"🔍 Zoom: {self.zoom_factor:.2f}x")

    def find_nearest_point(self, x, y):
        """Find the nearest polygon point to the given coordinates."""
        if not self.polygon_points:
            return -1

        min_distance = float("inf")
        nearest_index = -1
        click_threshold = max(10 / self.zoom_factor, 5)  # Adjust threshold based on zoom

        for i, (px, py) in enumerate(self.polygon_points):
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if distance < min_distance and distance < click_threshold:
                min_distance = distance
                nearest_index = i

        return nearest_index

    def display_to_image_coords(self, display_x, display_y):
        """Convert display coordinates to original image coordinates."""
        # Account for display offset and zoom
        img_x = int((display_x - self.display_offset[0]) / self.zoom_factor)
        img_y = int((display_y - self.display_offset[1]) / self.zoom_factor)
        return img_x, img_y

    def image_to_display_coords(self, img_x, img_y):
        """Convert image coordinates to display coordinates."""
        display_x = int(img_x * self.zoom_factor + self.display_offset[0])
        display_y = int(img_y * self.zoom_factor + self.display_offset[1])
        return display_x, display_y

    def is_valid_image_coord(self, x, y):
        """Check if coordinates are within image bounds."""
        h, w = self.original_image.shape[:2]
        return 0 <= x < w and 0 <= y < h

    def update_display(self):
        """Update the display with current zoom and selections."""
        h, w = self.original_image.shape[:2]

        if self.zoom_center is None:
            self.zoom_center = (w // 2, h // 2)

        # Calculate the region to display
        display_w = int(self.window_width / self.zoom_factor)
        display_h = int(self.window_height / self.zoom_factor)

        # Center the view around zoom_center
        start_x = max(0, self.zoom_center[0] - display_w // 2)
        start_y = max(0, self.zoom_center[1] - display_h // 2)
        end_x = min(w, start_x + display_w)
        end_y = min(h, start_y + display_h)

        # Adjust if we're at the edges
        if end_x - start_x < display_w:
            start_x = max(0, end_x - display_w)
        if end_y - start_y < display_h:
            start_y = max(0, end_y - display_h)

        # Extract and resize the region
        region = self.original_image[start_y:end_y, start_x:end_x]

        # Resize for display
        new_w = int((end_x - start_x) * self.zoom_factor)
        new_h = int((end_y - start_y) * self.zoom_factor)

        if new_w > 0 and new_h > 0:
            self.display_image = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            self.display_image = region.copy()

        # Calculate display offset for coordinate conversion
        self.display_offset = (-start_x * self.zoom_factor, -start_y * self.zoom_factor)

        # Draw polygon points and lines
        display_copy = self.display_image.copy()

        if len(self.polygon_points) > 0:
            # Convert polygon points to display coordinates
            display_points = []
            for pt in self.polygon_points:
                disp_pt = self.image_to_display_coords(pt[0], pt[1])
                # Check if point is visible in current view
                if (
                    0 <= disp_pt[0] < display_copy.shape[1]
                    and 0 <= disp_pt[1] < display_copy.shape[0]
                ):
                    display_points.append(disp_pt)

                    # Determine point color and size
                    point_index = len(display_points) - 1
                    original_index = self.polygon_points.index(pt)

                    if original_index == self.selected_point_idx:
                        # Selected point - larger and different color
                        cv2.circle(display_copy, disp_pt, 6, (0, 0, 255), -1)  # Red
                        cv2.circle(display_copy, disp_pt, 8, (255, 255, 255), 2)  # White border
                    else:
                        # Normal point
                        cv2.circle(display_copy, disp_pt, 4, (0, 255, 0), -1)  # Green

                    # Draw point number
                    cv2.putText(
                        display_copy,
                        str(original_index + 1),
                        (disp_pt[0] + 8, disp_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_copy,
                        str(original_index + 1),
                        (disp_pt[0] + 8, disp_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

            # Draw lines between points
            if len(display_points) > 1:
                for i in range(len(display_points) - 1):
                    cv2.line(display_copy, display_points[i], display_points[i + 1], (0, 255, 0), 2)

                # Draw line from last point to first if we have 3+ points
                if len(display_points) >= 3:
                    cv2.line(display_copy, display_points[-1], display_points[0], (0, 255, 0), 2)

        # Draw current mask if available
        if self.current_mask is not None:
            # Create colored overlay
            mask_region = self.current_mask[start_y:end_y, start_x:end_x]
            if mask_region.size > 0:
                mask_resized = cv2.resize(
                    mask_region,
                    (display_copy.shape[1], display_copy.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                overlay = display_copy.copy()
                overlay[mask_resized > 0] = [0, 255, 255]  # Yellow overlay
                display_copy = cv2.addWeighted(display_copy, 0.7, overlay, 0.3, 0)

        # Add status info
        cv2.putText(
            display_copy,
            f"Zoom: {self.zoom_factor:.2f}x",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_copy,
            f"Points: {len(self.polygon_points)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.selected_point_idx >= 0:
            cv2.putText(
                display_copy,
                f"Selected: Point {self.selected_point_idx + 1}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if self.dragging:
            cv2.putText(
                display_copy,
                "DRAGGING - Move mouse to reposition",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cv2.imshow(self.window_name, display_copy)

    def complete_polygon_selection(self):
        """Complete the polygon selection and create mask."""
        if len(self.polygon_points) < 3:
            print("❌ Need at least 3 points for a triangle!")
            return

        # Create mask from polygon
        h, w = self.original_image.shape[:2]
        self.current_mask = np.zeros((h, w), dtype=np.uint8)

        # Convert points to numpy array
        points = np.array(self.polygon_points, dtype=np.int32)
        cv2.fillPoly(self.current_mask, [points], 255)

        print(f"✅ Polygon completed with {len(self.polygon_points)} points!")
        print("📝 Press SPACE to save this triangle template")

        self.update_display()

    def extract_and_save_template(self):
        """Extract the selected triangle and save as template."""
        if self.current_mask is None:
            print("❌ No triangle selected! Create a polygon first.")
            return

        # Find bounding box of the mask
        coords = np.column_stack(np.where(self.current_mask > 0))
        if len(coords) == 0:
            print("❌ Empty selection!")
            return

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Extract the region
        roi_img = self.original_image[y_min : y_max + 1, x_min : x_max + 1]
        roi_mask = self.current_mask[y_min : y_max + 1, x_min : x_max + 1]

        # Apply mask to create clean template
        gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # Create template with black background
        template = np.zeros_like(gray_roi)
        template[roi_mask > 0] = gray_roi[roi_mask > 0]

        # Show the extracted template
        template_display = cv2.resize(
            template,
            (template.shape[1] * 4, template.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Extracted Triangle Template", template_display)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Extracted Triangle Template")

        # Get template information
        self.get_template_info()

        if self.template_name and self.triangle_type and self.direction:
            # Save the template
            self.detector.save_template(
                self.template_name, template, self.direction, self.triangle_type
            )

            print(f"✅ Triangle template '{self.template_name}' saved successfully!")
            print(f"📐 Size: {template.shape[1]}x{template.shape[0]} pixels")

            # Show saved location
            template_path = self.templates_dir / f"{self.template_name}.png"
            print(f"📁 Saved to: {template_path}")

            # Reset for next template
            self.reset_selection()
        else:
            print("❌ Template information incomplete. Try again.")

    def get_template_info(self):
        """Get template classification from user."""
        print("\n🏷️  TRIANGLE TEMPLATE CLASSIFICATION")
        print("=" * 40)

        # Get template name
        while True:
            name = input("Enter template name (e.g., 'madden_possession_left'): ").strip()
            if name and name.replace("_", "").replace("-", "").isalnum():
                self.template_name = name
                break
            print("❌ Please enter a valid name (letters, numbers, underscore, hyphen only)")

        # Get triangle type
        while True:
            print("\nTriangle Type:")
            print("1. Possession (arrows ◄ ► between team scores)")
            print("2. Territory (triangles ▲ ▼ showing field position)")
            choice = input("Enter choice (1 or 2): ").strip()

            if choice == "1":
                self.triangle_type = TriangleType.POSSESSION
                break
            elif choice == "2":
                self.triangle_type = TriangleType.TERRITORY
                break
            print("❌ Please enter 1 or 2")

        # Get direction
        while True:
            if self.triangle_type == TriangleType.POSSESSION:
                print("\nPossession Direction:")
                print("1. Left (◄ - pointing left)")
                print("2. Right (► - pointing right)")
                choice = input("Enter choice (1 or 2): ").strip()

                if choice == "1":
                    self.direction = Direction.LEFT
                    break
                elif choice == "2":
                    self.direction = Direction.RIGHT
                    break
            else:  # TERRITORY
                print("\nTerritory Direction:")
                print("1. Up (▲ - in opponent's territory)")
                print("2. Down (▼ - in own territory)")
                choice = input("Enter choice (1 or 2): ").strip()

                if choice == "1":
                    self.direction = Direction.UP
                    break
                elif choice == "2":
                    self.direction = Direction.DOWN
                    break
            print("❌ Please enter 1 or 2")

        print(f"\n📋 Template Classification:")
        print(f"   Name: {self.template_name}")
        print(f"   Type: {self.triangle_type.value}")
        print(f"   Direction: {self.direction.value}")

    def reset_selection(self):
        """Reset the current selection."""
        self.polygon_points = []
        self.current_mask = None
        self.selected_point_idx = -1
        self.dragging = False
        self.drag_start = None
        self.template_name = ""
        self.triangle_type = None
        self.direction = None
        print("🔄 Selection reset. Ready for new triangle!")
        self.update_display()

    def fit_to_window(self):
        """Fit the entire image to the window."""
        h, w = self.original_image.shape[:2]
        zoom_w = self.window_width / w
        zoom_h = self.window_height / h
        self.zoom_factor = min(zoom_w, zoom_h)
        self.zoom_center = (w // 2, h // 2)
        self.update_display()
        print(f"🔍 Fit to window: {self.zoom_factor:.2f}x")

    def run(self):
        """Run the interactive template extractor."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

        print("🖱️  Ready! Use mouse wheel to zoom, click to select triangle points...")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("👋 Exiting triangle extractor...")
                break
            elif key == ord(" "):  # Space key
                if self.current_mask is not None:
                    self.extract_and_save_template()
                else:
                    print("❌ No triangle selected! Create a polygon first by clicking points.")
            elif key == ord("r"):
                print("🔄 Resetting selection...")
                self.reset_selection()
            elif key == ord("f"):
                print("🔍 Fitting image to window...")
                self.fit_to_window()
            elif key == 8 or key == 127:  # Backspace or Delete key
                if self.selected_point_idx >= 0 and len(self.polygon_points) > 0:
                    removed_point = self.polygon_points.pop(self.selected_point_idx)
                    print(
                        f"🗑️ Removed point {self.selected_point_idx + 1}: ({removed_point[0]}, {removed_point[1]})"
                    )

                    # Adjust selected index
                    if self.selected_point_idx >= len(self.polygon_points):
                        self.selected_point_idx = len(self.polygon_points) - 1

                    # Clear mask if it exists
                    if self.current_mask is not None:
                        self.current_mask = None
                        print("🔄 Polygon changed - mask cleared. Right-click to recreate.")

                    self.update_display()
                else:
                    print("❌ No point selected to delete!")

        cv2.destroyAllWindows()


def main():
    # Allow user to specify image or use interactive selection
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Could not find {image_path}")
            return

    print("🎯 ADVANCED TRIANGLE TEMPLATE EXTRACTION")
    print("=" * 70)
    print("🔍 Features:")
    print("   • Zoom in/out with mouse wheel for pixel-perfect selection")
    print("   • Click around triangle edges to create precise polygon")
    print("   • Right-click to complete triangle selection")
    print("   • Automatic mask creation for clean templates")
    print()
    print("📍 What to extract:")
    print("   1. POSSESSION TRIANGLES (◄ ►):")
    print("      - Located between team scores on LEFT side of HUD")
    print("      - Points to team that HAS the ball")
    print()
    print("   2. TERRITORY TRIANGLES (▲ ▼):")
    print("      - Located on FAR RIGHT side of HUD")
    print("      - Shows field territory context")
    print()
    print("💡 Pro Tips:")
    print("   - Start with 'f' to fit image to window")
    print("   - Zoom in close to the triangle for precision")
    print("   - Click around the triangle edges in order")
    print("   - Include just the triangle shape, minimal background")
    print()

    try:
        extractor = AdvancedTriangleExtractor(image_path)
        extractor.run()

        print("\n✅ EXTRACTION COMPLETE!")
        print("🔄 Test your new templates:")
        print("   python test_template_matching.py")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

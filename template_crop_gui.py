#!/usr/bin/env python3
"""
Simple Template Crop GUI
Click and drag to select crop area, press 's' to save, 'n' for next image.
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np


class TemplateCropGUI:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.clone = None
        self.crop_coords = []
        self.cropping = False
        self.current_file = None
        self.template_files = []
        self.current_index = 0

        # Load template files
        self.load_template_files()

    def load_template_files(self):
        """Load all template files."""
        template_dir = Path("down templates")
        if template_dir.exists():
            self.template_files = list(template_dir.glob("*.png"))
            print(f"Found {len(self.template_files)} template files")
        else:
            print("Template directory not found!")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for cropping."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_coords = [(x, y)]
            self.cropping = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                # Show live preview of crop area
                temp_image = self.clone.copy()
                cv2.rectangle(temp_image, self.crop_coords[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("Template Cropper", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.crop_coords.append((x, y))
            self.cropping = False

            # Draw final rectangle
            cv2.rectangle(self.image, self.crop_coords[0], self.crop_coords[1], (0, 255, 0), 2)
            cv2.imshow("Template Cropper", self.image)

    def load_image(self, file_path):
        """Load an image for cropping."""
        self.current_file = file_path
        self.original_image = cv2.imread(str(file_path))

        if self.original_image is None:
            print(f"Failed to load {file_path}")
            return False

        # Scale image if too large
        height, width = self.original_image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200 / width, 800 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))

        self.image = self.original_image.copy()
        self.clone = self.original_image.copy()

        # Show image info
        print(f"\nLoaded: {file_path.name}")
        print(f"Size: {self.original_image.shape[1]}x{self.original_image.shape[0]}")

        return True

    def save_crop(self):
        """Save the cropped region."""
        if len(self.crop_coords) != 2:
            print("❌ No crop area selected!")
            return

        # Get crop coordinates
        x1, y1 = self.crop_coords[0]
        x2, y2 = self.crop_coords[1]

        # Ensure proper order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Crop the image
        cropped = self.original_image[y1:y2, x1:x2]

        if cropped.size == 0:
            print("❌ Invalid crop area!")
            return

        # Save the cropped image
        cv2.imwrite(str(self.current_file), cropped)
        print(f"✅ Saved cropped template: {self.current_file.name}")
        print(f"   New size: {cropped.shape[1]}x{cropped.shape[0]}")

        # Reset for next crop
        self.crop_coords = []
        self.image = self.original_image.copy()
        cv2.imshow("Template Cropper", self.image)

    def next_image(self):
        """Load next template image."""
        if not self.template_files:
            print("No template files found!")
            return

        self.current_index = (self.current_index + 1) % len(self.template_files)
        next_file = self.template_files[self.current_index]

        if self.load_image(next_file):
            cv2.imshow("Template Cropper", self.image)

    def previous_image(self):
        """Load previous template image."""
        if not self.template_files:
            print("No template files found!")
            return

        self.current_index = (self.current_index - 1) % len(self.template_files)
        prev_file = self.template_files[self.current_index]

        if self.load_image(prev_file):
            cv2.imshow("Template Cropper", self.image)

    def run(self):
        """Run the crop GUI."""
        if not self.template_files:
            print("❌ No template files found in 'down templates/'")
            return

        print("🎯 Template Crop GUI - Full Madden Screenshots")
        print("=" * 50)
        print("📸 Working with full 1920x1080 Madden screenshots")
        print("🎯 Crop the down/distance area (usually top-center)")
        print("=" * 50)
        print("Controls:")
        print("  • Click and drag to select crop area")
        print("  • 's' = Save crop")
        print("  • 'n' = Next image")
        print("  • 'p' = Previous image")
        print("  • 'r' = Reset crop")
        print("  • 'q' = Quit")
        print("=" * 50)

        # Load first image
        if self.load_image(self.template_files[0]):
            cv2.namedWindow("Template Cropper", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Template Cropper", self.mouse_callback)
            cv2.imshow("Template Cropper", self.image)

            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.save_crop()
                elif key == ord("n"):
                    self.next_image()
                elif key == ord("p"):
                    self.previous_image()
                elif key == ord("r"):
                    # Reset crop
                    self.crop_coords = []
                    self.image = self.original_image.copy()
                    cv2.imshow("Template Cropper", self.image)

        cv2.destroyAllWindows()
        print("👋 Crop GUI closed")


def main():
    """Main function."""
    cropper = TemplateCropGUI()
    cropper.run()


if __name__ == "__main__":
    main()

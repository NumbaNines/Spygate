#!/usr/bin/env python3
"""
Fixed Coordinate Helper - Text moved so it doesn't cover team scores area
"""

import cv2
import numpy as np

def create_coordinate_helper():
    """Create an image with pixel coordinates marked for easy reference."""
    
    # Load the HUD image
    image_path = "found_and_frame_3000.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    height, width = image.shape[:2]
    helper_image = image.copy()
    
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Add grid lines every 64 pixels (1/20th of 1280) for reference
    grid_spacing = 64
    grid_color = (0, 255, 255)  # Yellow
    
    # Vertical lines
    for x in range(0, width, grid_spacing):
        cv2.line(helper_image, (x, 0), (x, height), grid_color, 1)
        cv2.putText(helper_image, f"{x}", (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
    
    # Horizontal lines
    for y in range(0, height, grid_spacing):
        cv2.line(helper_image, (0, y), (width, y), grid_color, 1)
        cv2.putText(helper_image, f"{y}", (5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
    
    # Add existing down/distance region for reference
    down_coords = [0.750, 0.900, 0.200, 0.800]  # x_start, x_end, y_start, y_end
    down_x1 = int(down_coords[0] * width)
    down_x2 = int(down_coords[1] * width)
    down_y1 = int(down_coords[2] * height)
    down_y2 = int(down_coords[3] * height)
    
    cv2.rectangle(helper_image, (down_x1, down_y1), (down_x2, down_y2), (0, 0, 255), 2)
    cv2.putText(helper_image, "DOWN/DIST", (down_x1, down_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Move text to RIGHT SIDE where it won't interfere with team scores (top-left area)
    # Put info box on the right side, middle area
    info_x_start = width - 450  # Right side
    info_y_start = height // 2  # Middle vertically
    info_width = 440
    info_height = 120
    
    cv2.rectangle(helper_image, (info_x_start, info_y_start), (info_x_start + info_width, info_y_start + info_height), (0, 0, 0), -1)
    
    info_text = [
        f"Image: {width}x{height}",
        f"Down/Dist pixels: ({down_x1},{down_y1})-({down_x2},{down_y2})",
        f"Down/Dist normalized: ({down_coords[0]:.3f},{down_coords[2]:.3f})-({down_coords[1]:.3f},{down_coords[3]:.3f})",
        "Grid shows pixel coordinates",
        "Red box = working down/distance region",
        "Find team scores (top-left) & possession triangle"
    ]
    
    for i, text in enumerate(info_text):
        cv2.putText(helper_image, text, (info_x_start + 10, info_y_start + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Save the helper image
    output_path = "coordinate_helper_fixed.jpg"
    cv2.imwrite(output_path, helper_image)
    print(f"✅ Fixed coordinate helper saved: {output_path}")
    print(f"📍 Text box moved to right side - team scores area now visible!")
    
    return width, height

def convert_pixels_to_normalized(pixel_coords, width, height):
    """Convert pixel coordinates to normalized coordinates."""
    x1, y1, x2, y2 = pixel_coords
    
    normalized = [
        x1 / width,   # x_start
        x2 / width,   # x_end
        y1 / height,  # y_start
        y2 / height   # y_end
    ]
    
    return normalized

def main():
    print("🎯 FIXED COORDINATE HELPER")
    print("=" * 50)
    
    # Create the helper image
    width, height = create_coordinate_helper()
    
    print(f"\n📖 Instructions:")
    print(f"   1. Open 'coordinate_helper_fixed.jpg' in an image viewer")
    print(f"   2. Text box moved to RIGHT SIDE - team scores area is now clear!")
    print(f"   3. Look at yellow grid lines and pixel numbers")
    print(f"   4. Red box shows working down/distance region")
    print(f"   5. Find pixel coordinates for team scores (top-left) and possession triangle")
    
    print(f"\n📝 Format needed: x1,y1,x2,y2 (top-left to bottom-right pixels)")
    
    print(f"\n🔍 Working down/distance region for reference:")
    down_coords = [0.750, 0.900, 0.200, 0.800]
    down_pixels = [
        int(down_coords[0] * width),  # x1
        int(down_coords[2] * height), # y1  
        int(down_coords[1] * width),  # x2
        int(down_coords[3] * height)  # y2
    ]
    print(f"   Pixels: {down_pixels[0]},{down_pixels[1]},{down_pixels[2]},{down_pixels[3]}")
    print(f"   Normalized: {down_coords[0]:.3f},{down_coords[1]:.3f},{down_coords[2]:.3f},{down_coords[3]:.3f}")

if __name__ == "__main__":
    main() 
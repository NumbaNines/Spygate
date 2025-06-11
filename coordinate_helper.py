#!/usr/bin/env python3
"""
Coordinate Helper - Shows pixel positions and converts to normalized coordinates
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
    cv2.putText(helper_image, "DOWN/DIST (working)", (down_x1, down_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add coordinate info
    info_text = [
        f"Image: {width}x{height}",
        f"Down/Dist: pixels({down_x1},{down_y1})-({down_x2},{down_y2})",
        f"Down/Dist: normalized({down_coords[0]:.3f},{down_coords[2]:.3f})-({down_coords[1]:.3f},{down_coords[3]:.3f})",
        "Tell me pixel coordinates for team scores and possession regions!"
    ]
    
    # Black background for text
    cv2.rectangle(helper_image, (5, height-100), (800, height-5), (0, 0, 0), -1)
    
    for i, text in enumerate(info_text):
        cv2.putText(helper_image, text, (10, height - 80 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save the helper image
    output_path = "coordinate_helper.jpg"
    cv2.imwrite(output_path, helper_image)
    print(f"✅ Coordinate helper saved: {output_path}")
    
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
    print("🎯 COORDINATE HELPER")
    print("=" * 50)
    
    # Create the helper image
    width, height = create_coordinate_helper()
    
    print(f"\n📖 Instructions:")
    print(f"   1. Open 'coordinate_helper.jpg' in an image viewer")
    print(f"   2. Look at the grid lines and pixel numbers")
    print(f"   3. Note the red box shows the working down/distance region")
    print(f"   4. Find pixel coordinates for team scores and possession regions")
    print(f"   5. Come back and tell me the coordinates!")
    
    print(f"\n📝 Format needed: x1,y1,x2,y2 (top-left to bottom-right pixels)")
    print(f"📝 Example: For down/distance region = 960,144,1152,576")
    
    print(f"\n🔍 Current working down/distance region:")
    down_coords = [0.750, 0.900, 0.200, 0.800]
    down_pixels = [
        int(down_coords[0] * width),  # x1
        int(down_coords[2] * height), # y1  
        int(down_coords[1] * width),  # x2
        int(down_coords[3] * height)  # y2
    ]
    print(f"   Pixels: {down_pixels[0]},{down_pixels[1]},{down_pixels[2]},{down_pixels[3]}")
    print(f"   Normalized: {down_coords[0]:.3f},{down_coords[2]:.3f},{down_coords[1]:.3f},{down_coords[3]:.3f}")
    
    # Interactive coordinate conversion
    print(f"\n" + "="*50)
    print(f"COORDINATE CONVERTER")
    print(f"Enter pixel coordinates when ready (format: x1,y1,x2,y2)")
    print(f"Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nEnter coordinates: ").strip()
            if user_input.lower() == 'quit':
                break
                
            # Parse coordinates
            coords = [int(x.strip()) for x in user_input.split(',')]
            if len(coords) != 4:
                print("❌ Need exactly 4 coordinates: x1,y1,x2,y2")
                continue
            
            # Convert to normalized
            normalized = convert_pixels_to_normalized(coords, width, height)
            
            print(f"✅ Pixel coordinates: {coords[0]},{coords[1]},{coords[2]},{coords[3]}")
            print(f"✅ Normalized: {normalized[0]:.3f},{normalized[2]:.3f},{normalized[1]:.3f},{normalized[3]:.3f}")
            print(f"✅ Format for code: {normalized[0]:.3f}, {normalized[1]:.3f}, {normalized[2]:.3f}, {normalized[3]:.3f}")
            
        except ValueError:
            print("❌ Invalid format. Use: x1,y1,x2,y2 (numbers separated by commas)")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
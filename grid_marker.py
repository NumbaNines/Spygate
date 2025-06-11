import cv2
import os
import numpy as np

def create_grid_overlay():
    """Create a grid overlay on the HUD image for coordinate marking."""
    
    image_path = "debug_hud_regions/hud_original_frame_0.jpg"
    
    # Check if debug image exists
    if not os.path.exists(image_path):
        print(f"❌ Debug image not found: {image_path}")
        print("Please run the main app first to generate debug images!")
        return
        
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
        
    height, width = image.shape[:2]
    
    # Create a copy for grid overlay
    grid_image = image.copy()
    
    # Draw grid lines (10x10 grid)
    grid_size = 10
    
    # Vertical lines
    for i in range(1, grid_size):
        x = int(width * i / grid_size)
        cv2.line(grid_image, (x, 0), (x, height), (0, 255, 255), 1)
        
    # Horizontal lines  
    for i in range(1, grid_size):
        y = int(height * i / grid_size)
        cv2.line(grid_image, (0, y), (width, y), (0, 255, 255), 1)
        
    # Add grid numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 255, 255)
    thickness = 1
    
    # Add column numbers (0-9) at top
    for col in range(grid_size):
        x = int(width * (col + 0.5) / grid_size)
        cv2.putText(grid_image, str(col), (x-5, 15), font, font_scale, font_color, thickness)
        
    # Add row numbers (0-9) at left
    for row in range(grid_size):
        y = int(height * (row + 0.5) / grid_size)
        cv2.putText(grid_image, str(row), (5, y+5), font, font_scale, font_color, thickness)
        
    # Save the grid overlay image
    output_path = "hud_with_grid.jpg"
    cv2.imwrite(output_path, grid_image)
    
    print("🎯 Grid Coordinate Helper")
    print("=" * 40)
    print(f"✅ Grid overlay saved as: {output_path}")
    print(f"📐 Image size: {width} x {height} pixels")
    print(f"📊 Grid: 10x10 sections")
    print()
    print("Instructions:")
    print("1. Open 'hud_with_grid.jpg' in any image viewer")
    print("2. Find the '1st & 10' text in the image")
    print("3. Note which grid cells it spans")
    print("4. Tell me coordinates like: 'from column 6 row 3 to column 8 row 4'")
    print()
    print("Grid Layout:")
    print("   0 1 2 3 4 5 6 7 8 9  (columns)")
    print("0  +−+−+−+−+−+−+−+−+−+")
    print("1  +−+−+−+−+−+−+−+−+−+") 
    print("2  +−+−+−+−+−+−+−+−+−+")
    print("3  +−+−+−+−+−+−+−+−+−+")
    print("4  +−+−+−+−+−+−+−+−+−+")
    print("5  +−+−+−+−+−+−+−+−+−+")
    print("6  +−+−+−+−+−+−+−+−+−+")
    print("7  +−+−+−+−+−+−+−+−+−+")
    print("8  +−+−+−+−+−+−+−+−+−+")
    print("9  +−+−+−+−+−+−+−+−+−+")
    print("(rows)")
    
    # Also create coordinate reference
    print(f"\n📋 Coordinate Reference:")
    print(f"Each grid cell represents:")
    print(f"- Width: {width//grid_size} pixels ({100/grid_size:.1f}% of image)")
    print(f"- Height: {height//grid_size} pixels ({100/grid_size:.1f}% of image)")

def convert_grid_to_coordinates(col_start, row_start, col_end, row_end):
    """Convert grid coordinates to normalized coordinates."""
    grid_size = 10
    
    x_start = col_start / grid_size
    x_end = col_end / grid_size  
    y_start = row_start / grid_size
    y_end = row_end / grid_size
    
    print(f"\n🎯 Grid coordinates converted:")
    print(f"Grid: Column {col_start}-{col_end}, Row {row_start}-{row_end}")
    print(f"Normalized coordinates:")
    print(f"x_start = {x_start:.3f}")
    print(f"x_end = {x_end:.3f}")
    print(f"y_start = {y_start:.3f}")
    print(f"y_end = {y_end:.3f}")
    
    # Save coordinates
    with open("marked_coordinates.txt", "w") as f:
        f.write(f"# Coordinates for 1st & 10 text detection\n")
        f.write(f"# Grid: Column {col_start}-{col_end}, Row {row_start}-{row_end}\n")
        f.write(f"x_start = {x_start:.3f}\n")
        f.write(f"x_end = {x_end:.3f}\n")
        f.write(f"y_start = {y_start:.3f}\n")
        f.write(f"y_end = {y_end:.3f}\n")
    
    print(f"💾 Coordinates saved to 'marked_coordinates.txt'")

if __name__ == "__main__":
    create_grid_overlay()
    
    # Example of how to use grid coordinates
    print(f"\n💡 Example: If '1st & 10' spans from column 6 row 3 to column 8 row 4:")
    print(f"Run: convert_grid_to_coordinates(6, 3, 8, 4)") 
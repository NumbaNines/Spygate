import cv2
import os
import numpy as np

def create_better_grid():
    """Create a more readable grid overlay for the wide HUD image."""
    
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
    print(f"📐 Image dimensions: {width} x {height} pixels")
    
    # Scale up the image for better visibility (3x height)
    scale_factor = 3
    scaled_height = height * scale_factor
    scaled_image = cv2.resize(image, (width, scaled_height), interpolation=cv2.INTER_CUBIC)
    
    # Create a copy for grid overlay
    grid_image = scaled_image.copy()
    
    # Use fewer vertical divisions since the image is very thin
    h_divisions = 20  # Horizontal divisions (columns)
    v_divisions = 3   # Vertical divisions (rows) - since image is very short
    
    # Draw vertical lines (columns)
    for i in range(1, h_divisions):
        x = int(width * i / h_divisions)
        cv2.line(grid_image, (x, 0), (x, scaled_height), (0, 255, 255), 2)
        
    # Draw horizontal lines (rows)
    for i in range(1, v_divisions):
        y = int(scaled_height * i / v_divisions)
        cv2.line(grid_image, (0, y), (width, y), (0, 255, 255), 2)
        
    # Add larger, more readable numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 255, 255)
    thickness = 2
    
    # Add column numbers (0-19) at top and bottom for better visibility
    for col in range(h_divisions):
        x = int(width * (col + 0.5) / h_divisions)
        # Top numbers
        cv2.putText(grid_image, str(col), (x-10, 25), font, font_scale, font_color, thickness)
        # Bottom numbers  
        cv2.putText(grid_image, str(col), (x-10, scaled_height-10), font, font_scale, font_color, thickness)
        
    # Add row numbers (0-2) at left and right
    for row in range(v_divisions):
        y = int(scaled_height * (row + 0.5) / v_divisions)
        # Left numbers
        cv2.putText(grid_image, str(row), (5, y+10), font, font_scale, font_color, thickness)
        # Right numbers
        cv2.putText(grid_image, str(row), (width-25, y+10), font, font_scale, font_color, thickness)
    
    # Save the grid overlay image
    output_path = "hud_with_better_grid.jpg"
    cv2.imwrite(output_path, grid_image)
    
    print("🎯 Better Grid Coordinate Helper")
    print("=" * 50)
    print(f"✅ Improved grid overlay saved as: {output_path}")
    print(f"📊 Grid: {h_divisions} columns x {v_divisions} rows")
    print(f"🔍 Image scaled 3x taller for better visibility")
    print()
    print("Instructions:")
    print("1. Open 'hud_with_better_grid.jpg' in any image viewer")
    print("2. Find the '1st & 10' text in the image")
    print("3. Note which columns it spans (0-19)")
    print("4. Note which row it's in (0-2)")
    print("5. Tell me like: 'columns 12 to 15, row 1'")
    print()
    print("Grid Layout:")
    print("Columns: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19")
    print("Rows:    0 (top third)")
    print("         1 (middle third)")  
    print("         2 (bottom third)")
    
    return h_divisions, v_divisions

def convert_better_grid_to_coordinates(col_start, col_end, row_start, row_end, h_divisions=20, v_divisions=3):
    """Convert better grid coordinates to normalized coordinates."""
    
    x_start = col_start / h_divisions
    x_end = col_end / h_divisions  
    y_start = row_start / v_divisions
    y_end = row_end / v_divisions
    
    print(f"\n🎯 Grid coordinates converted:")
    print(f"Grid: Columns {col_start}-{col_end}, Rows {row_start}-{row_end}")
    print(f"Normalized coordinates:")
    print(f"x_start = {x_start:.3f}")
    print(f"x_end = {x_end:.3f}")
    print(f"y_start = {y_start:.3f}")
    print(f"y_end = {y_end:.3f}")
    
    # Save coordinates
    with open("marked_coordinates.txt", "w") as f:
        f.write(f"# Coordinates for 1st & 10 text detection\n")
        f.write(f"# Grid: Columns {col_start}-{col_end}, Rows {row_start}-{row_end}\n")
        f.write(f"x_start = {x_start:.3f}\n")
        f.write(f"x_end = {x_end:.3f}\n")
        f.write(f"y_start = {y_start:.3f}\n")
        f.write(f"y_end = {y_end:.3f}\n")
    
    print(f"💾 Coordinates saved to 'marked_coordinates.txt'")
    return x_start, x_end, y_start, y_end

if __name__ == "__main__":
    h_div, v_div = create_better_grid()
    
    print(f"\n💡 Example usage:")
    print(f"If '1st & 10' spans columns 12-15 and is in row 1:")
    print(f"Run in Python: convert_better_grid_to_coordinates(12, 15, 1, 1)")
    print(f"\nNote: row_end is typically the same as row_start for text on one line") 
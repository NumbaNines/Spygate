import cv2
import os
import numpy as np

def extract_specific_region():
    """Extract the specific region identified by the user for confirmation."""
    
    image_path = "debug_hud_regions/hud_original_frame_0.jpg"
    
    # Check if debug image exists
    if not os.path.exists(image_path):
        print(f"❌ Debug image not found: {image_path}")
        return
        
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
        
    height, width = image.shape[:2]
    print(f"📐 Original HUD dimensions: {width} x {height} pixels")
    
    # Grid settings (same as better_grid_marker.py)
    h_divisions = 20  # Columns 0-19
    v_divisions = 3   # Rows 0-2
    
    # User specified coordinates
    col_start = 15
    col_end = 17  
    row_start = 0
    row_end = 2
    
    # Convert grid coordinates to pixel coordinates
    x_start = int(width * col_start / h_divisions)
    x_end = int(width * (col_end + 1) / h_divisions)  # +1 to include the end column
    y_start = int(height * row_start / v_divisions)
    y_end = int(height * (row_end + 1) / v_divisions)  # +1 to include the end row
    
    print(f"🎯 Extracting region:")
    print(f"Grid: Columns {col_start}-{col_end}, Rows {row_start}-{row_end}")
    print(f"Pixels: x={x_start}-{x_end}, y={y_start}-{y_end}")
    print(f"Region size: {x_end-x_start} x {y_end-y_start} pixels")
    
    # Extract the region
    extracted_region = image[y_start:y_end, x_start:x_end]
    
    # Scale it up significantly for better visibility (10x)
    scale_factor = 10
    scaled_region = cv2.resize(extracted_region, 
                              (extracted_region.shape[1] * scale_factor, 
                               extracted_region.shape[0] * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
    
    # Save the extracted region
    output_path = "extracted_first_down_region.jpg"
    cv2.imwrite(output_path, scaled_region)
    
    # Also save the original size version
    output_path_orig = "extracted_first_down_region_original.jpg"
    cv2.imwrite(output_path_orig, extracted_region)
    
    print(f"✅ Extracted region saved as:")
    print(f"   📸 {output_path} (10x scaled for visibility)")
    print(f"   📸 {output_path_orig} (original size)")
    print()
    print(f"🔍 Please check '{output_path}' to confirm this shows the '1st & 10' text clearly!")
    
    # Also create a preview with the region highlighted in the original HUD
    preview_image = image.copy()
    cv2.rectangle(preview_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # Scale up the preview for visibility
    preview_scaled = cv2.resize(preview_image, (width, height * 3), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("hud_with_highlighted_region.jpg", preview_scaled)
    
    print(f"📋 Also saved: 'hud_with_highlighted_region.jpg' showing the selected area in context")
    
    # Return the normalized coordinates for the detection system
    x_norm_start = col_start / h_divisions
    x_norm_end = (col_end + 1) / h_divisions
    y_norm_start = row_start / v_divisions  
    y_norm_end = (row_end + 1) / v_divisions
    
    print(f"\n🎯 Normalized coordinates for detection:")
    print(f"x_start = {x_norm_start:.3f}")
    print(f"x_end = {x_norm_end:.3f}")
    print(f"y_start = {y_norm_start:.3f}")
    print(f"y_end = {y_norm_end:.3f}")
    
    return x_norm_start, x_norm_end, y_norm_start, y_norm_end

if __name__ == "__main__":
    coords = extract_specific_region()
    print("\n✅ If the extracted region looks correct, I'll update the detection system!") 
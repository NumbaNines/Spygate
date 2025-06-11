#!/usr/bin/env python3
"""
Manual region selector - Click and drag to select coordinate regions
"""

import cv2
import numpy as np

class RegionSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.height, self.width = self.image.shape[:2]
        self.display_image = self.image.copy()
        
        # Region selection state
        self.selecting = False
        self.start_point = None
        self.current_point = None
        self.regions = {}
        self.current_region = None
        
        # Define regions to select
        self.region_names = ["team_scores", "possession_triangle"]
        self.region_colors = [(0, 255, 0), (255, 0, 0)]  # Green, Blue
        self.region_index = 0
        
        print(f"📐 Image loaded: {self.width}x{self.height}")
        print(f"\n🎯 MANUAL REGION SELECTION")
        print(f"📖 Instructions:")
        print(f"   1. First select TEAM SCORES region (GREEN)")
        print(f"   2. Then select POSSESSION TRIANGLE region (BLUE)")
        print(f"   3. Click and drag to select each region")
        print(f"   4. Press SPACE to confirm current region")
        print(f"   5. Press 'r' to reset current region")
        print(f"   6. Press 'q' when done")
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.region_index >= len(self.region_names):
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.current_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.current_point = (x, y)
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            if self.start_point and self.current_point:
                # Store the region
                region_name = self.region_names[self.region_index]
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                
                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Convert to normalized coordinates (same format as working down/distance)
                norm_coords = {
                    'x_start': x1 / self.width,
                    'x_end': x2 / self.width,
                    'y_start': y1 / self.height,
                    'y_end': y2 / self.height
                }
                
                self.regions[region_name] = {
                    'pixels': (x1, y1, x2, y2),
                    'normalized': norm_coords
                }
                
                print(f"\n✅ {region_name.upper()} region selected:")
                print(f"   Pixels: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"   Normalized: {norm_coords['x_start']:.3f}, {norm_coords['x_end']:.3f}, {norm_coords['y_start']:.3f}, {norm_coords['y_end']:.3f}")
                print(f"   Press SPACE to confirm, 'r' to redo")
                
    def update_display(self):
        self.display_image = self.image.copy()
        
        # Draw existing confirmed regions
        for i, (name, data) in enumerate(self.regions.items()):
            x1, y1, x2, y2 = data['pixels']
            color = self.region_colors[i]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.display_image, name.upper(), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw current selection
        if self.selecting and self.start_point and self.current_point:
            color = self.region_colors[self.region_index] if self.region_index < len(self.region_colors) else (255, 255, 255)
            cv2.rectangle(self.display_image, self.start_point, self.current_point, color, 2)
            
        # Show current region being selected
        if self.region_index < len(self.region_names):
            current_region_name = self.region_names[self.region_index]
            cv2.putText(self.display_image, f"Select: {current_region_name.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Manual Region Selector", self.display_image)
    
    def run(self):
        cv2.namedWindow("Manual Region Selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Manual Region Selector", self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to confirm region
                if self.region_index < len(self.region_names):
                    current_region = self.region_names[self.region_index]
                    if current_region in self.regions:
                        print(f"✅ {current_region.upper()} confirmed!")
                        self.region_index += 1
                        if self.region_index >= len(self.region_names):
                            print(f"\n🎉 All regions selected! Press 'q' to finish.")
                        self.update_display()
            elif key == ord('r'):  # Reset current region
                if self.region_index < len(self.region_names):
                    current_region = self.region_names[self.region_index]
                    if current_region in self.regions:
                        del self.regions[current_region]
                        print(f"🔄 {current_region.upper()} reset. Select again.")
                        self.update_display()
        
        cv2.destroyAllWindows()
        return self.regions
    
    def save_coordinates(self, regions):
        """Save the selected coordinates in the working format."""
        print(f"\n📋 FINAL COORDINATES (format: x_start, x_end, y_start, y_end):")
        
        coord_file_lines = ["# Manual region selection results\n"]
        
        for region_name, data in regions.items():
            coords = data['normalized']
            coord_line = f"{coords['x_start']:.3f}, {coords['x_end']:.3f}, {coords['y_start']:.3f}, {coords['y_end']:.3f}"
            print(f"   {region_name}: {coord_line}")
            coord_file_lines.append(f"{region_name}: {coord_line}\n")
        
        # Save to file
        with open("manual_coordinates.txt", "w") as f:
            f.writelines(coord_file_lines)
        
        print(f"\n✅ Coordinates saved to: manual_coordinates.txt")
        return regions

def main():
    # Use the HUD image
    image_path = "found_and_frame_3000.png"
    
    try:
        selector = RegionSelector(image_path)
        regions = selector.run()
        
        if len(regions) == 2:
            selector.save_coordinates(regions)
            print(f"\n🎯 SUCCESS! You can now use these coordinates in the detection code.")
        else:
            print(f"\n⚠️ Only {len(regions)} regions selected. Need both team_scores and possession_triangle.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
"""
Triangle Template Extractor for SpygateAI

Interactive tool to help you cut out real triangle templates from your HUD image.
This creates precise templates for the template matching system.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.template_triangle_detector import TemplateTriangleDetector
from src.spygate.ml.triangle_orientation_detector import TriangleType, Direction

class TriangleTemplateExtractor:
    """Interactive tool for extracting triangle templates from HUD images."""
    
    def __init__(self, hud_image_path: str):
        self.hud_image = cv2.imread(hud_image_path)
        if self.hud_image is None:
            raise ValueError(f"Could not load image: {hud_image_path}")
        
        self.original_image = self.hud_image.copy()
        self.detector = TemplateTriangleDetector()
        
        # ROI selection state
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_roi = None
        
        # Template info
        self.template_name = ""
        self.triangle_type = None
        self.direction = None
        
        print("🎯 TRIANGLE TEMPLATE EXTRACTOR")
        print("=" * 50)
        print("📋 Instructions:")
        print("1. Click and drag to select a triangle region")
        print("2. Press SPACE to confirm selection")
        print("3. Follow prompts to name and classify the triangle")
        print("4. Press 'q' to quit")
        print("5. Press 'r' to reset selection")
        print()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)
            self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            self.finalize_selection()
    
    def update_display(self):
        """Update the display with current selection."""
        display_img = self.original_image.copy()
        
        if self.start_point and self.end_point:
            # Draw selection rectangle
            cv2.rectangle(display_img, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Show coordinates
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            cv2.putText(display_img, f"({x1},{y1}) to ({x2},{y2})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Triangle Template Extractor", display_img)
    
    def finalize_selection(self):
        """Finalize the current selection."""
        if self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Check minimum size
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                print("❌ Selection too small! Please select a larger area.")
                return
            
            self.current_roi = (x1, y1, x2 - x1, y2 - y1)  # x, y, w, h
            
            print(f"✅ Selected ROI: {self.current_roi}")
            print("📝 Press SPACE to save this template, or continue selecting...")
    
    def extract_and_save_template(self):
        """Extract the selected region and save as template."""
        if not self.current_roi:
            print("❌ No ROI selected!")
            return
        
        x, y, w, h = self.current_roi
        
        # Extract the region
        roi_img = self.original_image[y:y+h, x:x+w]
        
        # Show the extracted region
        cv2.imshow("Extracted Template", roi_img)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyWindow("Extracted Template")
        
        # Get template information from user
        self.get_template_info()
        
        if self.template_name and self.triangle_type and self.direction:
            # Convert to grayscale for template matching
            gray_template = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            
            # Save the template
            self.detector.save_template(
                self.template_name, 
                gray_template, 
                self.direction, 
                self.triangle_type
            )
            
            print(f"✅ Template '{self.template_name}' saved successfully!")
            
            # Show what was saved
            templates_dir = Path("templates/triangles")
            template_path = templates_dir / f"{self.template_name}.png"
            print(f"📁 Saved to: {template_path}")
            
            # Reset for next template
            self.reset_selection()
        else:
            print("❌ Template information incomplete. Try again.")
    
    def get_template_info(self):
        """Get template classification from user."""
        print("\n🏷️  TEMPLATE CLASSIFICATION")
        print("=" * 30)
        
        # Get template name
        while True:
            name = input("Enter template name (e.g., 'madden_possession_left'): ").strip()
            if name and name.replace('_', '').replace('-', '').isalnum():
                self.template_name = name
                break
            print("❌ Please enter a valid name (letters, numbers, underscore, hyphen only)")
        
        # Get triangle type
        while True:
            print("\nTriangle Type:")
            print("1. Possession (arrows between team scores)")
            print("2. Territory (triangles showing field position)")
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
                print("1. Left (◄)")
                print("2. Right (►)")
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
        
        print(f"\n📋 Template Info:")
        print(f"   Name: {self.template_name}")
        print(f"   Type: {self.triangle_type.value}")
        print(f"   Direction: {self.direction.value}")
    
    def reset_selection(self):
        """Reset the current selection."""
        self.start_point = None
        self.end_point = None
        self.current_roi = None
        self.template_name = ""
        self.triangle_type = None
        self.direction = None
        self.update_display()
    
    def run(self):
        """Run the interactive template extractor."""
        cv2.namedWindow("Triangle Template Extractor")
        cv2.setMouseCallback("Triangle Template Extractor", self.mouse_callback)
        
        self.update_display()
        
        print("🖱️  Ready! Click and drag to select triangles...")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Exiting template extractor...")
                break
            elif key == ord(' '):  # Space key
                if self.current_roi:
                    self.extract_and_save_template()
                else:
                    print("❌ No ROI selected! Click and drag to select an area first.")
            elif key == ord('r'):
                print("🔄 Resetting selection...")
                self.reset_selection()
        
        cv2.destroyAllWindows()

def main():
    # Check if HUD image exists
    hud_image_path = "extracted_frame.jpg"
    if not os.path.exists(hud_image_path):
        print(f"❌ Could not find {hud_image_path}")
        print("💡 Please ensure you have a HUD image to extract templates from.")
        return
    
    print("🎯 TRIANGLE TEMPLATE EXTRACTION GUIDE")
    print("=" * 60)
    print("📍 What to extract:")
    print("   1. POSSESSION TRIANGLES (◄ ►):")
    print("      - Located between team scores on LEFT side of HUD")
    print("      - Points to team that HAS the ball")
    print("      - Extract both left-pointing and right-pointing")
    print()
    print("   2. TERRITORY TRIANGLES (▲ ▼):")
    print("      - Located on FAR RIGHT side of HUD")
    print("      - ▲ = In opponent's territory")
    print("      - ▼ = In own territory")
    print()
    print("💡 Tips:")
    print("   - Select tightly around each triangle")
    print("   - Include a few pixels of background")
    print("   - Extract multiple examples if triangles vary")
    print("   - Start with the clearest, most visible triangles")
    print()
    
    try:
        extractor = TriangleTemplateExtractor(hud_image_path)
        extractor.run()
        
        print("\n✅ EXTRACTION COMPLETE!")
        print("🔄 Now run the template matching test again:")
        print("   python test_template_matching.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Interactive script to add triangle annotations to perfect images.
"""

import cv2
import numpy as np
from pathlib import Path
import json

class TriangleAnnotator:
    """Interactive annotator for adding triangle annotations."""
    
    def __init__(self):
        self.current_image = None
        self.current_annotations = []
        self.image_path = None
        self.drawing = False
        self.current_class = 6  # Start with possession_indicator
        
        # Class definitions
        self.classes = {
            6: "possession_indicator",  # LEFT side triangle
            7: "territory_indicator"    # RIGHT side triangle  
        }
        
        # Colors for visualization
        self.colors = {
            6: (0, 255, 0),    # Green for possession
            7: (0, 0, 255)     # Red for territory
        }
        
    def load_existing_annotations(self, label_path):
        """Load existing annotations from file."""
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append([class_id, x_center, y_center, width, height])
        return annotations
    
    def save_annotations(self, label_path, annotations):
        """Save annotations to file."""
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    def draw_annotations(self, image, annotations):
        """Draw existing annotations on image."""
        h, w = image.shape[:2]
        display_img = image.copy()
        
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            width_px = int(width * w)
            height_px = int(height * h)
            
            x1 = x_center_px - width_px // 2
            y1 = y_center_px - height_px // 2
            x2 = x_center_px + width_px // 2
            y2 = y_center_px + height_px // 2
            
            # Choose color based on class
            if class_id in self.colors:
                color = self.colors[class_id]
                class_name = self.classes.get(class_id, f"class_{class_id}")
            else:
                color = (255, 255, 255)  # White for other classes
                class_name = f"class_{class_id}"
            
            # Draw rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(display_img, f"{class_name}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Show preview rectangle
                temp_img = self.display_image.copy()
                cv2.rectangle(temp_img, self.start_point, (x, y), 
                            self.colors[self.current_class], 2)
                cv2.imshow('Annotate Triangles', temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Calculate bounding box
                x1, y1 = self.start_point
                x2, y2 = end_point
                
                # Ensure x1,y1 is top-left
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Convert to normalized coordinates
                h, w = self.current_image.shape[:2]
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Add annotation
                self.current_annotations.append([
                    self.current_class, x_center, y_center, width, height
                ])
                
                print(f"✅ Added {self.classes[self.current_class]} annotation")
                
                # Redraw
                self.display_image = self.draw_annotations(self.current_image, self.current_annotations)
                cv2.imshow('Annotate Triangles', self.display_image)
    
    def annotate_image(self, image_path):
        """Annotate a single image."""
        self.image_path = Path(image_path)
        label_path = Path(str(image_path).replace('.png', '.txt').replace('images', 'labels'))
        
        print(f"\n📸 Annotating: {self.image_path.name}")
        print(f"📄 Label file: {label_path}")
        
        # Load image
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"❌ Could not load image: {image_path}")
            return
            
        # Load existing annotations
        self.current_annotations = self.load_existing_annotations(label_path)
        print(f"📊 Loaded {len(self.current_annotations)} existing annotations")
        
        # Create display image
        self.display_image = self.draw_annotations(self.current_image, self.current_annotations)
        
        # Setup window
        cv2.namedWindow('Annotate Triangles', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotate Triangles', self.mouse_callback)
        
        print("\n🎯 TRIANGLE ANNOTATION INSTRUCTIONS:")
        print("=" * 50)
        print("🔴 POSSESSION INDICATOR (Class 6 - Green box):")
        print("   → LEFT side of HUD between team abbreviations")
        print("   → Triangle pointing to team that HAS the ball")
        print()
        print("🔵 TERRITORY INDICATOR (Class 7 - Red box):")  
        print("   → FAR RIGHT side of HUD next to yard line")
        print("   → ▲ = in opponent territory, ▼ = in own territory")
        print()
        print("⌨️  CONTROLS:")
        print("   6 = Switch to possession_indicator (Green)")
        print("   7 = Switch to territory_indicator (Red)")
        print("   u = Undo last annotation")
        print("   s = Save and continue")
        print("   q = Quit without saving")
        print("   ESC = Save and exit")
        
        while True:
            cv2.imshow('Annotate Triangles', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('6'):
                self.current_class = 6
                print(f"🟢 Switched to {self.classes[6]} (Green boxes)")
                
            elif key == ord('7'):
                self.current_class = 7
                print(f"🔴 Switched to {self.classes[7]} (Red boxes)")
                
            elif key == ord('u'):  # Undo
                if self.current_annotations:
                    removed = self.current_annotations.pop()
                    print(f"↩️  Undid {self.classes.get(removed[0], f'class_{removed[0]}')} annotation")
                    self.display_image = self.draw_annotations(self.current_image, self.current_annotations)
                    
            elif key == ord('s'):  # Save and continue
                self.save_annotations(label_path, self.current_annotations)
                print(f"💾 Saved {len(self.current_annotations)} annotations to {label_path}")
                break
                
            elif key == ord('q'):  # Quit without saving
                print("❌ Exiting without saving")
                break
                
            elif key == 27:  # ESC - Save and exit
                self.save_annotations(label_path, self.current_annotations)
                print(f"💾 Saved and exiting")
                cv2.destroyAllWindows()
                return True
                
        cv2.destroyAllWindows()
        return False

def main():
    """Main function to annotate triangle indicators."""
    print("🏈 Triangle Indicator Annotation Tool")
    print("=" * 40)
    
    # The 3 perfect images
    perfect_images = [
        "training_data/images/monitor3_screenshot_20250608_021042_6.png",
        "training_data/images/monitor3_screenshot_20250608_021427_50.png", 
        "training_data/images/monitor3_screenshot_20250608_021044_7.png"
    ]
    
    annotator = TriangleAnnotator()
    
    for image_path in perfect_images:
        if Path(image_path).exists():
            should_exit = annotator.annotate_image(image_path)
            if should_exit:
                break
        else:
            print(f"❌ Image not found: {image_path}")
    
    print("\n🎉 Triangle annotation session complete!")
    print("Now you can duplicate the newly annotated images and train!")

if __name__ == "__main__":
    main() 
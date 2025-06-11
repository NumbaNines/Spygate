"""
Manual annotation tool for correct triangle detection in Madden HUD.

This tool helps manually annotate the correct triangles:
- possession_indicator: Triangle on LEFT side between team abbreviations 
- territory_indicator: Triangle on FAR RIGHT side (▲/▼)

Usage:
    python create_correct_triangle_annotations.py
    
Controls:
    - Left click and drag to create bounding box
    - Press '1' for possession_indicator (class 6)
    - Press '2' for territory_indicator (class 7)
    - Press 's' to save annotations
    - Press 'n' for next image
    - Press 'q' to quit
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
import glob

class TriangleAnnotator:
    def __init__(self, image_dir="training_data", output_dir="correct_triangle_annotations"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get list of images to annotate
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(str(self.image_dir / ext)))
        
        self.current_idx = 0
        self.current_image = None
        self.original_image = None
        self.annotations = []
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_class = 6  # Default to possession_indicator
        
        # Class names
        self.class_names = {
            6: "possession_indicator",  # LEFT side triangle
            7: "territory_indicator"    # RIGHT side triangle
        }
        
        print(f"Found {len(self.image_files)} images to annotate")
        print("Controls:")
        print("  Left click & drag: Create bounding box")
        print("  Press '1': Set class to possession_indicator (LEFT triangle)")
        print("  Press '2': Set class to territory_indicator (RIGHT triangle)")
        print("  Press 's': Save annotations for current image")
        print("  Press 'n': Next image")
        print("  Press 'r': Reset current image annotations")
        print("  Press 'q': Quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Draw rectangle on copy of image
                img_copy = self.original_image.copy()
                self.draw_existing_annotations(img_copy)
                cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
                
                # Show current class
                class_name = self.class_names[self.current_class]
                cv2.putText(img_copy, f"Class: {class_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.current_image = img_copy
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Calculate bounding box
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])
                
                # Only add if box has reasonable size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    annotation = {
                        'class': self.current_class,
                        'class_name': self.class_names[self.current_class],
                        'bbox': [x1, y1, x2, y2]
                    }
                    self.annotations.append(annotation)
                    print(f"Added {self.class_names[self.current_class]} at [{x1}, {y1}, {x2}, {y2}]")
                
                self.refresh_display()
    
    def draw_existing_annotations(self, img):
        """Draw existing annotations on image"""
        for ann in self.annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_name = ann['class_name']
            
            # Different colors for different classes
            color = (255, 0, 0) if ann['class'] == 6 else (0, 0, 255)  # Blue for possession, Red for territory
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def refresh_display(self):
        """Refresh the display with current annotations"""
        self.current_image = self.original_image.copy()
        self.draw_existing_annotations(self.current_image)
        
        # Show current class
        class_name = self.class_names[self.current_class]
        cv2.putText(self.current_image, f"Current Class: {class_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show image info
        cv2.putText(self.current_image, f"Image {self.current_idx + 1}/{len(self.image_files)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show annotation count
        cv2.putText(self.current_image, f"Annotations: {len(self.annotations)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def load_image(self, idx):
        """Load image at given index"""
        if idx >= len(self.image_files):
            print("No more images to annotate!")
            return False
            
        image_path = self.image_files[idx]
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            print(f"Failed to load image: {image_path}")
            return False
            
        print(f"\nLoaded: {image_path}")
        
        # Reset annotations for new image
        self.annotations = []
        
        # Check if annotations already exist
        annotation_file = self.output_dir / f"{Path(image_path).stem}_annotations.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', [])
                print(f"Loaded existing {len(self.annotations)} annotations")
            except Exception as e:
                print(f"Failed to load existing annotations: {e}")
        
        self.refresh_display()
        return True
    
    def save_annotations(self):
        """Save annotations for current image"""
        if not self.annotations:
            print("No annotations to save")
            return
            
        image_path = self.image_files[self.current_idx]
        image_name = Path(image_path).stem
        
        # Save annotations in JSON format
        annotation_data = {
            'image_path': image_path,
            'image_name': image_name,
            'image_size': {
                'width': self.original_image.shape[1],
                'height': self.original_image.shape[0]
            },
            'annotations': self.annotations
        }
        
        annotation_file = self.output_dir / f"{image_name}_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        # Save visualization
        vis_image = self.original_image.copy()
        self.draw_existing_annotations(vis_image)
        vis_file = self.output_dir / f"{image_name}_visualization.jpg"
        cv2.imwrite(str(vis_file), vis_image)
        
        print(f"Saved {len(self.annotations)} annotations to {annotation_file}")
        print(f"Saved visualization to {vis_file}")
    
    def run(self):
        """Run the annotation tool"""
        if not self.image_files:
            print("No images found to annotate!")
            return
            
        cv2.namedWindow('Triangle Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Triangle Annotator', self.mouse_callback)
        
        # Load first image
        self.load_image(0)
        
        while True:
            cv2.imshow('Triangle Annotator', self.current_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_class = 6  # possession_indicator
                print("Selected: possession_indicator (LEFT triangle)")
                self.refresh_display()
            elif key == ord('2'):
                self.current_class = 7  # territory_indicator
                print("Selected: territory_indicator (RIGHT triangle)")
                self.refresh_display()
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('n'):
                self.current_idx += 1
                if not self.load_image(self.current_idx):
                    break
            elif key == ord('r'):
                self.annotations = []
                print("Reset annotations for current image")
                self.refresh_display()
        
        cv2.destroyAllWindows()
        print("Annotation session completed!")

def main():
    print("Triangle Annotation Tool for Madden HUD")
    print("=====================================")
    print()
    
    # Create annotator
    annotator = TriangleAnnotator()
    
    # Run annotation tool
    annotator.run()

if __name__ == "__main__":
    main() 
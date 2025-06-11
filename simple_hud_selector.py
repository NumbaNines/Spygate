import cv2
import numpy as np

class SimpleHUDSelector:
    def __init__(self):
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks and drags"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            
            # Calculate normalized coordinates
            h, w = self.image.shape[:2]
            x_start = min(self.start_point[0], self.end_point[0]) / w
            x_end = max(self.start_point[0], self.end_point[0]) / w
            y_start = min(self.start_point[1], self.end_point[1]) / h
            y_end = max(self.start_point[1], self.end_point[1]) / h
            
            print(f"\n🎯 SELECTED COORDINATES:")
            print(f"score_x_start = {x_start:.3f}")
            print(f"score_x_end = {x_end:.3f}")
            print(f"score_y_start = {y_start:.3f}")
            print(f"score_y_end = {y_end:.3f}")
    
    def select_region(self, image_path):
        """Load image and let user select region"""
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            print(f"Could not load image: {image_path}")
            return
            
        h, w = self.image.shape[:2]
        print(f"📸 Image loaded: {w}x{h} pixels")
        print(f"🖱️  Click and drag to select team scores region")
        print(f"📍 Press 'q' to quit")
        
        window_name = "Select Team Scores Region - Click and Drag"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            display = self.image.copy()
            
            # Draw current selection
            if self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.putText(display, "SELECTED REGION", 
                           (self.start_point[0], self.start_point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()

def main():
    selector = SimpleHUDSelector()
    
    print("🎮 SIMPLE HUD REGION SELECTOR")
    print("Enter the path to your Madden HUD screenshot:")
    
    image_path = input("Image path: ").strip()
    selector.select_region(image_path)

if __name__ == "__main__":
    main() 
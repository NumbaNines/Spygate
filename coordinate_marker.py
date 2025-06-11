import cv2
import os

class CoordinateMarker:
    def __init__(self):
        self.points = []
        self.image_path = "debug_hud_regions/hud_original_frame_0.jpg"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            
            # Draw the point
            cv2.circle(self.image, (x, y), 3, (0, 255, 0), -1)
            
            # If we have 2 points, draw the rectangle
            if len(self.points) == 2:
                x1, y1 = self.points[0]
                x2, y2 = self.points[1]
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate normalized coordinates (percentages)
                height, width = self.image.shape[:2]
                norm_x1 = min(x1, x2) / width
                norm_x2 = max(x1, x2) / width
                norm_y1 = min(y1, y2) / height
                norm_y2 = max(y1, y2) / height
                
                print(f"\n✅ Rectangle marked!")
                print(f"Top-left: ({min(x1, x2)}, {min(y1, y2)})")
                print(f"Bottom-right: ({max(x1, x2)}, {max(y1, y2)})")
                print(f"\n🎯 Normalized coordinates (for code):")
                print(f"x_start = {norm_x1:.3f}")
                print(f"x_end = {norm_x2:.3f}")
                print(f"y_start = {norm_y1:.3f}")
                print(f"y_end = {norm_y2:.3f}")
                
                # Save coordinates to file
                with open("marked_coordinates.txt", "w") as f:
                    f.write(f"# Coordinates for 1st & 10 text detection\n")
                    f.write(f"x_start = {norm_x1:.3f}\n")
                    f.write(f"x_end = {norm_x2:.3f}\n")
                    f.write(f"y_start = {norm_y1:.3f}\n")
                    f.write(f"y_end = {norm_y2:.3f}\n")
                    f.write(f"\n# Pixel coordinates:\n")
                    f.write(f"# Top-left: ({min(x1, x2)}, {min(y1, y2)})\n")
                    f.write(f"# Bottom-right: ({max(x1, x2)}, {max(y1, y2)})\n")
                
                print(f"\n💾 Coordinates saved to 'marked_coordinates.txt'")
                print(f"\nPress 'r' to reset and mark again, or 'q' to quit")
            
            cv2.imshow('HUD Coordinate Marker', self.image)
            
    def reset(self):
        self.points = []
        self.image = self.original_image.copy()
        cv2.imshow('HUD Coordinate Marker', self.image)
        print("\n🔄 Reset! Click two points to mark the '1st & 10' text area")
        
    def run(self):
        # Check if debug image exists
        if not os.path.exists(self.image_path):
            print(f"❌ Debug image not found: {self.image_path}")
            print("Please run the main app first to generate debug images!")
            return
            
        # Load the image
        self.original_image = cv2.imread(self.image_path)
        self.image = self.original_image.copy()
        
        if self.image is None:
            print(f"❌ Could not load image: {self.image_path}")
            return
            
        print("🎯 HUD Coordinate Marker Tool")
        print("=" * 40)
        print("Instructions:")
        print("1. Click the TOP-LEFT corner of the '1st & 10' text")
        print("2. Click the BOTTOM-RIGHT corner of the '1st & 10' text")
        print("3. Press 'r' to reset and try again")
        print("4. Press 'q' to quit")
        print("\n📍 Click two points to mark the exact '1st & 10' text area...")
        
        # Create window and set mouse callback
        cv2.namedWindow('HUD Coordinate Marker', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('HUD Coordinate Marker', self.mouse_callback)
        cv2.imshow('HUD Coordinate Marker', self.image)
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
                
        cv2.destroyAllWindows()
        print("\n👋 Coordinate marker closed!")

if __name__ == "__main__":
    marker = CoordinateMarker()
    marker.run() 
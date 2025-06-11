import cv2
import numpy as np
from ultralytics import YOLO

class HUDRegionSelector:
    def __init__(self):
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.hud_region = None
        self.hud_size = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            
            # Calculate normalized coordinates
            if self.hud_size:
                w, h = self.hud_size
                x_start = min(self.start_point[0], self.end_point[0]) / w
                x_end = max(self.start_point[0], self.end_point[0]) / w
                y_start = min(self.start_point[1], self.end_point[1]) / h
                y_end = max(self.start_point[1], self.end_point[1]) / h
                
                print(f"\n🎯 SELECTED COORDINATES:")
                print(f"   x_start = {x_start:.3f}")
                print(f"   x_end = {x_end:.3f}")
                print(f"   y_start = {y_start:.3f}")
                print(f"   y_end = {y_end:.3f}")
                print(f"\n📋 COPY THIS TO YOUR CODE:")
                print(f"   score_x_start = {x_start:.3f}")
                print(f"   score_x_end = {x_end:.3f}")
                print(f"   score_y_start = {y_start:.3f}")
                print(f"   score_y_end = {y_end:.3f}")
    
    def select_region_from_video(self, video_path, model_path, frame_number=1800):
        """Load video frame and let user select region"""
        # Load model and video
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
            
        # Try different frames if specified one doesn't work
        test_frames = [frame_number, 1800, 3600, 900, 5400]
        frame = None
        
        for test_frame in test_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
            ret, test_frame_data = cap.read()
            if ret:
                frame = test_frame_data
                print(f"Using frame {test_frame}")
                break
                
        if frame is None:
            print("Could not read any frames from video")
            cap.release()
            return
            
        # Run detection to get HUD
        results = model(frame, verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                hud_boxes = [box for box in boxes if int(box.cls[0]) == 0]
                
                if hud_boxes:
                    # Extract HUD region
                    hud_box = hud_boxes[0]
                    x1, y1, x2, y2 = hud_box.xyxy[0].cpu().numpy()
                    self.hud_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    h, w = self.hud_region.shape[:2]
                    self.hud_size = (w, h)
                    
                    print(f"🎮 HUD REGION EXTRACTED: {w}x{h} pixels")
                    print(f"\n📍 CURRENT COORDINATES:")
                    print(f"   Down & Distance: x(0.750-0.900), y(0.200-0.800)")
                    print(f"   Team Scores: x(0.200-0.700), y(0.050-0.350)")
                    print(f"\n🖱️  INSTRUCTIONS:")
                    print(f"   1. Click and drag to select the team scores/possession region")
                    print(f"   2. Press 'r' to reset selection")
                    print(f"   3. Press 'q' to quit")
                    print(f"   4. Press 's' to save current region as image")
                    
                    self.interactive_selection()
                else:
                    print("No HUD detected in frame")
            else:
                print("No objects detected in frame")
        else:
            print("Detection failed")
            
        cap.release()
    
    def select_region_from_image(self, image_path):
        """Load saved HUD image and let user select region"""
        self.hud_region = cv2.imread(image_path)
        if self.hud_region is None:
            print(f"Could not load image: {image_path}")
            return
            
        h, w = self.hud_region.shape[:2]
        self.hud_size = (w, h)
        
        print(f"🖼️  HUD IMAGE LOADED: {w}x{h} pixels")
        print(f"\n🖱️  INSTRUCTIONS:")
        print(f"   1. Click and drag to select the team scores/possession region")
        print(f"   2. Press 'r' to reset selection") 
        print(f"   3. Press 'q' to quit")
        
        self.interactive_selection()
    
    def interactive_selection(self):
        """Display HUD and handle interactive selection"""
        window_name = "HUD Region Selector - Click and Drag to Select Team Scores Area"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Scale up for better visibility
        display_scale = 3
        display_hud = cv2.resize(self.hud_region, None, fx=display_scale, fy=display_scale)
        
        while True:
            # Create display copy
            display_copy = display_hud.copy()
            
            # Draw existing regions for reference
            h, w = display_copy.shape[:2]
            
            # Draw down & distance region (red)
            dd_x1 = int(0.750 * w)
            dd_x2 = int(0.900 * w)
            dd_y1 = int(0.200 * h)
            dd_y2 = int(0.800 * h)
            cv2.rectangle(display_copy, (dd_x1, dd_y1), (dd_x2, dd_y2), (0, 0, 255), 2)
            cv2.putText(display_copy, "Down & Distance", (dd_x1, dd_y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw current team scores region (green)
            ts_x1 = int(0.200 * w)
            ts_x2 = int(0.700 * w)
            ts_y1 = int(0.050 * h)
            ts_y2 = int(0.350 * h)
            cv2.rectangle(display_copy, (ts_x1, ts_y1), (ts_x2, ts_y2), (0, 255, 0), 2)
            cv2.putText(display_copy, "Current Team Scores", (ts_x1, ts_y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw selection rectangle (blue)
            if self.start_point and self.end_point:
                scaled_start = (self.start_point[0] * display_scale, self.start_point[1] * display_scale)
                scaled_end = (self.end_point[0] * display_scale, self.end_point[1] * display_scale)
                cv2.rectangle(display_copy, scaled_start, scaled_end, (255, 0, 0), 3)
                cv2.putText(display_copy, "NEW SELECTION", 
                           (scaled_start[0], scaled_start[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            cv2.imshow(window_name, display_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.start_point = None
                self.end_point = None
                print("Selection reset")
            elif key == ord('s') and self.start_point and self.end_point:
                # Save selected region
                orig_w, orig_h = self.hud_size
                x1 = min(self.start_point[0], self.end_point[0])
                x2 = max(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                y2 = max(self.start_point[1], self.end_point[1])
                
                selected_region = self.hud_region[y1:y2, x1:x2]
                filename = "selected_team_scores_region.png"
                cv2.imwrite(filename, selected_region)
                print(f"💾 Selected region saved as: {filename}")
        
        cv2.destroyAllWindows()

def main():
    selector = HUDRegionSelector()
    
    print("🎮 HUD REGION SELECTOR")
    print("Choose input method:")
    print("1. Load from video file")
    print("2. Load from saved HUD image")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Use available video files
        video_options = ["video-720.mp4", "live_detections.mp4"]
        print("\nAvailable videos:")
        for i, video in enumerate(video_options):
            print(f"{i+1}. {video}")
        
        video_choice = input("Enter video number: ").strip()
        try:
            video_idx = int(video_choice) - 1
            video_path = video_options[video_idx]
            model_path = "../hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
            
            frame_num = input("Enter frame number (default 1800): ").strip()
            frame_num = int(frame_num) if frame_num else 1800
            
            selector.select_region_from_video(video_path, model_path, frame_num)
        except (ValueError, IndexError):
            print("Invalid choice")
            
    elif choice == "2":
        image_path = input("Enter path to HUD image: ").strip()
        selector.select_region_from_image(image_path)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 
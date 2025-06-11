import cv2
import numpy as np

def create_grid_overlay(image_path):
    """Create a grid overlay on the HUD image and let user select coordinates"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Draw grid lines every 10% (0.1 intervals)
    grid_color = (0, 255, 255)  # Yellow
    
    # Vertical lines
    for i in range(1, 10):
        x = int(w * i * 0.1)
        cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
        cv2.putText(overlay, f"{i*0.1:.1f}", (x-20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
    
    # Horizontal lines  
    for i in range(1, 10):
        y = int(h * i * 0.1)
        cv2.line(overlay, (0, y), (w, y), grid_color, 1)
        cv2.putText(overlay, f"{i*0.1:.1f}", (10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
    
    # Add coordinate labels at corners
    cv2.putText(overlay, "0.0,0.0", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, "1.0,1.0", (w-80, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Highlight existing regions
    # Down & Distance region (RED)
    dd_x1, dd_x2 = int(0.750 * w), int(0.900 * w)
    dd_y1, dd_y2 = int(0.200 * h), int(0.800 * h)
    cv2.rectangle(overlay, (dd_x1, dd_y1), (dd_x2, dd_y2), (0, 0, 255), 3)
    cv2.putText(overlay, "DOWN/DIST", (dd_x1, dd_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Current team scores region (GREEN)
    ts_x1, ts_x2 = int(0.200 * w), int(0.700 * w)
    ts_y1, ts_y2 = int(0.050 * h), int(0.350 * h)
    cv2.rectangle(overlay, (ts_x1, ts_y1), (ts_x2, ts_y2), (0, 255, 0), 2)
    cv2.putText(overlay, "CURRENT TEAM SCORES", (ts_x1, ts_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save and display
    grid_filename = f"grid_overlay_{image_path.replace('.png', '')}.png"
    cv2.imwrite(grid_filename, overlay)
    print(f"📊 Grid overlay saved as: {grid_filename}")
    
    # Display
    cv2.namedWindow("HUD Grid - Press Q to close", cv2.WINDOW_NORMAL)
    cv2.imshow("HUD Grid - Press Q to close", overlay)
    
    print(f"\n🎯 COORDINATE GRID CREATED")
    print(f"📍 Look at the image and tell me the coordinates for team scores!")
    print(f"📖 Format: x_start, x_end, y_start, y_end")
    print(f"📖 Example: 0.1, 0.6, 0.0, 0.3")
    print(f"\n🔍 CURRENT REGIONS:")
    print(f"   Down & Distance (RED): 0.750, 0.900, 0.200, 0.800")
    print(f"   Team Scores (GREEN): 0.200, 0.700, 0.050, 0.350")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Test with available HUD images
    hud_images = [
        "found_and_frame_3000.png",
        "found_and_frame_3060.png", 
        "frame_3240_analysis.png"
    ]
    
    print("🎮 HUD GRID COORDINATE SELECTOR")
    print("Available HUD images:")
    for i, img in enumerate(hud_images):
        print(f"{i+1}. {img}")
    
    choice = input("Enter image number (1-3): ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(hud_images):
            create_grid_overlay(hud_images[idx])
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    main() 
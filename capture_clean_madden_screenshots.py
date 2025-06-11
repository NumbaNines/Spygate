"""
Capture clean Madden HUD screenshots for triangle annotation.

This script captures live screenshots from the screen to create a clean dataset
of Madden HUD images that can be properly annotated with the correct triangles.

Usage:
    python capture_clean_madden_screenshots.py
    
Controls:
    - Space: Capture screenshot
    - 'q': Quit
"""

import cv2
import numpy as np
import mss
import time
from pathlib import Path
from datetime import datetime

class MaddenScreenCapture:
    def __init__(self, output_dir="clean_madden_screenshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.capture_count = 0
        
        # Screen capture setup
        self.sct = mss.mss()
        
        # Get primary monitor
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
        print(f"Screen capture initialized")
        print(f"Monitor resolution: {self.monitor['width']}x{self.monitor['height']}")
        print(f"Output directory: {self.output_dir}")
        print()
        print("Controls:")
        print("  Space: Capture screenshot")
        print("  'q': Quit")
        print()
        print("Tips for good triangle annotation:")
        print("- Capture during gameplay when HUD is visible")
        print("- Look for possession indicator (LEFT triangle between team names)")
        print("- Look for territory indicator (RIGHT triangle showing field position)")
        print("- Capture various game situations (different scores, times, etc.)")
        
    def capture_screen(self):
        """Capture current screen"""
        # Capture the screen
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert BGRA to BGR (remove alpha channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def save_screenshot(self, img):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        filename = f"madden_hud_{self.capture_count:03d}_{timestamp}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), img)
        self.capture_count += 1
        
        print(f"Saved: {filename}")
        return filepath
    
    def run(self):
        """Run the screen capture tool"""
        print("Starting screen capture... Position Madden window and press SPACE to capture")
        
        while True:
            # Capture current screen
            screen = self.capture_screen()
            
            # Resize for display (keep original resolution for saving)
            display_height = 600
            aspect_ratio = screen.shape[1] / screen.shape[0]
            display_width = int(display_height * aspect_ratio)
            display_screen = cv2.resize(screen, (display_width, display_height))
            
            # Add overlay text
            overlay = display_screen.copy()
            cv2.putText(overlay, f"Captures: {self.capture_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay, "SPACE: Capture, Q: Quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Madden Screen Capture - Live Preview', overlay)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):  # Space to capture
                self.save_screenshot(screen)
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        print(f"\nCapture session completed! Saved {self.capture_count} screenshots to {self.output_dir}")

def main():
    print("Madden HUD Screenshot Capture Tool")
    print("==================================")
    print()
    
    try:
        # Create capturer
        capturer = MaddenScreenCapture()
        
        # Run capture tool
        capturer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have mss installed: pip install mss")

if __name__ == "__main__":
    main() 
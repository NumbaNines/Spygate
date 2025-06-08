"""Process Madden video into frames for YOLO11 testing."""

import cv2
import os
from pathlib import Path
import numpy as np
from datetime import datetime

def extract_frames(video_path: str, output_dir: str, interval: int = 30):
    """Extract frames from video at specified interval.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        interval: Extract every Nth frame
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    saved_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            # Save frame
            frame_name = f"madden_frame_{timestamp}_{saved_count:04d}.png"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {frame_count} total frames")
    return saved_count

def main():
    # Setup paths
    video_path = "../dwitch_fxgr9mcd.mp4"
    output_dir = "test_dataset/images/madden_test"
    
    # Extract frames
    try:
        num_frames = extract_frames(video_path, output_dir, interval=15)  # Extract every 15th frame
        print(f"Successfully extracted {num_frames} frames to {output_dir}")
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main() 
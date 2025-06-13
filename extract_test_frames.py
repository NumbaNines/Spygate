"""
Extract frames from test videos for OCR comparison testing.
"""

import cv2
import os
from pathlib import Path

def extract_frames():
    """Extract frames from test videos."""
    # Create output directory
    output_dir = Path("tests/test_data/ocr_test_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for test videos
    video_dirs = [
        Path("data/test_videos"),
        Path("tests/test_videos"),
        Path("data/videos")
    ]
    
    videos = []
    for video_dir in video_dirs:
        if video_dir.exists():
            videos.extend(list(video_dir.glob("*.mp4")))
    
    if not videos:
        print("No test videos found!")
        return
        
    print(f"Found {len(videos)} videos")
    
    # Extract frames from each video
    frame_count = 0
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract one frame every 2 seconds
        frame_interval = fps * 2
        
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                output_path = output_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                frame_count += 1
                print(f"Saved {output_path}")
            
            frame_idx += frame_interval
            
        cap.release()
    
    print(f"\nExtracted {frame_count} frames for testing")

if __name__ == "__main__":
    extract_frames() 
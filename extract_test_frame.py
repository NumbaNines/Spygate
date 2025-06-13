import cv2
import sys
from pathlib import Path

def extract_frame(video_path: str, frame_num: int = 30):
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_num} from video")
        
    # Save frame
    cv2.imwrite("test_frame.png", frame)
    print(f"Extracted frame {frame_num} and saved as test_frame.png")
    
    cap.release()

if __name__ == "__main__":
    # Use the sample video from test_videos
    video_path = "test_videos/sample.mp4"
    extract_frame(video_path, frame_num=100)  # Extract frame 100 for better gameplay content 
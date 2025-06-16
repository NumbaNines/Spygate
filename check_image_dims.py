#!/usr/bin/env python3
"""Check dimensions of real Madden screenshots."""

import cv2
from pathlib import Path

def check_image_dimensions():
    """Check dimensions of real screenshots."""
    test_files = [
        "templates/raw_gameplay/1st_10.png",
        "templates/raw_gameplay/2nd_7.png", 
        "templates/raw_gameplay/3rd_goal.png"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            img = cv2.imread(file_path)
            if img is not None:
                height, width = img.shape[:2]
                print(f"{file_path}: {width}x{height}")
            else:
                print(f"{file_path}: Failed to load")
        else:
            print(f"{file_path}: File not found")

if __name__ == "__main__":
    check_image_dimensions() 
"""
Create a test video file with moving objects for visualization testing.
"""

from pathlib import Path

import cv2
import numpy as np

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / "test_data"
output_dir.mkdir(exist_ok=True)

# Video parameters
width = 1280
height = 720
fps = 30
duration = 10  # seconds
n_frames = fps * duration

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_dir / "test_video.mp4"), fourcc, fps, (width, height))

# Create moving objects
objects = [
    {
        "pos": np.array([100, 100], dtype=float),
        "vel": np.array([5, 3], dtype=float),
        "size": 50,
        "color": (0, 0, 255),  # Red
    },
    {
        "pos": np.array([width - 100, height - 100], dtype=float),
        "vel": np.array([-4, -2], dtype=float),
        "size": 40,
        "color": (0, 255, 0),  # Green
    },
    {
        "pos": np.array([width // 2, height // 2], dtype=float),
        "vel": np.array([3, -4], dtype=float),
        "size": 60,
        "color": (255, 0, 0),  # Blue
    },
]

print("Creating test video...")

# Generate frames
for frame_idx in range(n_frames):
    # Create blank frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Update and draw objects
    for obj in objects:
        # Update position
        obj["pos"] += obj["vel"]

        # Bounce off walls
        for i in range(2):
            if obj["pos"][i] < obj["size"] or obj["pos"][i] > ([width, height][i] - obj["size"]):
                obj["vel"][i] *= -1
                obj["pos"][i] = np.clip(
                    obj["pos"][i], obj["size"], [width, height][i] - obj["size"]
                )

        # Draw object
        cv2.circle(frame, tuple(map(int, obj["pos"])), obj["size"], obj["color"], -1)

    # Add frame number
    cv2.putText(
        frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # Write frame
    out.write(frame)

    # Show progress
    if frame_idx % fps == 0:
        print(f"Progress: {frame_idx/n_frames*100:.1f}%")

# Release video writer
out.release()
print("Test video created successfully!")

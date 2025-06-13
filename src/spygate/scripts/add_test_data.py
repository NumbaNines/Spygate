"""Script to add test data to the database."""

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from spygate.database import create_clip, get_db, init_db


def create_test_video(
    file_path: str,
    duration: int = 5,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
):
    """Create a test video file with proper metadata.

    Args:
        file_path: Path to save the video file
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    # Generate frames
    for _ in range(duration * fps):
        # Create a frame with some text and shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Test Video",
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
        )
        cv2.rectangle(frame, (100, 100), (width - 100, height - 100), (0, 255, 0), 2)
        out.write(frame)

    out.release()


def add_test_data():
    """Add test clips with different player names and tags."""
    # Initialize database
    init_db()

    # Get the data directory path
    data_dir = Path("data")
    test_clips_dir = data_dir / "test_clips"

    # Create test clips directory if it doesn't exist
    test_clips_dir.mkdir(parents=True, exist_ok=True)

    # Test data
    test_clips = [
        {
            "file_path": str(test_clips_dir / "offense_shotgun.mp4"),
            "title": "Shotgun Formation TD Pass",
            "player_name": "Self",
            "tags": ["offense", "shotgun", "touchdown", "pass"],
            "description": "Touchdown pass from shotgun formation",
            "duration": 8,
        },
        {
            "file_path": str(test_clips_dir / "defense_blitz.mp4"),
            "title": "Defensive Blitz Sack",
            "player_name": "Self",
            "tags": ["defense", "blitz", "sack"],
            "description": "Successful blitz leading to a sack",
            "duration": 6,
        },
        {
            "file_path": str(test_clips_dir / "opponent_run.mp4"),
            "title": "Opponent Run Play Analysis",
            "player_name": "Opponent: JonBeast",
            "tags": ["offense", "run", "analysis"],
            "description": "Analysis of opponent's running strategy",
            "duration": 10,
        },
        {
            "file_path": str(test_clips_dir / "opponent_defense.mp4"),
            "title": "Opponent Defense Formation",
            "player_name": "Opponent: CleffTheGod",
            "tags": ["defense", "formation", "study"],
            "description": "Study of opponent's defensive formation",
            "duration": 7,
        },
    ]

    # Create test video files
    for clip in test_clips:
        create_test_video(clip["file_path"], duration=clip["duration"])
        print(f"Created test video: {clip['file_path']}")

    # Add clips to database
    with get_db() as session:
        for clip_data in test_clips:
            # Get video metadata
            cap = cv2.VideoCapture(clip_data["file_path"])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else None
            cap.release()

            # Create clip with metadata
            create_clip(
                file_path=clip_data["file_path"],
                title=clip_data["title"],
                player_name=clip_data["player_name"],
                session=session,
                tags=clip_data["tags"],
                description=clip_data["description"],
                duration=duration,
                width=width,
                height=height,
                fps=fps,
            )
            print(f"Added test clip to database: {clip_data['title']}")


if __name__ == "__main__":
    add_test_data()

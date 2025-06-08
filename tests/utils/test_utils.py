"""Test utilities for video import functionality."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QWidget

from spygate.video.metadata import VideoMetadata


def create_test_video(
    path: str,
    duration: float = 5.0,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Create a test video file.

    Args:
        path: Path where to save the video
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height

    Returns:
        Path to the created video file
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    try:
        # Generate frames
        num_frames = int(duration * fps)
        for i in range(num_frames):
            # Create a frame with some content
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add frame number
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            out.write(frame)
    finally:
        out.release()

    return path


def create_test_files(tmp_path: Path) -> Dict[str, str]:
    """Create test files for import testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Dictionary mapping file types to their paths
    """
    files = {
        "valid_video": create_test_video(str(tmp_path / "valid.mp4")),
        "invalid_video": str(tmp_path / "invalid.mp4"),
        "non_video": str(tmp_path / "test.txt"),
        "missing": str(tmp_path / "missing.mp4"),
    }

    # Create invalid video file
    with open(files["invalid_video"], "wb") as f:
        f.write(b"not a video file")

    # Create non-video file
    with open(files["non_video"], "w") as f:
        f.write("text file content")

    return files


def create_drag_event(
    widget: QWidget, urls: List[str], event_type: str = "enter"
) -> QDragEnterEvent:
    """Create a drag event for testing.

    Args:
        widget: Widget to create event for
        urls: List of file URLs
        event_type: Type of event ('enter' or 'drop')

    Returns:
        Created event
    """
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(url) for url in urls])

    pos = widget.rect().center()
    if event_type == "enter":
        return QDragEnterEvent(
            pos,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    else:
        return QDropEvent(
            pos,
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            mime_data,
        )


def create_test_metadata(
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    duration: float = 60.0,
    codec: str = "h264",
) -> VideoMetadata:
    """Create test video metadata.

    Args:
        width: Video width
        height: Video height
        fps: Frames per second
        duration: Video duration in seconds
        codec: Video codec

    Returns:
        VideoMetadata instance
    """
    return VideoMetadata(
        width=width, height=height, fps=fps, duration=duration, codec=codec
    )


def cleanup_test_files(files: Dict[str, str]):
    """Clean up test files.

    Args:
        files: Dictionary of file paths to clean up
    """
    for path in files.values():
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

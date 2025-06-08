"""
Tests for the VideoTimeline component.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest

from spygate.gui.components.video_timeline import Annotation, AnnotationType, VideoTimeline


@pytest.fixture
def video_timeline(qtbot):
    """Create a VideoTimeline instance for testing."""
    timeline = VideoTimeline()
    timeline.resize(800, 200)  # Set a reasonable size for testing
    qtbot.addWidget(timeline)

    # Mock video info
    timeline.setVideoInfo(total_frames=300, fps=30)

    return timeline


@pytest.fixture
def sample_video_file():
    """Create a temporary video file for testing."""
    import cv2

    # Create a temporary video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_path = temp_file.name

    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))

    # Write 90 frames (3 seconds at 30fps)
    for _ in range(90):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)

    out.release()

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_timeline_initialization(video_timeline):
    """Test that the timeline is properly initialized."""
    assert video_timeline.total_frames == 300
    assert video_timeline.fps == 30
    assert video_timeline.current_frame == 0
    assert not video_timeline.is_playing
    assert len(video_timeline.annotations) == 0


def test_frame_navigation(video_timeline, qtbot):
    """Test frame navigation controls."""
    # Test next frame
    qtbot.keyClick(video_timeline, Qt.Key.Key_Right)
    assert video_timeline.current_frame == 1

    # Test previous frame
    qtbot.keyClick(video_timeline, Qt.Key.Key_Left)
    assert video_timeline.current_frame == 0

    # Test jump forward
    qtbot.keyClick(video_timeline, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert video_timeline.current_frame == 10

    # Test jump backward
    qtbot.keyClick(video_timeline, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert video_timeline.current_frame == 0


def test_playback_controls(video_timeline, qtbot):
    """Test playback controls."""
    # Test play/pause
    qtbot.mouseClick(video_timeline.play_button, Qt.MouseButton.LeftButton)
    assert video_timeline.is_playing

    qtbot.mouseClick(video_timeline.play_button, Qt.MouseButton.LeftButton)
    assert not video_timeline.is_playing

    # Test speed control
    video_timeline.speed_combo.setCurrentText("2x")
    assert video_timeline.playback_worker.playback_speed == 2.0

    # Test loop control
    qtbot.mouseClick(video_timeline.loop_button, Qt.MouseButton.LeftButton)
    assert video_timeline.loop_enabled


def test_marker_annotation(video_timeline, qtbot):
    """Test adding marker annotations."""
    # Mock color dialog
    with patch("PyQt6.QtWidgets.QColorDialog.getColor") as mock_color:
        mock_color.return_value = QColor(Qt.GlobalColor.red)

        # Add marker at frame 10
        video_timeline.setCurrentFrame(10)
        video_timeline.startAnnotation(AnnotationType.MARKER)

        # Verify marker was added
        assert len(video_timeline.annotations) == 1
        marker = video_timeline.annotations[0]
        assert marker.type == AnnotationType.MARKER
        assert marker.start_frame == 10
        assert marker.color == QColor(Qt.GlobalColor.red)


def test_region_annotation(video_timeline, qtbot):
    """Test adding region annotations."""
    # Start region at frame 10
    video_timeline.setCurrentFrame(10)
    video_timeline.startAnnotation(AnnotationType.REGION)

    # Drag to frame 20
    video_timeline.setCurrentFrame(20)
    video_timeline.finishAnnotation()

    # Verify region was added
    assert len(video_timeline.annotations) == 1
    region = video_timeline.annotations[0]
    assert region.type == AnnotationType.REGION
    assert region.start_frame == 10
    assert region.end_frame == 20


def test_text_annotation(video_timeline, qtbot):
    """Test adding text annotations."""
    # Mock input dialog
    with patch("PyQt6.QtWidgets.QInputDialog.getText") as mock_input:
        mock_input.return_value = ("Test annotation", True)

        # Add text annotation at frame 15
        video_timeline.setCurrentFrame(15)
        video_timeline.startAnnotation(AnnotationType.TEXT)

        # Verify text annotation was added
        assert len(video_timeline.annotations) == 1
        text_ann = video_timeline.annotations[0]
        assert text_ann.type == AnnotationType.TEXT
        assert text_ann.start_frame == 15
        assert text_ann.text == "Test annotation"


def test_annotation_selection(video_timeline, qtbot):
    """Test selecting and deleting annotations."""
    # Add a marker annotation
    with patch("PyQt6.QtWidgets.QColorDialog.getColor") as mock_color:
        mock_color.return_value = QColor(Qt.GlobalColor.red)
        video_timeline.setCurrentFrame(10)
        video_timeline.startAnnotation(AnnotationType.MARKER)

    # Mock annotation selection signal
    selection_signal = MagicMock()
    video_timeline.annotationSelected.connect(selection_signal)

    # Click on the annotation
    x = video_timeline._frame_to_x(10)
    y = (
        video_timeline.height()
        - video_timeline.TIMELINE_MARGIN
        - video_timeline.ANNOTATION_TRACK_HEIGHT // 2
    )
    qtbot.mouseClick(video_timeline, Qt.MouseButton.LeftButton, pos=QPoint(x, y))

    # Verify selection
    assert video_timeline.selected_annotation == video_timeline.annotations[0]
    selection_signal.assert_called_once()

    # Test deletion
    qtbot.keyClick(video_timeline, Qt.Key.Key_Delete)
    assert len(video_timeline.annotations) == 0


def test_thumbnail_generation(video_timeline, sample_video_file, qtbot):
    """Test thumbnail generation."""
    # Set video file and wait for thumbnails
    video_timeline.setVideoInfo(90, 30, sample_video_file)
    video_timeline.thumbnail_worker.thumbnailReady.emit(0, QSize(80, 45))

    # Verify thumbnail was added
    assert 0 in video_timeline.thumbnails


def test_performance_optimization(video_timeline, qtbot):
    """Test performance optimization features."""
    # Test hardware detection
    assert video_timeline.hardware is not None
    assert video_timeline.optimizer is not None

    # Test thumbnail optimization
    assert video_timeline.thumbnail_size == QSize(80, 45)

    # Add many annotations to test rendering performance
    for i in range(50):
        video_timeline.annotations.append(
            Annotation(
                type=AnnotationType.MARKER,
                start_frame=i * 5,
                color=QColor(Qt.GlobalColor.red),
            )
        )

    # Measure paint time
    with patch("time.perf_counter") as mock_time:
        mock_time.side_effect = [0, 0.016]  # Simulate 16ms paint time
        video_timeline.update()
        video_timeline.paintEvent(None)

    # Paint time should be under 16ms (60fps)
    assert len(video_timeline.annotations) == 50


def test_error_handling(video_timeline, qtbot):
    """Test error handling in the timeline component."""
    # Test invalid frame navigation
    video_timeline.setCurrentFrame(-1)
    assert video_timeline.current_frame == 0

    video_timeline.setCurrentFrame(1000)
    assert video_timeline.current_frame == 300

    # Test invalid video file
    with pytest.raises(Exception):
        video_timeline.setVideoInfo(100, 30, "nonexistent.mp4")

    # Test worker thread errors
    video_timeline.thumbnail_worker.error.emit("Test error")
    video_timeline.playback_worker.error.emit("Test error")


def test_memory_management(video_timeline, qtbot):
    """Test memory management and cleanup."""
    # Add some thumbnails
    for i in range(100):
        video_timeline.thumbnails[i] = None

    # Verify old thumbnails are cleaned up when limit is reached
    assert len(video_timeline.thumbnails) <= 100

    # Test cleanup on widget destruction
    video_timeline.cleanup()
    assert not video_timeline.thumbnail_worker.isRunning()
    assert not video_timeline.playback_worker.isRunning()

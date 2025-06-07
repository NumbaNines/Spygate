"""Tests for the VideoTimeline component."""

import os

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.ui.components.video_timeline import VideoTimeline

# Sample video file for testing
SAMPLE_VIDEO = os.path.join(os.path.dirname(__file__), "resources", "sample.mp4")


@pytest.fixture
def app():
    """Create a QApplication instance."""
    return QApplication([])


@pytest.fixture
def video_timeline(app):
    """Create a VideoTimeline instance."""
    timeline = VideoTimeline(SAMPLE_VIDEO, "Test Player")
    return timeline


def test_video_timeline_initialization(video_timeline):
    """Test VideoTimeline initialization."""
    assert video_timeline.player_name == "Test Player"
    assert video_timeline.video_path == SAMPLE_VIDEO
    assert video_timeline.current_frame == 0
    assert not video_timeline.is_playing


def test_playback_controls(video_timeline):
    """Test playback control buttons."""
    # Test play/pause
    QTest.mouseClick(video_timeline.play_button, Qt.MouseButton.LeftButton)
    assert video_timeline.is_playing
    QTest.mouseClick(video_timeline.play_button, Qt.MouseButton.LeftButton)
    assert not video_timeline.is_playing

    # Test frame navigation
    initial_frame = video_timeline.current_frame
    QTest.mouseClick(video_timeline.next_frame_button, Qt.MouseButton.LeftButton)
    assert video_timeline.current_frame == initial_frame + 1
    QTest.mouseClick(video_timeline.prev_frame_button, Qt.MouseButton.LeftButton)
    assert video_timeline.current_frame == initial_frame


def test_keyboard_shortcuts(video_timeline):
    """Test keyboard shortcuts."""
    # Test spacebar for play/pause
    QTest.keyClick(video_timeline, Qt.Key.Key_Space)
    assert video_timeline.is_playing
    QTest.keyClick(video_timeline, Qt.Key.Key_Space)
    assert not video_timeline.is_playing

    # Test arrow keys for frame navigation
    initial_frame = video_timeline.current_frame
    QTest.keyClick(video_timeline, Qt.Key.Key_Right)
    assert video_timeline.current_frame == initial_frame + 1
    QTest.keyClick(video_timeline, Qt.Key.Key_Left)
    assert video_timeline.current_frame == initial_frame


def test_timeline_slider(video_timeline):
    """Test timeline slider interaction."""
    # Test slider movement
    video_timeline.timeline_slider.setValue(50)
    assert video_timeline.current_frame == 50

    # Test time display format
    video_timeline.video_player.media_player.setPosition(65000)  # 1:05
    video_timeline.update_time_display()
    assert video_timeline.time_label.text().startswith("01:05")

from unittest.mock import Mock, patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from spygate.gui.video.video_timeline import VideoTimeline


# Required for Qt widgets
@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()


@pytest.fixture
def media_player():
    """Create a mock QMediaPlayer."""
    player = Mock(spec=QMediaPlayer)
    player.duration.return_value = 10000  # 10 seconds
    player.position.return_value = 0
    player.playbackState.return_value = QMediaPlayer.PlaybackState.StoppedState
    return player


@pytest.fixture
def video_timeline(qapp, media_player):
    """Create a VideoTimeline instance."""
    return VideoTimeline(media_player)


def test_initialization(video_timeline):
    """Test VideoTimeline initialization and UI setup."""
    # Check UI components exist
    assert video_timeline.current_time_label is not None
    assert video_timeline.duration_label is not None
    assert video_timeline.position_slider is not None
    assert video_timeline.prev_frame_btn is not None
    assert video_timeline.next_frame_btn is not None


def test_time_formatting(video_timeline):
    """Test time formatting."""
    assert video_timeline._format_time(0) == "00:00:00"
    assert video_timeline._format_time(1000) == "00:00:01"
    assert video_timeline._format_time(61000) == "00:01:01"
    assert video_timeline._format_time(3661000) == "01:01:01"


def test_position_update(video_timeline):
    """Test position updates."""
    video_timeline.timeline_manager._frame_rate = 30.0
    video_timeline._update_duration(10000)  # Set max range first
    video_timeline._update_position(5000)  # 5 seconds
    assert video_timeline.position_slider.value() == 5000


def test_duration_update(video_timeline):
    """Test duration updates."""
    video_timeline._update_duration(10000)  # 10 seconds
    assert video_timeline.position_slider.maximum() == 10000
    assert video_timeline.duration_label.text() == "00:00:10"


def test_frame_display_update(video_timeline):
    """Test frame counter updates."""
    video_timeline._update_frame_display(150)
    assert video_timeline.frame_label.text() == "Frame: 150 / 0"


def test_play_state_update(video_timeline):
    """Test play/pause button state updates."""
    video_timeline._update_play_state(QMediaPlayer.PlaybackState.PlayingState)
    assert video_timeline.play_pause_btn.icon() is not None

    video_timeline._update_play_state(QMediaPlayer.PlaybackState.PausedState)
    assert video_timeline.play_pause_btn.icon() is not None


def test_keyboard_shortcuts(video_timeline):
    """Test keyboard shortcuts."""
    video_timeline.timeline_manager.previous_frame = Mock()
    video_timeline.timeline_manager.next_frame = Mock()
    video_timeline.timeline_manager.is_playing = False

    # Test left arrow key
    QTest.keyClick(video_timeline, Qt.Key.Key_Left)
    video_timeline.timeline_manager.previous_frame.assert_called_once()

    # Test right arrow key
    QTest.keyClick(video_timeline, Qt.Key.Key_Right)
    video_timeline.timeline_manager.next_frame.assert_called_once()


def test_button_clicks(video_timeline):
    """Test button click handlers."""
    # Mock timeline manager methods
    video_timeline.timeline_manager.previous_frame = Mock()
    video_timeline.timeline_manager.next_frame = Mock()
    video_timeline.timeline_manager.is_playing = False

    # Test previous frame button
    video_timeline.prev_frame_btn.clicked.emit()
    video_timeline.timeline_manager.previous_frame.assert_called_once()

    # Test next frame button
    video_timeline.next_frame_btn.clicked.emit()
    video_timeline.timeline_manager.next_frame.assert_called_once()


def test_slider_interaction(video_timeline):
    """Test position slider interaction."""
    video_timeline.timeline_manager.seek_to_position = Mock()

    # Set up slider range and value
    video_timeline._update_duration(10000)  # Set max range first
    video_timeline.position_slider.setValue(5000)
    video_timeline._on_slider_released()  # Call the handler directly

    video_timeline.timeline_manager.seek_to_position.assert_called_once_with(5000)

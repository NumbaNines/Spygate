"""Tests for video transcode widget."""

import os
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from spygate.gui.video.video_transcode_widget import VideoTranscodeWidget
from spygate.video.transcoder import TranscodeError, TranscodeOptions


@pytest.fixture
def widget(qtbot):
    """Create a video transcode widget."""
    widget = VideoTranscodeWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def mock_video_service():
    """Create a mock video service."""
    with patch("spygate.gui.video.video_transcode_widget.VideoService") as mock:
        yield mock


def test_initial_state(widget):
    """Test initial widget state."""
    # Check default codec
    assert widget.codec_combo.currentText() == "h264"

    # Check default resolution
    assert widget.resolution_combo.currentText() == "Original"

    # Check default FPS
    assert widget.fps_spin.value() == 30

    # Check default quality
    assert widget.quality_spin.value() == 23

    # Check default audio state
    assert widget.audio_check.isChecked()

    # Check initial visibility
    assert not widget.progress_bar.isVisible()
    assert not widget.status_label.isVisible()
    assert not widget.cancel_button.isVisible()
    assert widget.transcode_button.isEnabled()


def test_transcode_video(widget, qtbot, tmp_path):
    """Test video transcoding process."""
    # Create a test video file
    input_path = str(tmp_path / "test.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Mock the worker's transcode method
    widget.worker.transcode = MagicMock()

    # Start transcoding
    widget.transcode_video(input_path)

    # Check UI state during transcoding
    assert widget.progress_bar.isVisible()
    assert widget.status_label.isVisible()
    assert widget.cancel_button.isVisible()
    assert not widget.transcode_button.isEnabled()

    # Verify worker was called with correct options
    widget.worker.transcode.assert_called_once()
    args = widget.worker.transcode.call_args[0]
    assert args[0] == input_path  # Input path
    assert args[1].endswith("_transcoded.mp4")  # Output path

    options = args[2]  # TranscodeOptions
    assert isinstance(options, TranscodeOptions)
    assert options.codec == "h264"
    assert options.fps == 30
    assert options.quality == 23
    assert options.include_audio is True


def test_transcode_custom_options(widget, qtbot, tmp_path):
    """Test transcoding with custom options."""
    input_path = str(tmp_path / "test.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Set custom options
    widget.codec_combo.setCurrentText("h265")
    widget.resolution_combo.setCurrentText("1080p (1920x1080)")
    widget.fps_spin.setValue(60)
    widget.quality_spin.setValue(18)
    widget.audio_check.setChecked(False)

    # Mock worker
    widget.worker.transcode = MagicMock()

    # Start transcoding
    widget.transcode_video(input_path)

    # Verify options
    options = widget.worker.transcode.call_args[0][2]
    assert options.codec == "h265"
    assert options.fps == 60
    assert options.quality == 18
    assert options.width == 1920
    assert options.height == 1080
    assert not options.include_audio


def test_progress_updates(widget, qtbot):
    """Test progress updates during transcoding."""
    # Connect to signals
    started_signal = MagicMock()
    progress_signal = MagicMock()
    finished_signal = MagicMock()
    widget.transcode_started.connect(started_signal)
    widget.transcode_finished.connect(finished_signal)

    # Simulate transcoding start
    widget._on_transcode_started("test.mp4")
    assert started_signal.called
    assert widget.status_label.text().startswith("Transcoding")

    # Simulate progress updates
    widget._on_transcode_progress("test.mp4", 50.0)
    assert widget.progress_bar.value() == 50

    # Simulate completion
    widget._on_transcode_finished("test.mp4", "test_transcoded.mp4")
    assert finished_signal.called
    assert widget.progress_bar.value() == 100
    assert not widget.progress_bar.isVisible()
    assert widget.transcode_button.isEnabled()


def test_error_handling(widget, qtbot):
    """Test error handling during transcoding."""
    # Connect to error signal
    error_signal = MagicMock()
    widget.transcode_error.connect(error_signal)

    # Simulate error
    with patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_critical:
        widget._on_transcode_error("test.mp4", "Test error")

    # Verify error handling
    assert error_signal.called
    assert mock_critical.called
    assert widget.status_label.text().startswith("Error")
    assert not widget.progress_bar.isVisible()
    assert widget.transcode_button.isEnabled()


def test_cancel_transcoding(widget, qtbot):
    """Test cancelling transcoding operation."""
    # Start transcoding
    widget.transcode_video("test.mp4")

    # Mock worker's stop method
    widget.worker.stop = MagicMock()

    # Click cancel button
    qtbot.mouseClick(widget.cancel_button, Qt.LeftButton)

    # Verify cancellation
    assert widget.worker.stop.called
    assert widget.status_label.text() == "Cancelling transcoding..."


def test_cleanup(widget):
    """Test cleanup of resources."""
    # Mock thread
    widget.worker_thread = MagicMock()

    # Call cleanup
    widget.cleanup()

    # Verify thread cleanup
    assert widget.worker_thread.quit.called
    assert widget.worker_thread.wait.called

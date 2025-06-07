"""Tests for video import widget."""

import os
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QMimeData, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QMessageBox

from spygate.gui.dialogs.player_name_dialog import PlayerNameDialog
from spygate.gui.video.video_import import VideoImportWidget
from spygate.video.codec_validator import VideoMetadata


@pytest.fixture
def video_widget(qtbot):
    """Create a video import widget."""
    widget = VideoImportWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def mock_video_service():
    """Create a mock video service."""
    with patch("spygate.gui.video.video_import_widget.VideoService") as mock:
        yield mock


@pytest.fixture
def mock_codec_validator():
    """Create a mock codec validator."""
    with patch("spygate.gui.video.video_import_widget.CodecValidator") as mock:
        yield mock


@pytest.fixture
def mock_player_dialog():
    """Create a mock player name dialog."""
    with patch("spygate.gui.video.video_import_widget.PlayerNameDialog") as mock:
        dialog_instance = MagicMock()
        dialog_instance.exec.return_value = True
        dialog_instance.get_player_name.return_value = "Self"
        mock.return_value = dialog_instance
        yield mock


def test_drag_enter_valid(video_widget, qtbot):
    """Test drag enter with valid file."""
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile("test.mp4")])
    event = QDragEnterEvent(
        video_widget.pos(), Qt.CopyAction, mime_data, Qt.LeftButton, Qt.NoModifier
    )

    video_widget.dragEnterEvent(event)
    assert event.isAccepted()


def test_drag_enter_invalid(video_widget, qtbot):
    """Test drag enter with invalid data."""
    mime_data = QMimeData()
    event = QDragEnterEvent(
        video_widget.pos(), Qt.CopyAction, mime_data, Qt.LeftButton, Qt.NoModifier
    )

    video_widget.dragEnterEvent(event)
    assert not event.isAccepted()


def test_drop_valid_file(
    video_widget,
    mock_video_service,
    mock_codec_validator,
    mock_player_dialog,
    qtbot,
    tmp_path,
):
    """Test dropping a valid video file."""
    # Create a temporary video file
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy video content")

    # Mock codec validation
    metadata = VideoMetadata(width=1920, height=1080, fps=30, codec="h264", duration=60)
    mock_codec_validator.return_value.validate.return_value = metadata

    # Create drop event
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(video_file))])
    event = QDropEvent(
        video_widget.pos(),
        Qt.CopyAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
        mime_data,
    )

    # Handle drop
    with qtbot.waitSignal(video_widget.import_started):
        video_widget.dropEvent(event)

    # Verify player dialog was shown
    mock_player_dialog.assert_called_once()

    # Verify video service was called with correct arguments
    mock_video_service.return_value.upload_videos.assert_called_once()
    args = mock_video_service.return_value.upload_videos.call_args[0]
    assert len(args[0]) == 1  # One file
    assert args[0][0][0] == str(video_file)  # File path
    assert args[0][0][1] == metadata  # Metadata
    assert args[0][0][2] == "Self"  # Player name


def test_drop_invalid_file(
    video_widget,
    mock_video_service,
    mock_codec_validator,
    mock_player_dialog,
    qtbot,
    tmp_path,
):
    """Test dropping an invalid video file."""
    # Create a temporary file
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("not a video")

    # Mock codec validation to fail
    mock_codec_validator.return_value.validate.side_effect = Exception("Invalid codec")

    # Create drop event
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(invalid_file))])
    event = QDropEvent(
        video_widget.pos(),
        Qt.CopyAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
        mime_data,
    )

    # Handle drop
    with qtbot.waitSignal(video_widget.import_error):
        video_widget.dropEvent(event)

    # Verify player dialog was not shown
    mock_player_dialog.assert_not_called()

    # Verify video service was not called
    mock_video_service.return_value.upload_videos.assert_not_called()


def test_select_files_valid(
    video_widget,
    mock_video_service,
    mock_codec_validator,
    mock_player_dialog,
    qtbot,
    monkeypatch,
):
    """Test selecting valid files via dialog."""

    # Mock file dialog
    def mock_get_open_file_names(*args, **kwargs):
        return ["test1.mp4", "test2.mp4"], None

    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames", mock_get_open_file_names
    )

    # Mock codec validation
    metadata = VideoMetadata(width=1920, height=1080, fps=30, codec="h264", duration=60)
    mock_codec_validator.return_value.validate.return_value = metadata

    # Select files
    with qtbot.waitSignal(video_widget.import_started):
        video_widget.select_files()

    # Verify player dialog was shown
    mock_player_dialog.assert_called_once()

    # Verify video service was called with correct arguments
    mock_video_service.return_value.upload_videos.assert_called_once()
    args = mock_video_service.return_value.upload_videos.call_args[0]
    assert len(args[0]) == 2  # Two files
    assert args[0][0][2] == "Self"  # Player name for first file
    assert args[0][1][2] == "Self"  # Player name for second file


def test_select_files_cancel_player_dialog(
    video_widget,
    mock_video_service,
    mock_codec_validator,
    mock_player_dialog,
    qtbot,
    monkeypatch,
):
    """Test canceling player name dialog."""

    # Mock file dialog
    def mock_get_open_file_names(*args, **kwargs):
        return ["test.mp4"], None

    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames", mock_get_open_file_names
    )

    # Mock codec validation
    metadata = VideoMetadata(width=1920, height=1080, fps=30, codec="h264", duration=60)
    mock_codec_validator.return_value.validate.return_value = metadata

    # Mock player dialog to cancel
    mock_player_dialog.return_value.exec.return_value = False

    # Select files
    video_widget.select_files()

    # Verify player dialog was shown but video service was not called
    mock_player_dialog.assert_called_once()
    mock_video_service.return_value.upload_videos.assert_not_called()


def test_import_progress(video_widget, mock_video_service, qtbot):
    """Test import progress updates."""
    # Start import
    video_widget.start_import([("test.mp4", MagicMock(), "Self")])

    # Get progress callback
    progress_callback = mock_video_service.return_value.upload_videos.call_args[0][1]

    # Simulate progress updates
    with qtbot.waitSignal(video_widget.import_progress, timeout=1000):
        progress_callback(50)

    assert video_widget.progress_bar.value() == 50


def test_import_error(video_widget, mock_video_service, qtbot, monkeypatch):
    """Test import error handling."""
    # Mock QMessageBox
    mock_critical = MagicMock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    # Mock video service to raise error
    mock_video_service.return_value.upload_videos.side_effect = Exception(
        "Import failed"
    )

    # Start import
    with qtbot.waitSignal(video_widget.import_error):
        video_widget.start_import([("test.mp4", MagicMock(), "Self")])

    # Verify error dialog was shown
    mock_critical.assert_called_once()
    assert "Import failed" in mock_critical.call_args[0]


def test_import_complete(video_widget, mock_video_service, qtbot, monkeypatch):
    """Test import completion."""
    # Mock QMessageBox
    mock_information = MagicMock()
    monkeypatch.setattr(QMessageBox, "information", mock_information)

    # Start import
    with qtbot.waitSignal(video_widget.import_finished):
        video_widget.start_import([("test.mp4", MagicMock(), "Self")])
        # Simulate worker completion
        video_widget.import_worker.finished.emit()

    # Verify success dialog was shown
    mock_information.assert_called_once()
    assert "completed successfully" in mock_information.call_args[0]

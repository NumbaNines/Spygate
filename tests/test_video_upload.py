"""Tests for video upload functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QFileDialog, QLineEdit, QPushButton, QRadioButton

from src.database.models import Video, VideoMetadata
from src.database.session import Session
from src.ui.video import VideoImportWidget


@pytest.fixture
def sample_video(tmp_path):
    """Create a sample video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"dummy video content")
    return str(video_path)


def test_video_import_widget_creation(test_app, qtbot):
    """Test that the VideoImportWidget can be created."""
    widget = VideoImportWidget(skip_validation=True)
    qtbot.addWidget(widget)
    assert widget is not None
    assert widget.import_btn is not None


def test_video_import_self_gameplay(test_app, qtbot, test_session, sample_video):
    """Test uploading a video as self gameplay."""
    # Create widget with validation disabled
    widget = VideoImportWidget(skip_validation=True)
    qtbot.addWidget(widget)

    # Mock file dialog
    def mock_get_open_file_names(*args, **kwargs):
        return [sample_video], ""

    original_dialog = QFileDialog.getOpenFileNames
    QFileDialog.getOpenFileNames = MagicMock(side_effect=mock_get_open_file_names)

    try:
        # Click import button to trigger file dialog
        with qtbot.waitSignal(widget.dialog_about_to_show, timeout=5000):
            qtbot.mouseClick(widget.import_btn, Qt.MouseButton.LeftButton)

        # Wait for dialog to be created and shown
        qtbot.wait(500)  # Increased delay to ensure dialog is ready
        dialog = widget.get_current_dialog()
        assert dialog is not None

        # Select "Self" gameplay
        self_radio = dialog.findChild(QRadioButton, "self_radio")
        assert self_radio is not None
        qtbot.mouseClick(self_radio, Qt.MouseButton.LeftButton)

        # Accept dialog
        ok_button = dialog.findChild(QPushButton, "ok_button")
        assert ok_button is not None
        qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        # Wait for database operation to complete
        qtbot.wait(100)

        # Verify database entry
        video = test_session.query(Video).first()
        assert video is not None
        assert video.player_name == "Self"

    finally:
        QFileDialog.getOpenFileNames = original_dialog


def test_video_import_opponent_gameplay(test_app, qtbot, test_session, sample_video):
    """Test uploading a video as opponent gameplay."""
    # Create widget with validation disabled
    widget = VideoImportWidget(skip_validation=True)
    qtbot.addWidget(widget)

    # Mock file dialog
    def mock_get_open_file_names(*args, **kwargs):
        return [sample_video], ""

    original_dialog = QFileDialog.getOpenFileNames
    QFileDialog.getOpenFileNames = MagicMock(side_effect=mock_get_open_file_names)

    try:
        # Click import button to trigger file dialog
        with qtbot.waitSignal(widget.dialog_about_to_show, timeout=5000):
            qtbot.mouseClick(widget.import_btn, Qt.MouseButton.LeftButton)

        # Wait for dialog to be created and shown
        qtbot.wait(500)  # Increased delay to ensure dialog is ready
        dialog = widget.get_current_dialog()
        assert dialog is not None

        # Select opponent gameplay and enter name
        opponent_radio = dialog.findChild(QRadioButton, "opponent_radio")
        assert opponent_radio is not None
        qtbot.mouseClick(opponent_radio, Qt.MouseButton.LeftButton)

        opponent_name = dialog.findChild(QLineEdit, "opponent_name")
        assert opponent_name is not None
        QTest.keyClicks(opponent_name, "TestPlayer")

        # Accept dialog
        ok_button = dialog.findChild(QPushButton, "ok_button")
        assert ok_button is not None
        qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        # Wait for database operation to complete
        qtbot.wait(100)

        # Verify database entry
        video = test_session.query(Video).first()
        assert video is not None
        assert video.player_name == "Opponent: TestPlayer"

    finally:
        QFileDialog.getOpenFileNames = original_dialog

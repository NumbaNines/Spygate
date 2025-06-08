"""Comprehensive test suite for video import functionality."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from spygate.database.video_manager import VideoManager
from spygate.gui.components.video_import import VideoImportWidget
from spygate.gui.workers.video_import_worker import VideoImportWorker
from spygate.services.video_service import VideoService
from spygate.video.metadata import VideoMetadata

from .utils.test_utils import (
    cleanup_test_files,
    create_drag_event,
    create_test_files,
    create_test_metadata,
)


# Fixtures
@pytest.fixture
def test_files(tmp_path) -> dict[str, str]:
    """Create test files for import testing."""
    files = create_test_files(tmp_path)
    yield files
    cleanup_test_files(files)


@pytest.fixture
def mock_video_service():
    """Mock video service."""
    with patch("spygate.services.video_service.VideoService") as mock:
        yield mock


@pytest.fixture
def mock_video_manager():
    """Mock video manager."""
    with patch("spygate.database.video_manager.VideoManager") as mock:
        yield mock


@pytest.fixture
def video_import_widget(qtbot, mock_video_service):
    """Create video import widget."""
    widget = VideoImportWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def video_import_worker(test_files):
    """Create video import worker."""
    return VideoImportWorker([test_files["valid_video"]])


# UI Tests
def test_widget_initialization(video_import_widget):
    """Test widget initialization."""
    assert video_import_widget.acceptDrops()
    assert video_import_widget.drop_frame is not None
    assert video_import_widget.label is not None
    assert video_import_widget.formats_label is not None
    assert "Drop video files here" in video_import_widget.label.text()
    assert "Supported formats" in video_import_widget.formats_label.text()


def test_drag_enter_valid_video(video_import_widget, test_files, qtbot):
    """Test drag enter with valid video file."""
    event = create_drag_event(video_import_widget, [test_files["valid_video"]])
    video_import_widget.dragEnterEvent(event)
    assert event.isAccepted()
    assert "highlight" in video_import_widget.drop_frame.styleSheet().lower()


def test_drag_enter_invalid_video(video_import_widget, test_files, qtbot):
    """Test drag enter with invalid video file."""
    event = create_drag_event(video_import_widget, [test_files["invalid_video"]])
    video_import_widget.dragEnterEvent(event)
    assert event.isAccepted()  # Accept initially, validation happens on drop


def test_drag_enter_non_video(video_import_widget, test_files, qtbot):
    """Test drag enter with non-video file."""
    event = create_drag_event(video_import_widget, [test_files["non_video"]])
    video_import_widget.dragEnterEvent(event)
    assert not event.isAccepted()


def test_drop_valid_video(video_import_widget, test_files, qtbot, monkeypatch):
    """Test dropping valid video file."""
    # Mock progress dialog
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # Track signals
    import_started = False
    video_import_widget.import_started.connect(lambda: setattr(locals(), "import_started", True))

    # Create and process drop event
    event = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event)

    # Wait for worker
    qtbot.wait(1000)
    assert import_started
    assert video_import_widget.worker is not None


def test_drop_invalid_video(video_import_widget, test_files, qtbot, monkeypatch):
    """Test dropping invalid video file."""
    # Mock message box
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    # Create and process drop event
    event = create_drag_event(video_import_widget, [test_files["invalid_video"]], "drop")
    video_import_widget.dropEvent(event)

    # Wait for worker
    qtbot.wait(1000)
    assert mock_warning.called
    assert "invalid" in mock_warning.call_args[0][1].lower()


def test_drop_multiple_files(video_import_widget, test_files, qtbot, monkeypatch):
    """Test dropping multiple files."""
    # Mock progress dialog
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # Track signals
    imported_files = []
    video_import_widget.import_finished.connect(lambda files: imported_files.extend(files))

    # Create and process drop event
    event = create_drag_event(
        video_import_widget,
        [test_files["valid_video"], test_files["invalid_video"]],
        "drop",
    )
    video_import_widget.dropEvent(event)

    # Wait for worker
    qtbot.wait(1000)
    assert len(imported_files) == 1  # Only valid file should be imported


# Worker Tests
def test_worker_initialization(video_import_worker):
    """Test worker initialization."""
    assert isinstance(video_import_worker, QThread)
    assert len(video_import_worker.file_paths) == 1


def test_worker_progress_signals(video_import_worker, qtbot):
    """Test worker progress signals."""
    progress_values = []
    video_import_worker.progress.connect(progress_values.append)

    with qtbot.waitSignal(video_import_worker.finished, timeout=5000):
        video_import_worker.start()

    assert len(progress_values) > 0
    assert progress_values[-1] == 100


def test_worker_error_handling(video_import_worker, qtbot, monkeypatch):
    """Test worker error handling."""

    # Mock video service to raise error
    def mock_import(*args, **kwargs):
        raise Exception("Import failed")

    monkeypatch.setattr(VideoService, "import_video", mock_import)

    errors = []
    video_import_worker.error.connect(errors.append)

    with qtbot.waitSignal(video_import_worker.finished, timeout=5000):
        video_import_worker.start()

    assert len(errors) == 1
    assert "Import failed" in errors[0]


# Integration Tests
def test_full_import_flow(video_import_widget, test_files, qtbot, monkeypatch):
    """Test full import flow from UI to database."""
    # Mock database operations
    mock_db = MagicMock()
    monkeypatch.setattr(VideoManager, "add_video", mock_db.add_video)

    # Mock progress dialog
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # Track signals
    imported_files = []
    video_import_widget.import_finished.connect(lambda files: imported_files.extend(files))

    # Start import
    event = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event)

    # Wait for completion
    qtbot.wait(2000)

    # Verify database call
    assert mock_db.add_video.called
    assert len(imported_files) == 1


def test_import_cancellation(video_import_widget, test_files, qtbot, monkeypatch):
    """Test import cancellation."""
    # Mock progress dialog
    mock_dialog = MagicMock()
    mock_dialog.wasCanceled.return_value = True
    monkeypatch.setattr(QProgressDialog, "__new__", lambda *args, **kwargs: mock_dialog)

    # Start import
    event = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event)

    # Wait for cancellation
    qtbot.wait(1000)
    assert not hasattr(video_import_widget, "worker")


def test_concurrent_imports(video_import_widget, test_files, qtbot, monkeypatch):
    """Test handling of concurrent import attempts."""
    # Mock progress dialog
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # Start first import
    event1 = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event1)

    # Try second import immediately
    event2 = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event2)

    # Verify only one worker exists
    assert hasattr(video_import_widget, "worker")
    worker_id = id(video_import_widget.worker)

    # Try third import
    event3 = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event3)

    # Verify still same worker
    assert id(video_import_widget.worker) == worker_id


# Error Recovery Tests
def test_import_retry_after_error(video_import_widget, test_files, qtbot, monkeypatch):
    """Test retrying import after error."""
    # Mock video service to fail first time
    fail_count = [0]

    def mock_import(*args, **kwargs):
        if fail_count[0] == 0:
            fail_count[0] += 1
            raise Exception("First attempt fails")
        return True

    monkeypatch.setattr(VideoService, "import_video", mock_import)
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # First attempt
    event1 = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event1)
    qtbot.wait(1000)

    # Second attempt
    event2 = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event2)
    qtbot.wait(1000)

    assert fail_count[0] == 1  # Verify first attempt failed
    assert not hasattr(video_import_widget, "worker")  # Worker should be done


def test_cleanup_after_error(video_import_widget, test_files, qtbot, monkeypatch):
    """Test cleanup after import error."""

    # Mock video service to fail
    def mock_import(*args, **kwargs):
        raise Exception("Import failed")

    monkeypatch.setattr(VideoService, "import_video", mock_import)
    monkeypatch.setattr(QProgressDialog, "show", lambda x: None)

    # Start import
    event = create_drag_event(video_import_widget, [test_files["valid_video"]], "drop")
    video_import_widget.dropEvent(event)

    # Wait for error handling
    qtbot.wait(1000)

    # Verify cleanup
    assert not hasattr(video_import_widget, "worker")
    assert not os.path.exists(test_files["valid_video"] + ".tmp")

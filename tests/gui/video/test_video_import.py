import pytest
import os
from PyQt6.QtCore import QMimeData, QUrl, Qt, QPoint, QPointF, QEvent, QThread
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from spygate.gui.video.video_import import VideoImportWidget, VideoImportWorker
from spygate.video.codec_validator import VideoMetadata
from tests.video.test_codec_validator import sample_video_path

@pytest.fixture
def video_import_widget(qtbot):
    widget = VideoImportWidget()
    qtbot.addWidget(widget)
    return widget

@pytest.fixture
def video_import_worker(sample_video_path):
    """Create a VideoImportWorker instance with a sample video path."""
    return VideoImportWorker([sample_video_path])

def test_widget_initialization(video_import_widget):
    """Test that the widget is properly initialized."""
    assert video_import_widget.acceptDrops()
    assert video_import_widget.drop_frame is not None
    assert video_import_widget.label is not None
    assert video_import_widget.formats_label is not None
    assert "Drop video files here" in video_import_widget.label.text()
    assert "Supported formats" in video_import_widget.formats_label.text()

def test_drag_enter_with_video_file(video_import_widget, qtbot):
    """Test drag enter event with valid video file."""
    # Create mime data with video file URL
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile("test.mp4")])
    
    # Create drag enter event
    pos = video_import_widget.rect().center()
    event = QDragEnterEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    # Process the event
    video_import_widget.dragEnterEvent(event)
    
    # Check that the event was accepted
    assert event.isAccepted()
    
    # Check that the frame style changed
    assert "0078d4" in video_import_widget.drop_frame.styleSheet()

def test_drag_enter_with_non_video_file(video_import_widget, qtbot):
    """Test drag enter event with non-video file."""
    # Create mime data with non-video file URL
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile("test.txt")])
    
    # Create drag enter event
    pos = video_import_widget.rect().center()
    event = QDragEnterEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    # Process the event
    video_import_widget.dragEnterEvent(event)
    
    # Check that the event was not accepted
    assert not event.isAccepted()
    
    # Check that the frame style didn't change
    assert "666" in video_import_widget.drop_frame.styleSheet()

def test_drag_enter_with_mixed_files(video_import_widget, qtbot):
    """Test drag enter event with both video and non-video files."""
    mime_data = QMimeData()
    mime_data.setUrls([
        QUrl.fromLocalFile("test.mp4"),
        QUrl.fromLocalFile("test.txt")
    ])
    
    pos = video_import_widget.rect().center()
    event = QDragEnterEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    video_import_widget.dragEnterEvent(event)
    assert event.isAccepted()
    assert "0078d4" in video_import_widget.drop_frame.styleSheet()

def test_drag_leave_event(video_import_widget, qtbot):
    """Test drag leave event resets the frame style."""
    # First trigger a drag enter
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile("test.mp4")])
    
    pos = video_import_widget.rect().center()
    enter_event = QDragEnterEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    video_import_widget.dragEnterEvent(enter_event)
    assert "0078d4" in video_import_widget.drop_frame.styleSheet()
    
    # Then trigger drag leave
    video_import_widget.dragLeaveEvent(None)
    assert "666" in video_import_widget.drop_frame.styleSheet()

def test_drop_event_with_video_files(video_import_widget, qtbot, monkeypatch, sample_video_path):
    """Test drop event with video files."""
    # Track emitted signals
    videos_imported = []
    video_import_widget.videosImported.connect(lambda videos: videos_imported.extend(videos))
    
    # Create mime data with video file URLs
    mime_data = QMimeData()
    urls = [
        QUrl.fromLocalFile(sample_video_path),
        QUrl.fromLocalFile("invalid.mp4")  # Invalid file
    ]
    mime_data.setUrls(urls)
    
    # Create drop event
    center = video_import_widget.rect().center()
    pos = QPointF(center.x(), center.y())
    event = QDropEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    # Mock QProgressDialog to prevent it from showing
    def mock_progress_dialog(*args, **kwargs):
        dialog = QProgressDialog(*args, **kwargs)
        dialog.show = lambda: None
        return dialog
    monkeypatch.setattr("PyQt6.QtWidgets.QProgressDialog", mock_progress_dialog)
    
    # Mock QMessageBox to prevent it from showing
    def mock_warning(*args, **kwargs):
        pass
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)
    
    # Process the event
    video_import_widget.dropEvent(event)
    
    # Wait for the worker thread to finish
    while video_import_widget.worker.isRunning():
        qtbot.wait(100)
    
    # Check that only the valid video was imported
    assert len(videos_imported) == 1
    path, metadata = videos_imported[0]
    assert os.path.normpath(path) == os.path.normpath(sample_video_path)
    assert isinstance(metadata, VideoMetadata)
    assert metadata.codec == "H.264"

def test_drop_event_with_no_video_files(video_import_widget, qtbot):
    """Test drop event with no video files."""
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile("test.txt")])
    
    center = video_import_widget.rect().center()
    pos = QPointF(center.x(), center.y())
    event = QDropEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier
    )
    
    # Process the event
    video_import_widget.dropEvent(event)
    
    # No worker should be created
    assert not hasattr(video_import_widget, 'worker')

def test_worker_initialization(video_import_worker):
    """Test VideoImportWorker initialization."""
    assert isinstance(video_import_worker, QThread)
    assert len(video_import_worker.file_paths) == 1

def test_worker_signals(video_import_worker, qtbot):
    """Test VideoImportWorker signals."""
    # Track progress updates
    progress_values = []
    video_import_worker.progress.connect(progress_values.append)
    
    # Track finished signal
    results = []
    video_import_worker.finished.connect(results.append)
    
    # Run the worker
    with qtbot.waitSignal(video_import_worker.finished, timeout=5000):
        video_import_worker.start()
    
    # Check progress values
    assert len(progress_values) > 0
    assert progress_values[-1] == 100  # Final progress should be 100%
    
    # Check results
    assert len(results) == 1
    result_list = results[0]
    assert len(result_list) == 1
    
    path, is_valid, error_msg, metadata = result_list[0]
    assert is_valid
    assert error_msg == ""
    assert isinstance(metadata, VideoMetadata)
    assert metadata.codec == "H.264"

def test_worker_with_invalid_file(qtbot, tmp_path):
    """Test VideoImportWorker with an invalid file."""
    # Create an invalid file
    invalid_file = tmp_path / "invalid.mp4"
    invalid_file.write_bytes(b"not a video file")
    
    # Create worker
    worker = VideoImportWorker([str(invalid_file)])
    
    # Track results
    results = []
    worker.finished.connect(results.append)
    
    # Run the worker
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()
    
    # Check results
    assert len(results) == 1
    result_list = results[0]
    assert len(result_list) == 1
    
    path, is_valid, error_msg, metadata = result_list[0]
    assert not is_valid
    assert "Failed to open video file" in error_msg
    assert metadata is None

def test_worker_with_multiple_files(video_import_worker, qtbot, tmp_path):
    """Test VideoImportWorker with multiple files."""
    # Create an invalid file
    invalid_file = tmp_path / "invalid.mp4"
    invalid_file.write_bytes(b"not a video file")
    
    # Create worker with both valid and invalid files
    worker = VideoImportWorker([
        video_import_worker.file_paths[0],  # Valid file
        str(invalid_file)  # Invalid file
    ])
    
    # Track results
    results = []
    worker.finished.connect(results.append)
    
    # Run the worker
    with qtbot.waitSignal(worker.finished, timeout=5000):
        worker.start()
    
    # Check results
    assert len(results) == 1
    result_list = results[0]
    assert len(result_list) == 2
    
    # Check valid file results
    path1, is_valid1, error_msg1, metadata1 = result_list[0]
    assert is_valid1
    assert error_msg1 == ""
    assert isinstance(metadata1, VideoMetadata)
    
    # Check invalid file results
    path2, is_valid2, error_msg2, metadata2 = result_list[1]
    assert not is_valid2
    assert "Failed to open video file" in error_msg2
    assert metadata2 is None 
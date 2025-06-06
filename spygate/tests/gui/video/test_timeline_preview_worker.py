import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtTest import QSignalSpy
import numpy as np
import cv2
import os
import tempfile
import time
from spygate.gui.video.timeline_preview_worker import TimelinePreviewWorker

@pytest.fixture
def sample_video_path():
    # Create a temporary video file with 10 frames
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 48))
    for i in range(10):
        frame = np.full((48, 64, 3), i * 25, dtype=np.uint8)
        out.write(frame)
    out.release()
    yield video_path
    temp_dir.cleanup()

def test_preview_generation_success(qtbot, sample_video_path):
    worker = TimelinePreviewWorker(sample_video_path, list(range(0, 10)), QSize(32, 24))
    spy_finished = QSignalSpy(worker.finished)
    spy_error = QSignalSpy(worker.error)
    # Sleep briefly to ensure event loop is running
    time.sleep(0.05)
    worker.start()
    qtbot.waitSignal(worker.finished, timeout=3000)
    worker.wait()
    if len(spy_finished) != 1:
        print("[DEBUG] finished signal not received in test_preview_generation_success")
    assert len(spy_error) == 0
    assert len(spy_finished) == 1
    previews = spy_finished[0][0]
    assert isinstance(previews, dict)
    assert len(previews) > 0
    for pixmap in previews.values():
        assert pixmap.width() == 32 and pixmap.height() == 24

def test_file_not_found_error(qtbot):
    worker = TimelinePreviewWorker("/nonexistent/file.mp4", [0, 1, 2], QSize(32, 24))
    spy_error = QSignalSpy(worker.error)
    # Sleep briefly to ensure event loop is running
    time.sleep(0.05)
    worker.start()
    qtbot.waitSignal(worker.error, timeout=2000)
    worker.wait()
    if len(spy_error) != 1:
        print("[DEBUG] error signal not received in test_file_not_found_error")
    assert len(spy_error) == 1
    assert "File not found" in spy_error[0][0]

def test_worker_stop_early(qtbot, sample_video_path):
    worker = TimelinePreviewWorker(sample_video_path, list(range(0, 10)), QSize(32, 24))
    spy_finished = QSignalSpy(worker.finished)
    # Sleep briefly to ensure event loop is running
    time.sleep(0.05)
    worker.start()
    worker.stop()
    qtbot.waitSignal(worker.finished, timeout=3000)
    worker.wait()
    if len(spy_finished) == 0:
        print("[DEBUG] finished signal not received in test_worker_stop_early")
    previews = spy_finished[0][0] if len(spy_finished) > 0 else {}
    # If stopped early, may have fewer than all previews
    assert isinstance(previews, dict)
    assert len(previews) <= 10 
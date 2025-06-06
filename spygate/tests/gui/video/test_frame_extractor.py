import pytest
import cv2
import numpy as np
import os
import tempfile
from spygate.video.frame_extractor import FrameExtractor
from spygate.video.frame_extraction_worker import FrameExtractionWorker

def create_temp_video(path, num_frames=10, width=64, height=48, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), i * 25, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_extract_frame_and_release():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        create_temp_video(video_path)
        extractor = FrameExtractor(video_path)
        frame = extractor.extract_frame(0.5)  # Should be frame 5
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        extractor.release()


def test_extract_frames_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        create_temp_video(video_path, num_frames=10, fps=2)
        extractor = FrameExtractor(video_path)
        frames = list(extractor.extract_frames(interval=0.5))
        assert len(frames) > 0
        for t, frame in frames:
            assert frame is not None
            assert frame.shape == (48, 64, 3)
        extractor.release()


def test_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        FrameExtractor("nonexistent.mp4")


def test_frame_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        create_temp_video(video_path)
        extractor = FrameExtractor(video_path)
        frame1 = extractor.extract_frame(0.5)
        frame2 = extractor.extract_frame(0.5)
        assert frame1 is not None and frame2 is not None
        # Should be the same object from cache
        assert frame1 is frame2
        extractor.clear_cache()
        frame3 = extractor.extract_frame(0.5)
        assert frame3 is not None
        # After clearing cache, should be a new object
        assert frame3 is not frame1
        extractor.release()


def test_frame_extraction_worker(qtbot):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        # Create a 2-second, 10 fps video (20 frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10, (32, 24))
        for i in range(20):
            frame = np.full((24, 32, 3), i * 10, dtype=np.uint8)
            out.write(frame)
        out.release()

        worker = FrameExtractionWorker(video_path, interval=0.5)
        progress = []
        errors = []
        results = []
        worker.progress.connect(lambda c, t: progress.append((c, t)))
        worker.error.connect(errors.append)
        worker.finished.connect(results.append)
        worker.start()
        result = qtbot.waitSignal(worker.finished, timeout=3000, raising=True)
        qtbot.wait(100)  # Allow event loop to process callback
        frames = result.args[0] if result.args is not None else results[0]
        assert len(frames) > 0
        for t, frame in frames:
            assert frame is not None
            assert frame.shape == (24, 32, 3)
        assert not errors
        # Robust: progress count matches actual frames extracted
        assert progress[-1][0] == len(frames)


def test_frame_extraction_worker_stop(qtbot):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10, (32, 24))
        for i in range(20):
            frame = np.full((24, 32, 3), i * 10, dtype=np.uint8)
            out.write(frame)
        out.release()

        worker = FrameExtractionWorker(video_path, interval=0.1)
        results = []
        worker.finished.connect(results.append)
        worker.start()
        worker.stop()
        result = qtbot.waitSignal(worker.finished, timeout=3000, raising=True)
        qtbot.wait(100)
        frames = result.args[0] if result.args is not None else results[0]
        # Should emit finished, but with fewer frames
        assert len(frames) < 20 
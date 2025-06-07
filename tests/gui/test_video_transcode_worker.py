"""Tests for video transcoding worker."""

import os
from unittest.mock import MagicMock, patch

import pytest

from spygate.gui.workers.video_transcode_worker import VideoTranscodeWorker
from spygate.video.transcoder import TranscodeError, TranscodeOptions, Transcoder


@pytest.fixture
def mock_transcoder():
    """Create a mock transcoder."""
    return MagicMock(spec=Transcoder)


@pytest.fixture
def worker(mock_transcoder):
    """Create a video transcode worker."""
    return VideoTranscodeWorker(transcoder=mock_transcoder)


def test_transcode_success(worker, mock_transcoder, tmp_path):
    """Test successful video transcoding."""
    # Set up test files
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Set up signal tracking
    started_signal = MagicMock()
    progress_signal = MagicMock()
    finished_signal = MagicMock()
    error_signal = MagicMock()

    worker.started.connect(started_signal)
    worker.progress.connect(progress_signal)
    worker.finished.connect(finished_signal)
    worker.error.connect(error_signal)

    # Mock transcoder progress callback
    def mock_transcode(input_path, output_path, options, progress_callback):
        progress_callback(50.0)
        return {"duration": 10.0, "frames": 300}

    mock_transcoder.transcode.side_effect = mock_transcode

    # Start transcoding
    options = TranscodeOptions(codec="h264")
    worker.transcode(input_path, output_path, options)

    # Verify signals
    started_signal.assert_called_once_with(input_path)
    progress_signal.assert_called_once_with(input_path, 50.0)
    finished_signal.assert_called_once_with(input_path, output_path)
    assert not error_signal.called

    # Verify transcoder was called correctly
    mock_transcoder.transcode.assert_called_once()
    args = mock_transcoder.transcode.call_args[0]
    assert args[0] == input_path
    assert args[1] == output_path
    assert args[2] == options


def test_transcode_error(worker, mock_transcoder, tmp_path):
    """Test error handling during transcoding."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Set up signal tracking
    error_signal = MagicMock()
    finished_signal = MagicMock()
    worker.error.connect(error_signal)
    worker.finished.connect(finished_signal)

    # Make transcoder raise an error
    mock_transcoder.transcode.side_effect = TranscodeError("Test error")

    # Start transcoding
    options = TranscodeOptions(codec="h264")
    worker.transcode(input_path, output_path, options)

    # Verify error handling
    error_signal.assert_called_once_with(input_path, "Test error")
    assert not finished_signal.called


def test_transcode_stop(worker, mock_transcoder, tmp_path):
    """Test stopping transcoding operation."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Mock transcoder with a function that checks stop flag
    def mock_transcode(input_path, output_path, options, progress_callback):
        if worker._stop:
            raise TranscodeError("Transcoding stopped")
        return {"duration": 10.0, "frames": 300}

    mock_transcoder.transcode.side_effect = mock_transcode

    # Set up signal tracking
    error_signal = MagicMock()
    worker.error.connect(error_signal)

    # Start transcoding and immediately stop
    options = TranscodeOptions(codec="h264")
    worker.transcode(input_path, output_path, options)
    worker.stop()

    # Verify error was emitted
    error_signal.assert_called_once()
    assert "stopped" in error_signal.call_args[0][1].lower()


def test_transcode_cleanup(worker, mock_transcoder, tmp_path):
    """Test cleanup after transcoding."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Mock transcoder to simulate long operation
    def mock_transcode(input_path, output_path, options, progress_callback):
        if worker._stop:
            # Create partial output file
            with open(output_path, "wb") as f:
                f.write(b"partial data")
            raise TranscodeError("Transcoding stopped")
        return {"duration": 10.0, "frames": 300}

    mock_transcoder.transcode.side_effect = mock_transcode

    # Start transcoding and stop
    options = TranscodeOptions(codec="h264")
    worker.transcode(input_path, output_path, options)
    worker.stop()

    # Verify partial output was cleaned up
    assert not os.path.exists(output_path)


def test_transcode_multiple(worker, mock_transcoder, tmp_path):
    """Test transcoding multiple files in sequence."""
    # Create test files
    files = []
    for i in range(3):
        input_path = str(tmp_path / f"input{i}.mp4")
        output_path = str(tmp_path / f"output{i}.mp4")
        with open(input_path, "wb") as f:
            f.write(b"dummy video data")
        files.append((input_path, output_path))

    # Set up signal tracking
    finished_signal = MagicMock()
    worker.finished.connect(finished_signal)

    # Mock successful transcoding
    mock_transcoder.transcode.return_value = {"duration": 10.0, "frames": 300}

    # Transcode all files
    options = TranscodeOptions(codec="h264")
    for input_path, output_path in files:
        worker.transcode(input_path, output_path, options)

    # Verify all files were transcoded
    assert mock_transcoder.transcode.call_count == len(files)
    assert finished_signal.call_count == len(files)

    # Verify correct order
    for i, (input_path, output_path) in enumerate(files):
        call_args = finished_signal.call_args_list[i][0]
        assert call_args[0] == input_path
        assert call_args[1] == output_path

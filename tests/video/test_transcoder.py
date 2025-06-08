"""Tests for video transcoding functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from spygate.video.transcoder import TranscodeError, TranscodeOptions, Transcoder, TranscodeStatus


@pytest.fixture
def transcoder():
    """Create a transcoder instance."""
    return Transcoder()


@pytest.fixture
def mock_ffmpeg():
    """Mock ffmpeg subprocess calls."""
    with patch("spygate.video.transcoder.subprocess.Popen") as mock:
        # Mock process with progress simulation
        process_mock = MagicMock()
        process_mock.stderr.readline.side_effect = [
            b"frame=   10 fps=0.0 q=0.0 size=     0kB time=00:00:00.40",
            b"frame=   20 fps=0.0 q=0.0 size=     0kB time=00:00:00.80",
            b"frame=   30 fps=0.0 q=0.0 size=     0kB time=00:00:01.20",
            b"",  # EOF
        ]
        process_mock.wait.return_value = 0
        mock.return_value = process_mock
        yield mock


def test_transcode_basic(transcoder, mock_ffmpeg, tmp_path):
    """Test basic video transcoding."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")

    # Create dummy input file
    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    options = TranscodeOptions(codec="h264", fps=30, quality=23, include_audio=True)

    # Mock progress callback
    progress_callback = MagicMock()

    # Perform transcoding
    metadata = transcoder.transcode(input_path, output_path, options, progress_callback)

    # Verify ffmpeg was called correctly
    mock_ffmpeg.assert_called_once()
    args = mock_ffmpeg.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert input_path in args
    assert "-c:v" in args
    assert "libx264" in args
    assert "-crf" in args
    assert "23" in args
    assert output_path in args

    # Verify progress callback was called
    assert progress_callback.call_count > 0
    for call in progress_callback.call_args_list:
        progress = call[0][0]
        assert 0 <= progress <= 100


def test_transcode_no_audio(transcoder, mock_ffmpeg, tmp_path):
    """Test transcoding without audio."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")

    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    options = TranscodeOptions(codec="h264", fps=30, quality=23, include_audio=False)

    transcoder.transcode(input_path, output_path, options)

    # Verify ffmpeg was called with -an flag
    args = mock_ffmpeg.call_args[0][0]
    assert "-an" in args


def test_transcode_custom_resolution(transcoder, mock_ffmpeg, tmp_path):
    """Test transcoding with custom resolution."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")

    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    options = TranscodeOptions(codec="h264", fps=30, quality=23, width=1920, height=1080)

    transcoder.transcode(input_path, output_path, options)

    # Verify ffmpeg was called with scale filter
    args = mock_ffmpeg.call_args[0][0]
    assert "-vf" in args
    scale_idx = args.index("-vf") + 1
    assert "scale=1920:1080" in args[scale_idx]


def test_transcode_error_handling(transcoder, mock_ffmpeg, tmp_path):
    """Test error handling during transcoding."""
    input_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")

    with open(input_path, "wb") as f:
        f.write(b"dummy video data")

    # Make ffmpeg fail
    mock_ffmpeg.return_value.wait.return_value = 1
    mock_ffmpeg.return_value.stderr.readline.side_effect = [
        b"Error: something went wrong",
        b"",
    ]

    options = TranscodeOptions(codec="h264")

    with pytest.raises(TranscodeError) as exc_info:
        transcoder.transcode(input_path, output_path, options)

    assert "FFmpeg error" in str(exc_info.value)


def test_transcode_invalid_input(transcoder):
    """Test transcoding with invalid input file."""
    with pytest.raises(TranscodeError) as exc_info:
        transcoder.transcode("nonexistent.mp4", "output.mp4", TranscodeOptions(codec="h264"))

    assert "Input file not found" in str(exc_info.value)


def test_transcode_options_validation():
    """Test TranscodeOptions validation."""
    # Test valid options
    options = TranscodeOptions(
        codec="h264", fps=30, quality=23, width=1920, height=1080, include_audio=True
    )
    assert options.codec == "h264"
    assert options.fps == 30

    # Test invalid codec
    with pytest.raises(ValueError):
        TranscodeOptions(codec="invalid")

    # Test invalid fps
    with pytest.raises(ValueError):
        TranscodeOptions(codec="h264", fps=0)

    # Test invalid quality
    with pytest.raises(ValueError):
        TranscodeOptions(codec="h264", quality=52)  # CRF > 51

    # Test invalid resolution
    with pytest.raises(ValueError):
        TranscodeOptions(codec="h264", width=-1)

"""Unit tests for the ClipExtractor class."""

import os
from unittest.mock import Mock, patch

import pytest

from ...video.clip_extractor import ClipExtractor


@pytest.fixture(autouse=True)
def setup_test_dirs():
    """Ensure test directories exist."""
    os.makedirs(os.path.join("uploads", "videos"), exist_ok=True)
    os.makedirs(os.path.join("uploads", "clips"), exist_ok=True)


@pytest.fixture
def sample_video_path():
    """Create a sample video path for testing."""
    return os.path.join("uploads", "videos", "test.mp4")


@pytest.fixture
def sample_output_path():
    """Create a sample output path for testing."""
    return os.path.join("uploads", "clips", "test_clip.mp4")


def test_extract_segment_success(sample_video_path, sample_output_path):
    """Test successful clip extraction."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=10.0,
            end_time=20.0,
        )

        assert success is True
        assert error is None

        # Verify ffmpeg was called with correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]

        assert args[0] == "ffmpeg"
        assert "-i" in args
        assert sample_video_path in args
        assert "-ss" in args
        assert "10.0" in args
        assert "-t" in args
        assert "10.0" in args  # Duration is end_time - start_time
        assert sample_output_path in args


def test_extract_segment_with_copy_codec(sample_video_path, sample_output_path):
    """Test clip extraction with codec copying enabled."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=10.0,
            end_time=20.0,
            copy_codec=True,
        )

        assert success is True
        assert error is None

        # Verify ffmpeg was called with -c copy
        args = mock_run.call_args[0][0]
        assert "-c" in args
        assert "copy" in args


def test_extract_segment_failure(sample_video_path, sample_output_path):
    """Test handling of ffmpeg failure."""
    with patch("subprocess.run") as mock_run:
        # Simulate ffmpeg error
        mock_run.return_value = Mock(
            returncode=1, stderr=b"Error: Invalid data found when processing input"
        )

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=10.0,
            end_time=20.0,
        )

        assert success is False
        assert error is not None
        assert "Invalid data" in error


def test_extract_segment_invalid_times(sample_video_path, sample_output_path):
    """Test handling of invalid time values."""
    # Test negative start time
    success, error = ClipExtractor.extract_segment(
        source_path=sample_video_path,
        output_path=sample_output_path,
        start_time=-1.0,
        end_time=10.0,
    )
    assert success is False
    assert error is not None
    assert "Invalid start time" in error

    # Test end time before start time
    success, error = ClipExtractor.extract_segment(
        source_path=sample_video_path,
        output_path=sample_output_path,
        start_time=10.0,
        end_time=5.0,
    )
    assert success is False
    assert error is not None
    assert "End time must be after start time" in error


def test_extract_segment_missing_file(sample_output_path):
    """Test handling of missing source file."""
    success, error = ClipExtractor.extract_segment(
        source_path="/nonexistent/path/video.mp4",
        output_path=sample_output_path,
        start_time=0.0,
        end_time=10.0,
    )
    assert success is False
    assert error is not None
    assert "Source file not found" in error


def test_extract_segment_invalid_output_dir(sample_video_path):
    """Test handling of invalid output directory."""
    success, error = ClipExtractor.extract_segment(
        source_path=sample_video_path,
        output_path="/invalid/directory/clip.mp4",
        start_time=0.0,
        end_time=10.0,
    )
    assert success is False
    assert error is not None
    assert "Output directory does not exist" in error


def test_extract_segment_zero_duration(sample_video_path, sample_output_path):
    """Test handling of zero duration clips."""
    success, error = ClipExtractor.extract_segment(
        source_path=sample_video_path,
        output_path=sample_output_path,
        start_time=10.0,
        end_time=10.0,
    )
    assert success is False
    assert error is not None
    assert "Duration must be greater than 0" in error


def test_extract_segment_very_short_duration(sample_video_path, sample_output_path):
    """Test handling of very short duration clips."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=10.0,
            end_time=10.1,
        )

        assert success is True
        assert error is None

        args = mock_run.call_args[0][0]
        assert "0.1" in args  # Duration should be preserved


def test_extract_segment_with_custom_options(sample_video_path, sample_output_path):
    """Test clip extraction with custom ffmpeg options."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=10.0,
            end_time=20.0,
            extra_options=["-vf", "scale=1280:720"],
        )

        assert success is True
        assert error is None

        args = mock_run.call_args[0][0]
        assert "-vf" in args
        assert "scale=1280:720" in args


def test_extract_segment_with_output_exists(sample_video_path, sample_output_path):
    """Test handling when output file already exists."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=0.0,
            end_time=10.0,
            overwrite=False,
        )

        assert success is False
        assert error is not None
        assert "Output file already exists" in error


def test_extract_segment_with_overwrite(sample_video_path, sample_output_path):
    """Test overwriting existing output file."""
    with patch("subprocess.run") as mock_run, patch("os.path.exists") as mock_exists:
        mock_run.return_value = Mock(returncode=0)
        mock_exists.return_value = True

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=0.0,
            end_time=10.0,
            overwrite=True,
        )

        assert success is True
        assert error is None

        args = mock_run.call_args[0][0]
        assert "-y" in args  # ffmpeg overwrite flag


def test_extract_segment_subprocess_exception(sample_video_path, sample_output_path):
    """Test handling of subprocess exceptions."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Subprocess error")

        success, error = ClipExtractor.extract_segment(
            source_path=sample_video_path,
            output_path=sample_output_path,
            start_time=0.0,
            end_time=10.0,
        )

        assert success is False
        assert error is not None
        assert "Failed to execute ffmpeg" in error

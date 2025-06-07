import io
import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from spygate.video.codec_validator import (
    CodecValidator,
    VideoCorruptionError,
    VideoFormatError,
    VideoMetadata,
    VideoSizeError,
    VideoSpecificationError,
    VideoValidationError,
)


@pytest.fixture
def log_stream():
    """Create a stream to capture log output."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("spygate.video.codec_validator")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield stream
    logger.removeHandler(handler)
    stream.close()


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a temporary video file for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        str: Path to the temporary video file
    """
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"dummy video data")
    return str(video_path)


@pytest.fixture
def codec_validator():
    """Create a codec validator instance."""
    return CodecValidator()


def test_validate_valid_video(tmp_path, log_stream):
    """Test validation of a valid video file."""
    # Create a temporary test video
    video_path = tmp_path / "test.mp4"

    # Create video writer
    width, height = 1280, 720  # HD resolution
    fps = 30.0
    frame_count = 30  # 1 second of video
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        # Generate frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frame_count):
            out.write(frame)
    finally:
        out.release()

    # Ensure file is larger than minimum size
    file_size = os.path.getsize(video_path)
    if file_size < CodecValidator.MIN_FILE_SIZE:
        with open(video_path, "ab") as f:
            f.write(b"\0" * (CodecValidator.MIN_FILE_SIZE - file_size + 1024))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(video_path))

    assert is_valid
    assert error_msg == ""
    assert metadata is not None
    assert metadata.width == width
    assert metadata.height == height
    assert metadata.fps == fps
    assert metadata.frame_count == frame_count
    assert metadata.codec == "H.264"
    assert "Successfully validated video" in log_stream.getvalue()


def test_validate_nonexistent_file(tmp_path, log_stream):
    """Test validation of a nonexistent file."""
    video_path = tmp_path / "nonexistent.mp4"

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(video_path))

    assert not is_valid
    assert "File not found" in error_msg
    assert metadata is None
    assert "Error accessing video file" in log_stream.getvalue()


def test_validate_invalid_file(tmp_path, log_stream):
    """Test validation of an invalid file."""
    # Create an invalid video file (just some random bytes)
    invalid_file = tmp_path / "invalid.mp4"
    # Create a file larger than MIN_FILE_SIZE but with invalid content
    invalid_file.write_bytes(b"x" * (CodecValidator.MIN_FILE_SIZE + 1024))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(invalid_file))

    assert not is_valid
    assert (
        "corrupted" in error_msg.lower()
        or "failed to open video file" in error_msg.lower()
    )
    assert metadata is None
    assert "Video validation failed" in log_stream.getvalue()


def test_validate_too_small_file(tmp_path, log_stream):
    """Test validation of a file that's too small."""
    small_file = tmp_path / "small.mp4"
    small_file.write_bytes(b"x" * (CodecValidator.MIN_FILE_SIZE - 1))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(small_file))

    assert not is_valid
    assert "too small" in error_msg.lower()
    assert metadata is None
    assert "Video validation failed" in log_stream.getvalue()


def test_validate_too_large_file(tmp_path, log_stream):
    """Test validation of a file that's too large."""
    # Mock the file size check by temporarily changing MAX_FILE_SIZE
    original_max_size = CodecValidator.MAX_FILE_SIZE
    CodecValidator.MAX_FILE_SIZE = 1024  # Set to 1KB temporarily

    try:
        large_file = tmp_path / "large.mp4"
        large_file.write_bytes(b"x" * (CodecValidator.MAX_FILE_SIZE + 1))

        is_valid, error_msg, metadata = CodecValidator.validate_video(str(large_file))

        assert not is_valid
        assert "too large" in error_msg.lower()
        assert metadata is None
        assert "Video validation failed" in log_stream.getvalue()
    finally:
        CodecValidator.MAX_FILE_SIZE = original_max_size


def test_validate_corrupted_video(tmp_path, log_stream):
    """Test validation of a corrupted video file."""
    # Create a video file with invalid data
    video_path = tmp_path / "corrupted.mp4"

    # Create a file that looks like an MP4 but is actually corrupted
    with open(video_path, "wb") as f:
        # Write MP4 header signature
        f.write(b"\x00\x00\x00\x18ftypmp42")
        # Write some random data
        f.write(os.urandom(2048))
        # Write some more recognizable but invalid data
        f.write(b"mdat" + os.urandom(1024))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(video_path))

    assert not is_valid
    assert (
        "video file is corrupted" in error_msg.lower()
        or "failed to open video file" in error_msg.lower()
    )
    assert metadata is None
    assert "Video validation failed" in log_stream.getvalue()


def test_validate_low_resolution_video(tmp_path, log_stream):
    """Test validation of a video with resolution below minimum."""
    video_path = tmp_path / "low_res.mp4"

    # Create a video with resolution below minimum
    width, height = 100, 100  # Too small
    fps = 30.0
    frame_count = 30
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frame_count):
            out.write(frame)
    finally:
        out.release()

    # Ensure file is larger than minimum size
    file_size = os.path.getsize(video_path)
    if file_size < CodecValidator.MIN_FILE_SIZE:
        with open(video_path, "ab") as f:
            f.write(b"\0" * (CodecValidator.MIN_FILE_SIZE - file_size + 1024))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(video_path))

    assert not is_valid
    assert "resolution too low" in error_msg.lower()
    assert metadata is None
    assert "Video validation failed" in log_stream.getvalue()


def test_validate_high_fps_video(tmp_path, log_stream):
    """Test validation of a video with FPS above maximum."""
    video_path = tmp_path / "high_fps.mp4"

    # Create a video with FPS above maximum
    width, height = 1280, 720
    fps = CodecValidator.MAX_FPS + 1
    frame_count = 30
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frame_count):
            out.write(frame)
    finally:
        out.release()

    # Ensure file is larger than minimum size
    file_size = os.path.getsize(video_path)
    if file_size < CodecValidator.MIN_FILE_SIZE:
        with open(video_path, "ab") as f:
            f.write(b"\0" * (CodecValidator.MIN_FILE_SIZE - file_size + 1024))

    is_valid, error_msg, metadata = CodecValidator.validate_video(str(video_path))

    assert not is_valid
    assert "frame rate too high" in error_msg.lower()
    assert metadata is None
    assert "Video validation failed" in log_stream.getvalue()


def test_supported_codecs():
    """Test that supported codecs are properly defined."""
    assert len(CodecValidator.SUPPORTED_CODECS) > 0
    assert "H.264" in CodecValidator.SUPPORTED_CODECS.values()
    assert "H.265" in CodecValidator.SUPPORTED_CODECS.values()

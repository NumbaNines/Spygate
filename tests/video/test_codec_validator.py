import pytest
import cv2
import os
import tempfile
import numpy as np
from spygate.video.codec_validator import CodecValidator, VideoMetadata

@pytest.fixture
def sample_video_path():
    """Create a temporary test video file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
    
    # Create a sample video
    width, height = 640, 480
    fps = 30.0
    frame_count = 30  # 1 second of video
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    try:
        # Generate some frames (white noise)
        for _ in range(frame_count):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
    finally:
        out.release()
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

def test_validate_valid_video(sample_video_path):
    """Test validation of a valid video file."""
    is_valid, error_msg, metadata = CodecValidator.validate_video(sample_video_path)
    
    assert is_valid
    assert error_msg == ""
    assert isinstance(metadata, VideoMetadata)
    assert metadata.codec == "H.264"
    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.fps == pytest.approx(30.0)
    assert metadata.frame_count == 30
    assert metadata.duration == pytest.approx(1.0)

def test_validate_nonexistent_file():
    """Test validation of a non-existent file."""
    is_valid, error_msg, metadata = CodecValidator.validate_video("nonexistent.mp4")
    
    assert not is_valid
    assert "File not found." in error_msg
    assert metadata is None

def test_validate_invalid_file(tmp_path):
    """Test validation of an invalid file."""
    # Create an invalid video file (just some random bytes)
    invalid_file = tmp_path / "invalid.mp4"
    invalid_file.write_bytes(b"not a video file")
    
    is_valid, error_msg, metadata = CodecValidator.validate_video(str(invalid_file))
    
    assert not is_valid
    assert "Failed to open video file" in error_msg
    assert metadata is None

def test_supported_codecs():
    """Test that the supported codecs dictionary contains expected codecs."""
    assert "avc1" in CodecValidator.SUPPORTED_CODECS
    assert "h264" in CodecValidator.SUPPORTED_CODECS
    assert "hevc" in CodecValidator.SUPPORTED_CODECS
    assert "hev1" in CodecValidator.SUPPORTED_CODECS
    assert "mp4v" in CodecValidator.SUPPORTED_CODECS
    
    assert CodecValidator.SUPPORTED_CODECS["avc1"] == "H.264"
    assert CodecValidator.SUPPORTED_CODECS["hevc"] == "H.265"
    assert CodecValidator.SUPPORTED_CODECS["mp4v"] == "MPEG-4" 
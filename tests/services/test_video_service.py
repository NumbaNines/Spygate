"""Tests for video service."""

import os
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from spygate.database.models import Clip, TranscodedClip, TranscodeStatus
from spygate.services.video_service import VideoService
from spygate.video.codec_validator import VideoMetadata
from spygate.video.transcoder import TranscodeError, TranscodeOptions


@pytest.fixture
def video_service():
    """Create a video service instance."""
    return VideoService()


@pytest.fixture
def sample_metadata():
    """Create sample video metadata."""
    return VideoMetadata(
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        duration=10.0,
        has_audio=True,
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=44100,
        bit_rate=5000000,
        pixel_format="yuv420p",
        color_range="tv",
        color_space="bt709",
    )


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a temporary video file."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"dummy video data")
    return str(video_path)


def test_upload_video(video_service, sample_video_path, sample_metadata, db_session):
    """Test uploading a video file."""
    # Create test player
    player_id = uuid.uuid4()

    # Mock progress callback
    progress_callback = MagicMock()

    # Upload video
    clip_id, dest_path = video_service.upload_video(
        sample_video_path,
        sample_metadata,
        player_id,
        "Test Video",
        ["test", "demo"],
        progress_callback,
    )

    # Verify file was copied
    assert os.path.exists(dest_path)

    # Verify database entry
    clip = db_session.query(Clip).filter_by(id=clip_id).first()
    assert clip is not None
    assert clip.title == "Test Video"
    assert clip.player_id == player_id
    assert clip.duration == sample_metadata.duration
    assert clip.width == sample_metadata.width
    assert clip.height == sample_metadata.height
    assert clip.fps == sample_metadata.fps
    assert clip.codec == sample_metadata.codec
    assert clip.bitrate == sample_metadata.bit_rate

    # Verify tags were created
    assert len(clip.tags) == 2
    assert {tag.name for tag in clip.tags} == {"test", "demo"}


def test_transcode_video(video_service, sample_video_path, db_session):
    """Test transcoding a video file."""
    # Create test clip
    player_id = uuid.uuid4()
    clip_id, _ = video_service.upload_video(
        sample_video_path, sample_metadata, player_id, "Test Video"
    )

    # Mock progress callback
    progress_callback = MagicMock()

    # Transcode video
    success, error, transcoded = video_service.transcode_video(
        clip_id,
        width=1280,
        height=720,
        fps=30.0,
        codec="h264",
        crf=23,
        preset="medium",
        has_audio=True,
    )

    assert success
    assert error is None
    assert transcoded is not None
    assert transcoded.width == 1280
    assert transcoded.height == 720
    assert transcoded.fps == 30.0
    assert transcoded.codec == "h264"
    assert transcoded.crf == 23
    assert transcoded.preset == "medium"
    assert transcoded.has_audio is True
    assert transcoded.status == TranscodeStatus.PENDING


def test_start_transcode(video_service, sample_video_path, db_session):
    """Test starting a transcode operation."""
    # Create test clip and transcoded version
    player_id = uuid.uuid4()
    clip_id, _ = video_service.upload_video(
        sample_video_path, sample_metadata, player_id, "Test Video"
    )

    success, error, transcoded = video_service.transcode_video(
        clip_id, width=1280, height=720, fps=30.0, codec="h264"
    )

    # Mock progress callback
    progress_callback = MagicMock()

    # Start transcoding
    success, error = video_service.start_transcode(transcoded.id, progress_callback)

    assert success
    assert error is None

    # Verify status was updated
    db_session.refresh(transcoded)
    assert transcoded.status == TranscodeStatus.COMPLETED
    assert transcoded.progress == 100.0


def test_cancel_transcode(video_service, sample_video_path, db_session):
    """Test cancelling a transcode operation."""
    # Create test clip and transcoded version
    player_id = uuid.uuid4()
    clip_id, _ = video_service.upload_video(
        sample_video_path, sample_metadata, player_id, "Test Video"
    )

    success, error, transcoded = video_service.transcode_video(
        clip_id, width=1280, height=720, fps=30.0, codec="h264"
    )

    # Start transcoding in a way we can cancel
    with patch("spygate.video.transcoder.Transcoder.transcode") as mock_transcode:

        def mock_transcode_func(*args, **kwargs):
            # Simulate long-running operation
            import time

            time.sleep(0.1)

        mock_transcode.side_effect = mock_transcode_func

        # Start transcode
        success, error = video_service.start_transcode(transcoded.id)
        assert success

        # Cancel transcode
        success, error = video_service.cancel_transcode(transcoded.id)
        assert success
        assert error is None

        # Verify status was updated
        db_session.refresh(transcoded)
        assert transcoded.status == TranscodeStatus.CANCELLED

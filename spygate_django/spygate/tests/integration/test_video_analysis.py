"""Integration tests for the video analysis pipeline."""

import os
import shutil
import uuid
from datetime import datetime

import cv2
import numpy as np
import pytest

from ...database import AnalysisJob, AnalysisStatus, Clip, Tag, create_clip, get_db
from ...services.analysis_service import AnalysisService
from ...services.video_service import VideoService


@pytest.fixture(scope="module")
def test_video():
    """Create a test video file for analysis."""
    video_path = os.path.join("uploads", "videos", "test_analysis.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Create a 10-second test video with some motion
    width, height = 1280, 720
    fps = 30
    duration = 10  # seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        for i in range(duration * fps):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add moving rectangle to simulate motion
            x = int((i / (duration * fps)) * width)
            cv2.rectangle(
                frame,
                (x, height // 2),
                (x + 100, height // 2 + 100),
                (255, 255, 255),
                -1,
            )

            # Add simulated HUD
            cv2.putText(
                frame,
                f"1st & 10",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Q1 15:00",
                (width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "HOME 7 - AWAY 0",
                (width // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            out.write(frame)
    finally:
        out.release()

    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)


@pytest.fixture
def services():
    """Initialize services for testing."""
    video_service = VideoService()
    analysis_service = AnalysisService()
    return video_service, analysis_service


@pytest.fixture
def test_clip(test_video):
    """Create a test clip in the database."""
    clip = Clip(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        title="Test Analysis Clip",
        description="Test clip for analysis",
        file_path=test_video,
        duration=10.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db = next(get_db())
    db.add(clip)
    db.commit()

    yield clip

    # Cleanup
    db.delete(clip)
    db.commit()


def test_full_analysis_pipeline(services, test_clip):
    """Test the complete video analysis pipeline."""
    video_service, analysis_service = services

    # Start analysis
    job_id = analysis_service.start_analysis(test_clip.id)
    assert job_id is not None

    # Wait for analysis to complete (with timeout)
    import time

    timeout = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = analysis_service.get_analysis_status(job_id)
        if status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED):
            break
        time.sleep(1)

    # Get final status
    db = next(get_db())
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()

    assert job is not None
    assert job.status == AnalysisStatus.COMPLETED
    assert job.error_message is None

    # Check that clips were created
    clips = db.query(Clip).filter(Clip.source_clip_id == test_clip.id).all()

    assert len(clips) > 0

    # Verify clip properties
    for clip in clips:
        assert clip.file_path is not None
        assert os.path.exists(clip.file_path)
        assert clip.duration > 0
        assert clip.duration <= 20  # Should be around 10s (5s before + 5s after event)

        # Check tags
        assert len(clip.tags) > 0
        tag_names = [tag.name for tag in clip.tags]
        assert any(name in tag_names for name in ["motion", "play"])

        # Cleanup generated clip file
        if os.path.exists(clip.file_path):
            os.remove(clip.file_path)


def test_analysis_cancellation(services, test_clip):
    """Test cancelling an analysis job."""
    video_service, analysis_service = services

    # Start analysis
    job_id = analysis_service.start_analysis(test_clip.id)
    assert job_id is not None

    # Cancel the job immediately
    success = analysis_service.cancel_analysis(job_id)
    assert success is True

    # Verify job was cancelled
    db = next(get_db())
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()

    assert job is not None
    assert job.status == AnalysisStatus.CANCELLED

    # Check that no clips were created
    clips = db.query(Clip).filter(Clip.source_clip_id == test_clip.id).all()

    assert len(clips) == 0


def test_analysis_error_handling(services, test_clip):
    """Test handling of analysis errors."""
    video_service, analysis_service = services

    # Corrupt the video file to trigger an error
    with open(test_clip.file_path, "wb") as f:
        f.write(b"corrupted data")

    # Start analysis
    job_id = analysis_service.start_analysis(test_clip.id)
    assert job_id is not None

    # Wait for job to fail
    import time

    timeout = 30  # seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = analysis_service.get_analysis_status(job_id)
        if status == AnalysisStatus.FAILED:
            break
        time.sleep(1)

    # Verify job failed
    db = next(get_db())
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()

    assert job is not None
    assert job.status == AnalysisStatus.FAILED
    assert job.error_message is not None

    # Check that no clips were created
    clips = db.query(Clip).filter(Clip.source_clip_id == test_clip.id).all()

    assert len(clips) == 0

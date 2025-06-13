"""Unit tests for the AnalysisService class."""

import os
import uuid
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from ...database import AnalysisJob, AnalysisStatus, Clip, Tag
from ...services.analysis_service import AnalysisService


@pytest.fixture
def analysis_service():
    """Create an AnalysisService instance for testing."""
    return AnalysisService()


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.query = Mock()
    return session


@pytest.fixture
def sample_clip():
    """Create a sample clip for testing."""
    return Clip(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        title="Test Clip",
        description="Test Description",
        file_path="/path/to/test.mp4",
        duration=300.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_analysis_job(sample_clip):
    """Create a sample analysis job for testing."""
    return AnalysisJob(
        id=str(uuid.uuid4()),
        clip_id=sample_clip.id,
        status=AnalysisStatus.PENDING,
        metadata={"fps": 30.0, "frame_count": 9000, "duration": 300.0},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def test_start_analysis(analysis_service, mock_db_session, sample_clip):
    """Test starting a new analysis job."""
    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        job_id = analysis_service.start_analysis(sample_clip.id)

        assert job_id is not None
        assert isinstance(job_id, str)

        # Check that job was added to database
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Verify job was added to active jobs
        assert job_id in analysis_service._active_jobs
        assert analysis_service._active_jobs[job_id] is True


def test_cancel_analysis(analysis_service, mock_db_session, sample_analysis_job):
    """Test cancelling an analysis job."""
    analysis_service._active_jobs[sample_analysis_job.id] = True

    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        mock_db_session.query().filter().first.return_value = sample_analysis_job

        success = analysis_service.cancel_analysis(sample_analysis_job.id)
        assert success is True

        # Check job was marked as cancelled
        assert sample_analysis_job.status == AnalysisStatus.CANCELLED
        mock_db_session.commit.assert_called_once()

        # Check job was removed from active jobs
        assert sample_analysis_job.id not in analysis_service._active_jobs


def test_get_analysis_status(analysis_service, mock_db_session, sample_analysis_job):
    """Test getting analysis job status."""
    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        mock_db_session.query().filter().first.return_value = sample_analysis_job

        status = analysis_service.get_analysis_status(sample_analysis_job.id)
        assert status == sample_analysis_job.status


def test_process_frame_batch(analysis_service, mock_db_session, sample_analysis_job):
    """Test processing a batch of frames."""
    frames = [
        # Create 5 test frames
        *[Mock() for _ in range(5)]
    ]

    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        with patch("spygate.services.analysis_service.SituationDetector") as mock_detector:
            # Mock detector to return some test situations
            mock_detector.return_value.detect_situations.return_value = {
                "frame_number": 0,
                "timestamp": 0.0,
                "situations": [
                    {
                        "type": "high_motion_event",
                        "confidence": 0.9,
                        "frame": 0,
                        "timestamp": 0.0,
                        "details": {"motion_score": 75.0},
                    }
                ],
                "metadata": {"motion_score": 75.0},
            }

            results = analysis_service._process_frame_batch(
                frames, sample_analysis_job.id, 0, 30.0  # start_frame  # fps
            )

            assert len(results) == len(frames)
            assert all(isinstance(r, dict) for r in results)
            assert all("situations" in r for r in results)


def test_create_situation_clips(analysis_service, mock_db_session, sample_analysis_job):
    """Test creating clips from detected situations."""
    situations = [
        {
            "type": "high_motion_event",
            "confidence": 0.9,
            "frame": 150,  # 5 seconds at 30fps
            "timestamp": 5.0,
            "details": {"motion_score": 75.0},
        }
    ]

    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        with patch("spygate.services.analysis_service.ClipExtractor") as mock_extractor:
            mock_extractor.extract_segment.return_value = (True, None)

            clips = analysis_service._create_situation_clips(
                sample_analysis_job.id, situations, "/path/to/source.mp4"
            )

            assert len(clips) == len(situations)
            assert all(isinstance(c, dict) for c in clips)
            assert all("clip_id" in c for c in clips)
            assert all("situation" in c for c in clips)

            # Verify clip extraction was called
            mock_extractor.extract_segment.assert_called_once()

            # Verify clips were added to database
            assert mock_db_session.add.call_count == len(clips)
            mock_db_session.commit.assert_called_once()


def test_handle_analysis_error(analysis_service, mock_db_session, sample_analysis_job):
    """Test error handling during analysis."""
    error_msg = "Test error message"

    with patch("spygate.services.analysis_service.get_db", return_value=mock_db_session):
        mock_db_session.query().filter().first.return_value = sample_analysis_job

        analysis_service._handle_analysis_error(sample_analysis_job.id, error_msg)

        # Check job was marked as failed
        assert sample_analysis_job.status == AnalysisStatus.FAILED
        assert sample_analysis_job.error_message == error_msg
        mock_db_session.commit.assert_called_once()

        # Check job was removed from active jobs
        assert sample_analysis_job.id not in analysis_service._active_jobs

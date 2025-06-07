"""Tests for database models."""

import uuid
from datetime import datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from spygate.database.config import Base, DatabaseSession, engine
from spygate.database.models import (
    Clip,
    MotionEvent,
    Player,
    Tag,
    TranscodedClip,
    TranscodeStatus,
)
from spygate.database.utils import (
    cleanup_stale_transcodes,
    create_clip,
    create_motion_event,
    create_player,
    create_tag,
    create_transcoded_clip,
    find_transcoded_version,
    get_transcoded_versions,
    update_transcode_status,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Set up test database."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after tests
    Base.metadata.drop_all(bind=engine)


def test_player_model():
    """Test Player model."""
    session = DatabaseSession()

    # Create player
    player = Player(name="John Doe", team="Red Team", position="Forward")
    session.add(player)
    session.commit()

    # Verify fields
    assert isinstance(player.id, uuid.UUID)
    assert player.name == "John Doe"
    assert player.team == "Red Team"
    assert player.position == "Forward"
    assert isinstance(player.created_at, datetime)
    assert isinstance(player.updated_at, datetime)

    # Test relationships
    assert isinstance(player.clips, list)
    assert len(player.clips) == 0

    session.close()


def test_clip_model():
    """Test Clip model."""
    session = DatabaseSession()

    # Create player first
    player = Player(name="John Doe", team="Red Team", position="Forward")
    session.add(player)
    session.commit()

    # Create clip
    clip = Clip(file_path="/path/to/video.mp4", title="Test Clip", player_id=player.id)
    session.add(clip)
    session.commit()

    # Verify fields
    assert isinstance(clip.id, uuid.UUID)
    assert clip.file_path == "/path/to/video.mp4"
    assert clip.title == "Test Clip"
    assert isinstance(clip.created_at, datetime)
    assert isinstance(clip.updated_at, datetime)

    # Test relationships
    assert clip.player == player
    assert isinstance(clip.tags, list)
    assert len(clip.tags) == 0
    assert isinstance(clip.motion_events, list)
    assert len(clip.motion_events) == 0

    session.close()


def test_tag_model():
    """Test Tag model."""
    session = DatabaseSession()

    # Create tag
    tag = Tag(name="goal", description="Goal scoring moments")
    session.add(tag)
    session.commit()

    # Verify fields
    assert isinstance(tag.id, uuid.UUID)
    assert tag.name == "goal"
    assert tag.description == "Goal scoring moments"
    assert isinstance(tag.created_at, datetime)
    assert isinstance(tag.updated_at, datetime)

    # Test relationships
    assert isinstance(tag.clips, list)
    assert len(tag.clips) == 0

    # Test unique constraint on name
    with pytest.raises(IntegrityError):
        duplicate_tag = Tag(name="goal")
        session.add(duplicate_tag)
        session.commit()

    session.rollback()
    session.close()


def test_motion_event_model():
    """Test MotionEvent model."""
    session = DatabaseSession()

    # Create required objects
    player = Player(name="John Doe", team="Red Team", position="Forward")
    session.add(player)
    session.commit()

    clip = Clip(file_path="/path/to/video.mp4", title="Test Clip", player_id=player.id)
    session.add(clip)
    session.commit()

    # Create motion event
    event = MotionEvent(
        clip_id=clip.id,
        start_time=10.5,
        end_time=15.2,
        event_type="run",
        confidence=0.95,
        event_metadata={"speed": 20.5, "direction": "left"},
    )
    session.add(event)
    session.commit()

    # Verify fields
    assert isinstance(event.id, uuid.UUID)
    assert event.start_time == 10.5
    assert event.end_time == 15.2
    assert event.event_type == "run"
    assert event.confidence == 0.95
    assert event.event_metadata == {"speed": 20.5, "direction": "left"}
    assert isinstance(event.created_at, datetime)
    assert isinstance(event.updated_at, datetime)

    # Test relationships
    assert event.clip == clip

    session.close()


def test_clip_tag_relationship():
    """Test many-to-many relationship between Clip and Tag."""
    session = DatabaseSession()

    # Create player
    player = Player(name="John Doe", team="Red Team", position="Forward")
    session.add(player)
    session.commit()

    # Create clip
    clip = Clip(file_path="/path/to/video.mp4", title="Test Clip", player_id=player.id)
    session.add(clip)

    # Create tags
    tag1 = Tag(name="goal", description="Goal scoring moments")
    tag2 = Tag(name="assist", description="Assist moments")
    session.add_all([tag1, tag2])

    # Associate tags with clip
    clip.tags.extend([tag1, tag2])
    session.commit()

    # Verify relationships
    assert len(clip.tags) == 2
    assert clip.tags[0].name in ["goal", "assist"]
    assert clip.tags[1].name in ["goal", "assist"]
    assert clip in tag1.clips
    assert clip in tag2.clips

    # Test removing a tag
    clip.tags.remove(tag1)
    session.commit()
    assert len(clip.tags) == 1
    assert clip.tags[0] == tag2
    assert clip not in tag1.clips
    assert clip in tag2.clips

    session.close()


def test_create_transcoded_clip(db_session):
    """Test creating a transcoded clip."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)

    # Create transcoded version
    transcoded = create_transcoded_clip(
        clip.id,
        "test_720p.mp4",
        1280,
        720,
        30.0,
        "h264",
        crf=23,
        preset="medium",
        has_audio=True,
        session=db_session,
    )

    # Verify transcoded clip
    assert transcoded.original_clip_id == clip.id
    assert transcoded.file_path == "test_720p.mp4"
    assert transcoded.width == 1280
    assert transcoded.height == 720
    assert transcoded.fps == 30.0
    assert transcoded.codec == "h264"
    assert transcoded.crf == 23
    assert transcoded.preset == "medium"
    assert transcoded.has_audio is True
    assert transcoded.status == TranscodeStatus.PENDING
    assert transcoded.progress == 0.0
    assert transcoded.error_message is None
    assert transcoded.start_time is None
    assert transcoded.end_time is None


def test_update_transcode_status(db_session):
    """Test updating transcode status."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)
    transcoded = create_transcoded_clip(
        clip.id, "test_720p.mp4", 1280, 720, 30.0, "h264", session=db_session
    )

    # Update to in progress
    update_transcode_status(
        transcoded.id, TranscodeStatus.IN_PROGRESS, progress=50.0, session=db_session
    )

    # Verify status
    transcoded = db_session.query(TranscodedClip).get(transcoded.id)
    assert transcoded.status == TranscodeStatus.IN_PROGRESS
    assert transcoded.progress == 50.0
    assert transcoded.start_time is not None
    assert transcoded.end_time is None

    # Update to completed
    update_transcode_status(
        transcoded.id, TranscodeStatus.COMPLETED, progress=100.0, session=db_session
    )

    # Verify status
    transcoded = db_session.query(TranscodedClip).get(transcoded.id)
    assert transcoded.status == TranscodeStatus.COMPLETED
    assert transcoded.progress == 100.0
    assert transcoded.start_time is not None
    assert transcoded.end_time is not None


def test_update_transcode_status_with_error(db_session):
    """Test updating transcode status with error."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)
    transcoded = create_transcoded_clip(
        clip.id, "test_720p.mp4", 1280, 720, 30.0, "h264", session=db_session
    )

    # Update to failed with error
    error_msg = "Test error message"
    update_transcode_status(
        transcoded.id,
        TranscodeStatus.FAILED,
        error_message=error_msg,
        session=db_session,
    )

    # Verify status
    transcoded = db_session.query(TranscodedClip).get(transcoded.id)
    assert transcoded.status == TranscodeStatus.FAILED
    assert transcoded.error_message == error_msg
    assert transcoded.end_time is not None


def test_get_transcoded_versions(db_session):
    """Test getting transcoded versions of a clip."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)

    # Create multiple transcoded versions
    versions = [
        create_transcoded_clip(
            clip.id, f"test_{res}p.mp4", width, height, 30.0, "h264", session=db_session
        )
        for res, (width, height) in {
            "720": (1280, 720),
            "1080": (1920, 1080),
            "1440": (2560, 1440),
        }.items()
    ]

    # Get versions
    result = get_transcoded_versions(clip.id, session=db_session)

    # Verify versions
    assert len(result) == len(versions)
    for v in versions:
        assert v in result


def test_find_transcoded_version(db_session):
    """Test finding a specific transcoded version."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)

    # Create transcoded version
    transcoded = create_transcoded_clip(
        clip.id, "test_720p.mp4", 1280, 720, 30.0, "h264", session=db_session
    )

    # Find version
    result = find_transcoded_version(
        clip.id, 1280, 720, 30.0, "h264", session=db_session
    )

    # Verify result
    assert result == transcoded

    # Try finding non-existent version
    result = find_transcoded_version(
        clip.id, 3840, 2160, 60.0, "h265", session=db_session
    )
    assert result is None


def test_cleanup_stale_transcodes(db_session):
    """Test cleaning up stale transcoded clips."""
    # Create test data
    player = create_player("Test Player", "Test Team", "Forward", db_session)
    clip = create_clip("test.mp4", "Test Clip", player.id, [], db_session)

    # Create transcoded versions with different states
    versions = [
        # Pending for too long
        create_transcoded_clip(
            clip.id, "test_720p.mp4", 1280, 720, 30.0, "h264", session=db_session
        ),
        # In progress for too long
        create_transcoded_clip(
            clip.id, "test_1080p.mp4", 1920, 1080, 30.0, "h264", session=db_session
        ),
        # Recently started
        create_transcoded_clip(
            clip.id, "test_1440p.mp4", 2560, 1440, 30.0, "h264", session=db_session
        ),
        # Completed
        create_transcoded_clip(
            clip.id, "test_4k.mp4", 3840, 2160, 30.0, "h264", session=db_session
        ),
    ]

    # Set up states
    now = datetime.utcnow()
    old_time = now - timedelta(hours=2)

    # Pending for too long
    versions[0].created_at = old_time

    # In progress for too long
    update_transcode_status(
        versions[1].id, TranscodeStatus.IN_PROGRESS, session=db_session
    )
    versions[1].start_time = old_time

    # Recently started
    update_transcode_status(
        versions[2].id, TranscodeStatus.IN_PROGRESS, session=db_session
    )

    # Completed
    update_transcode_status(
        versions[3].id, TranscodeStatus.COMPLETED, session=db_session
    )

    # Clean up stale transcodes
    cleanup_stale_transcodes(max_age_hours=1, session=db_session)

    # Verify results
    versions = get_transcoded_versions(clip.id, session=db_session)
    assert len(versions) == 2  # Only recent and completed should remain
    statuses = {v.status for v in versions}
    assert TranscodeStatus.IN_PROGRESS in statuses  # Recent in-progress
    assert TranscodeStatus.COMPLETED in statuses  # Completed

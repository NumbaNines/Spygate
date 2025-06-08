"""Tests for database utility functions."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy.sql import text

from spygate.database.config import Base, engine
from spygate.database.models import Clip, MotionEvent, Player, Tag
from spygate.database.utils import (
    create_clip,
    create_motion_event,
    create_player,
    create_tag,
    get_clip,
    get_clips_by_player,
    get_clips_by_tag,
    get_clips_by_tags,
    get_db_session,
    get_motion_events,
    get_motion_events_for_clip,
    get_player,
    get_tag,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Set up test database."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after tests
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Create a test database session."""
    with get_db_session() as session:
        yield session


@pytest.fixture
def sample_player(db_session):
    """Create a sample player."""
    return create_player(name="John Doe", team="Red Team", position="Forward", session=db_session)


@pytest.fixture
def sample_clip(db_session, sample_player):
    """Create a sample clip."""
    return create_clip(
        file_path="/path/to/video.mp4",
        title="Test Clip",
        player_id=sample_player.id,
        tags=["goal", "assist"],
        session=db_session,
    )


def test_create_and_get_player(db_session):
    """Test creating and retrieving a player."""
    player = create_player(name="John Doe", team="Red Team", position="Forward", session=db_session)

    retrieved = get_player(player.id, db_session)
    assert retrieved.name == "John Doe"
    assert retrieved.team == "Red Team"
    assert retrieved.position == "Forward"


def test_create_and_get_clip(db_session, sample_player):
    """Test creating and retrieving a clip."""
    clip = create_clip(
        file_path="/path/to/video.mp4",
        title="Test Clip",
        player_id=sample_player.id,
        tags=["goal", "assist"],
        session=db_session,
    )

    retrieved = get_clip(clip.id, db_session)
    assert retrieved.file_path == "/path/to/video.mp4"
    assert retrieved.title == "Test Clip"
    assert retrieved.player_id == sample_player.id
    assert len(retrieved.tags) == 2
    assert {tag.name for tag in retrieved.tags} == {"goal", "assist"}


def test_create_and_get_motion_event(db_session, sample_clip):
    """Test creating and retrieving a motion event."""
    event = create_motion_event(
        clip_id=sample_clip.id,
        start_time=1.0,
        end_time=2.0,
        event_type="kick",
        confidence=0.9,
        event_metadata={"speed": 20.5},
        session=db_session,
    )

    events = get_motion_events(sample_clip.id, db_session)
    assert len(events) == 1
    assert events[0].start_time == 1.0
    assert events[0].end_time == 2.0
    assert events[0].event_type == "kick"
    assert events[0].confidence == 0.9
    assert events[0].event_metadata == {"speed": 20.5}


def test_create_and_get_tag(db_session):
    """Test creating and retrieving a tag."""
    tag = create_tag(name="goal", description="Goal scoring moments", session=db_session)

    retrieved = get_tag(tag.id, db_session)
    assert retrieved.name == "goal"
    assert retrieved.description == "Goal scoring moments"


def test_get_clips_by_tag(db_session, sample_clip):
    """Test retrieving clips by tag."""
    clips = get_clips_by_tag("goal", db_session)
    assert len(clips) == 1
    assert clips[0].id == sample_clip.id

    # Test nonexistent tag
    clips = get_clips_by_tag("nonexistent", db_session)
    assert len(clips) == 0


def test_get_clips_by_player(db_session, sample_player, sample_clip):
    """Test retrieving clips by player."""
    clips = get_clips_by_player(sample_player.id, db_session)
    assert len(clips) == 1
    assert clips[0].id == sample_clip.id


def test_get_clips_by_tags(db_session, sample_clip):
    """Test retrieving clips by multiple tags."""
    # Match any tag
    clips = get_clips_by_tags(["goal", "nonexistent"], session=db_session)
    assert len(clips) == 1
    assert clips[0].id == sample_clip.id

    # Match all tags
    clips = get_clips_by_tags(["goal", "assist"], session=db_session, match_all=True)
    assert len(clips) == 1
    assert clips[0].id == sample_clip.id

    # Match all tags (should return no results)
    clips = get_clips_by_tags(["goal", "nonexistent"], session=db_session, match_all=True)
    assert len(clips) == 0


def test_get_motion_events_for_clip(db_session, sample_clip):
    """Test retrieving motion events for a clip."""
    # Create multiple events
    event1 = create_motion_event(
        clip_id=sample_clip.id,
        start_time=1.0,
        end_time=2.0,
        event_type="kick",
        confidence=0.9,
        session=db_session,
    )
    event2 = create_motion_event(
        clip_id=sample_clip.id,
        start_time=3.0,
        end_time=4.0,
        event_type="run",
        confidence=0.8,
        session=db_session,
    )

    # Get all events
    events = get_motion_events_for_clip(sample_clip.id, db_session)
    assert len(events) == 2

    # Filter by type
    events = get_motion_events_for_clip(sample_clip.id, db_session, event_type="kick")
    assert len(events) == 1
    assert events[0].event_type == "kick"


def test_db_session_context_manager():
    """Test database session context manager."""
    # Test successful transaction
    with get_db_session() as session:
        player = Player(name="Test Player", team="Test Team", position="Test Position")
        session.add(player)

    # Verify the player was saved
    with get_db_session() as session:
        saved_player = session.query(Player).filter_by(name="Test Player").first()
        assert saved_player is not None
        assert saved_player.team == "Test Team"

    # Test transaction rollback
    with pytest.raises(ValueError):
        with get_db_session() as session:
            player = Player(name="Another Player", team="Another Team", position="Another Position")
            session.add(player)
            raise ValueError("Test error")

    # Verify the player was not saved
    with get_db_session() as session:
        player = session.query(Player).filter_by(name="Another Player").first()
        assert player is None

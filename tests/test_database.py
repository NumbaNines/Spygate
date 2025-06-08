"""Test database functionality."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import (
    create_clip,
    create_motion_event,
    create_player,
    create_tag,
    get_clip,
    get_clips_by_tag,
    get_motion_events,
    get_player,
    get_tag,
)
from src.database.models import Base, Clip, MotionEvent, Player, Tag


@pytest.fixture(scope="function")
def test_db():
    """Create a test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def test_create_player(test_db):
    """Test creating a player."""
    player = create_player(name="John Doe", team="Test Team", position="Forward", session=test_db)
    assert player.id is not None
    assert player.name == "John Doe"
    assert player.team == "Test Team"
    assert player.position == "Forward"


def test_get_player(test_db):
    """Test getting a player."""
    player = create_player("John Doe", session=test_db)
    retrieved_player = get_player(player.id, session=test_db)
    assert retrieved_player.name == "John Doe"


def test_create_clip(test_db):
    """Test creating a clip."""
    player = create_player("John Doe", session=test_db)
    clip = create_clip(
        file_path="test.mp4",
        title="Test Clip",
        player_id=player.id,
        tags=["tag1", "tag2"],
        session=test_db,
    )
    assert clip.id is not None
    assert clip.file_path == "test.mp4"
    assert clip.title == "Test Clip"
    assert clip.player_id == player.id
    assert len(clip.tags) == 2


def test_get_clip(test_db):
    """Test getting a clip."""
    player = create_player("John Doe", session=test_db)
    clip = create_clip("test.mp4", player_id=player.id, session=test_db)
    retrieved_clip = get_clip(clip.id, session=test_db)
    assert retrieved_clip.file_path == "test.mp4"


def test_create_motion_event(test_db):
    """Test creating a motion event."""
    player = create_player("John Doe", session=test_db)
    clip = create_clip("test.mp4", player_id=player.id, session=test_db)
    event = create_motion_event(
        clip_id=clip.id,
        start_time=1.0,
        end_time=2.0,
        event_type="test_event",
        confidence=0.9,
        event_metadata={"test": "data"},
        session=test_db,
    )
    assert event.id is not None
    assert event.clip_id == clip.id
    assert event.start_time == 1.0
    assert event.end_time == 2.0
    assert event.event_type == "test_event"
    assert event.confidence == 0.9
    assert event.event_metadata == {"test": "data"}


def test_get_motion_events(test_db):
    """Test getting motion events."""
    player = create_player("John Doe", session=test_db)
    clip = create_clip("test.mp4", player_id=player.id, session=test_db)
    event1 = create_motion_event(
        clip_id=clip.id,
        start_time=1.0,
        end_time=2.0,
        event_type="test_event",
        confidence=0.9,
        session=test_db,
    )
    event2 = create_motion_event(
        clip_id=clip.id,
        start_time=3.0,
        end_time=4.0,
        event_type="test_event",
        confidence=0.8,
        session=test_db,
    )
    events = get_motion_events(clip.id, session=test_db)
    assert len(events) == 2
    assert events[0].start_time == 1.0
    assert events[1].start_time == 3.0


def test_create_tag(test_db):
    """Test creating a tag."""
    tag = create_tag(name="test_tag", description="Test Description", session=test_db)
    assert tag.id is not None
    assert tag.name == "test_tag"
    assert tag.description == "Test Description"


def test_get_tag(test_db):
    """Test getting a tag."""
    tag = create_tag("test_tag", session=test_db)
    retrieved_tag = get_tag(tag.id, session=test_db)
    assert retrieved_tag.name == "test_tag"


def test_get_clips_by_tag(test_db):
    """Test getting clips by tag."""
    player = create_player("John Doe", session=test_db)
    clip1 = create_clip("test1.mp4", player_id=player.id, tags=["tag1"], session=test_db)
    clip2 = create_clip("test2.mp4", player_id=player.id, tags=["tag1", "tag2"], session=test_db)
    clips = get_clips_by_tag("tag1", session=test_db)
    assert len(clips) == 2
    assert clips[0].file_path == "test1.mp4"
    assert clips[1].file_path == "test2.mp4"

    clips = get_clips_by_tag("tag2", session=test_db)
    assert len(clips) == 1
    assert clips[0].file_path == "test2.mp4"

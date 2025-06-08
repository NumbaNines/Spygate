"""Test suite for video service and database integration."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from spygate.database.schema import Base, Player, Video, video_players
from spygate.database.video_manager import VideoManager
from spygate.services.video_service import VideoService
from spygate.video.metadata import VideoMetadata

from .utils.test_utils import (
    cleanup_test_files,
    create_test_files,
    create_test_metadata,
)


# Database fixtures
@pytest.fixture
def db_engine():
    """Create in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def video_manager(db_session):
    """Create video manager with test session."""
    return VideoManager(session=db_session)


@pytest.fixture
def video_service(video_manager):
    """Create video service with test video manager."""
    return VideoService(video_manager=video_manager)


@pytest.fixture
def test_files(tmp_path) -> Dict[str, str]:
    """Create test files."""
    files = create_test_files(tmp_path)
    yield files
    cleanup_test_files(files)


# Basic Service Tests
def test_service_initialization():
    """Test video service initialization."""
    service = VideoService()
    assert service.video_manager is not None

    manager = VideoManager()
    service = VideoService(video_manager=manager)
    assert service.video_manager == manager


def test_import_single_video(video_service, test_files, db_session):
    """Test importing a single video."""
    metadata = create_test_metadata()

    # Import video
    video_service.import_video(
        file_path=test_files["valid_video"],
        metadata=metadata,
        player_name="Test Player",
        team="Test Team",
    )

    # Verify database entry
    video = db_session.query(Video).first()
    assert video is not None
    assert video.file_path == test_files["valid_video"]
    assert video.width == metadata.width
    assert video.height == metadata.height
    assert video.fps == metadata.fps
    assert video.duration == metadata.duration
    assert video.codec == metadata.codec

    # Verify player
    player = db_session.query(Player).first()
    assert player is not None
    assert player.name == "Test Player"
    assert player.team == "Test Team"

    # Verify relationship
    assert video in player.videos


def test_import_multiple_videos(video_service, test_files, db_session):
    """Test importing multiple videos."""
    metadata = create_test_metadata()
    videos = [
        (test_files["valid_video"], metadata, "Player 1", "Team A"),
        (test_files["valid_video"], metadata, "Player 2", "Team B"),
    ]

    # Import videos
    for file_path, meta, player, team in videos:
        video_service.import_video(
            file_path=file_path, metadata=meta, player_name=player, team=team
        )

    # Verify database entries
    assert db_session.query(Video).count() == 2
    assert db_session.query(Player).count() == 2

    # Verify relationships
    for player in db_session.query(Player).all():
        assert len(player.videos) == 1


def test_import_same_player(video_service, test_files, db_session):
    """Test importing multiple videos for same player."""
    metadata = create_test_metadata()

    # Import two videos for same player
    for i in range(2):
        video_service.import_video(
            file_path=test_files["valid_video"],
            metadata=metadata,
            player_name="Same Player",
            team="Team A",
        )

    # Verify database entries
    assert db_session.query(Video).count() == 2
    assert db_session.query(Player).count() == 1

    # Verify relationships
    player = db_session.query(Player).first()
    assert len(player.videos) == 2


# Error Handling Tests
def test_import_invalid_file(video_service, test_files):
    """Test importing invalid file."""
    metadata = create_test_metadata()

    with pytest.raises(Exception) as exc:
        video_service.import_video(
            file_path=test_files["invalid_video"],
            metadata=metadata,
            player_name="Test Player",
        )
    assert "invalid" in str(exc.value).lower()


def test_import_missing_file(video_service, test_files):
    """Test importing missing file."""
    metadata = create_test_metadata()

    with pytest.raises(FileNotFoundError):
        video_service.import_video(
            file_path=test_files["missing"],
            metadata=metadata,
            player_name="Test Player",
        )


def test_import_duplicate_file(video_service, test_files, db_session):
    """Test importing same file twice."""
    metadata = create_test_metadata()

    # First import should succeed
    video_service.import_video(
        file_path=test_files["valid_video"],
        metadata=metadata,
        player_name="Test Player",
    )

    # Second import should fail
    with pytest.raises(Exception) as exc:
        video_service.import_video(
            file_path=test_files["valid_video"],
            metadata=metadata,
            player_name="Test Player",
        )
    assert "already exists" in str(exc.value).lower()


# Database Integration Tests
def test_player_creation(video_manager):
    """Test player creation and retrieval."""
    # Create player
    player1 = video_manager.get_or_create_player(
        name="Test Player", team="Test Team", is_self=False
    )
    assert player1.name == "Test Player"
    assert player1.team == "Test Team"

    # Get existing player
    player2 = video_manager.get_or_create_player(name="Test Player", team="Test Team")
    assert player1 == player2


def test_video_metadata_storage(video_service, test_files, db_session):
    """Test video metadata storage and retrieval."""
    metadata = create_test_metadata(
        width=1920, height=1080, fps=60.0, duration=120.0, codec="h265"
    )

    # Import video
    video_service.import_video(
        file_path=test_files["valid_video"],
        metadata=metadata,
        player_name="Test Player",
    )

    # Verify stored metadata
    video = db_session.query(Video).first()
    assert video.width == 1920
    assert video.height == 1080
    assert video.fps == 60.0
    assert video.duration == 120.0
    assert video.codec == "h265"


def test_video_player_relationship(video_service, test_files, db_session):
    """Test video-player relationship management."""
    metadata = create_test_metadata()

    # Import video with multiple players
    video_service.import_video(
        file_path=test_files["valid_video"],
        metadata=metadata,
        player_name="Player 1",
        additional_players=["Player 2", "Player 3"],
    )

    # Verify relationships
    video = db_session.query(Video).first()
    assert len(video.players) == 3

    player_names = {p.name for p in video.players}
    assert player_names == {"Player 1", "Player 2", "Player 3"}


# Transaction Tests
def test_transaction_rollback(video_service, test_files, db_session):
    """Test transaction rollback on error."""
    metadata = create_test_metadata()

    # Force error during import
    with patch.object(
        video_service.video_manager, "add_video", side_effect=Exception("Forced error")
    ):
        with pytest.raises(Exception):
            video_service.import_video(
                file_path=test_files["valid_video"],
                metadata=metadata,
                player_name="Test Player",
            )

    # Verify no data was committed
    assert db_session.query(Video).count() == 0
    assert db_session.query(Player).count() == 0


def test_cleanup_on_error(video_service, test_files, db_session):
    """Test cleanup of temporary files on error."""
    metadata = create_test_metadata()

    # Force error during import
    with patch.object(
        video_service.video_manager, "add_video", side_effect=Exception("Forced error")
    ):
        with pytest.raises(Exception):
            video_service.import_video(
                file_path=test_files["valid_video"],
                metadata=metadata,
                player_name="Test Player",
            )

    # Verify temporary files are cleaned up
    assert not os.path.exists(test_files["valid_video"] + ".tmp")
    assert not os.path.exists(test_files["valid_video"] + ".processing")

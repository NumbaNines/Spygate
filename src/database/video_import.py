"""Database operations for video import system."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..video.codec_validator import VideoMetadata
from .config import DatabaseSession
from .models import Clip, Tag


def create_video_import(
    file_path: str,
    metadata: VideoMetadata,
    player_name: str,
    title: str,
    tags: Optional[list[str]] = None,
    session: Optional[Session] = None,
) -> Clip:
    """Create a new video import entry in the database.

    Args:
        file_path: Path where the video file will be stored
        metadata: Validated video metadata
        player_name: Name of the player ("Self" or "Opponent: Name")
        title: Title for the video clip
        tags: Optional list of tags to apply
        session: Optional database session (creates new one if not provided)

    Returns:
        Clip: The created clip object

    Raises:
        SQLAlchemyError: If database operations fail
    """
    if session is None:
        session = DatabaseSession()
        should_close = True
    else:
        should_close = False

    try:
        # Create the clip
        clip = Clip(
            file_path=file_path,
            title=title,
            player_name=player_name,
            duration=int(metadata.duration),
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            codec=metadata.codec,
            is_processed=False,
            created_at=datetime.utcnow(),
        )
        session.add(clip)

        # Add tags if provided
        if tags:
            for tag_name in tags:
                # Get or create tag
                tag = session.query(Tag).filter_by(name=tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    session.add(tag)
                clip.tags.append(tag)

        session.commit()
        return clip

    except SQLAlchemyError as e:
        session.rollback()
        raise

    finally:
        if should_close:
            session.close()


def get_video_import(clip_id: UUID, session: Optional[Session] = None) -> Optional[Clip]:
    """Get a video import by its ID.

    Args:
        clip_id: UUID of the clip to retrieve
        session: Optional database session (creates new one if not provided)

    Returns:
        Optional[Clip]: The clip if found, None otherwise
    """
    if session is None:
        session = DatabaseSession()
        should_close = True
    else:
        should_close = False

    try:
        clip = session.query(Clip).filter_by(id=clip_id).first()
        return clip

    finally:
        if should_close:
            session.close()


def update_video_import_status(
    clip_id: UUID,
    is_processed: bool,
    error_message: Optional[str] = None,
    session: Optional[Session] = None,
) -> bool:
    """Update the processing status of a video import.

    Args:
        clip_id: UUID of the clip to update
        is_processed: Whether the video has been processed
        error_message: Optional error message if processing failed
        session: Optional database session (creates new one if not provided)

    Returns:
        bool: True if update was successful, False otherwise
    """
    if session is None:
        session = DatabaseSession()
        should_close = True
    else:
        should_close = False

    try:
        clip = session.query(Clip).filter_by(id=clip_id).first()
        if not clip:
            return False

        clip.is_processed = is_processed
        if error_message:
            clip.error_message = error_message
        clip.updated_at = datetime.utcnow()

        session.commit()
        return True

    except SQLAlchemyError:
        session.rollback()
        return False

    finally:
        if should_close:
            session.close()


def update_video_thumbnail(
    clip_id: UUID, thumbnail_path: str, session: Optional[Session] = None
) -> bool:
    """Update the thumbnail path for a video import.

    Args:
        clip_id: UUID of the clip to update
        thumbnail_path: Path to the generated thumbnail
        session: Optional database session (creates new one if not provided)

    Returns:
        bool: True if update was successful, False otherwise
    """
    if session is None:
        session = DatabaseSession()
        should_close = True
    else:
        should_close = False

    try:
        clip = session.query(Clip).filter_by(id=clip_id).first()
        if not clip:
            return False

        clip.thumbnail_path = thumbnail_path
        clip.updated_at = datetime.utcnow()

        session.commit()
        return True

    except SQLAlchemyError:
        session.rollback()
        return False

    finally:
        if should_close:
            session.close()

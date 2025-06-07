"""Database operations for the application."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from .models import Clip, MotionEvent, Tag, TranscodedClip, TranscodeStatus


def create_clip(
    file_path: str,
    title: str,
    player_name: str,  # "Self" or "Opponent: <gamertag>"
    session: Session,
    tags: List[str] = None,
    description: Optional[str] = None,
    duration: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    codec: Optional[str] = None,
    bitrate: Optional[int] = None,
    thumbnail_path: Optional[str] = None,
    thumbnail_updated_at: Optional[datetime] = None,
) -> Clip:
    """Create a new clip in the database.

    Args:
        file_path: Path to the video file
        title: Title for the clip
        player_name: Name of the player ("Self" or "Opponent: <gamertag>")
        session: SQLAlchemy session
        tags: Optional list of tag names
        description: Optional description for the clip
        duration: Optional video duration in seconds
        width: Optional video width in pixels
        height: Optional video height in pixels
        fps: Optional frames per second
        codec: Optional video codec name
        bitrate: Optional video bitrate
        thumbnail_path: Optional path to thumbnail image
        thumbnail_updated_at: Optional timestamp of last thumbnail update

    Returns:
        Clip: The created clip
    """
    # Create clip
    clip = Clip(
        file_path=file_path,
        title=title,
        player_name=player_name,
        description=description,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        bitrate=bitrate,
        thumbnail_path=thumbnail_path,
        thumbnail_updated_at=thumbnail_updated_at,
    )

    # Add tags
    if tags:
        for tag_name in tags:
            # Get or create tag
            tag = session.query(Tag).filter_by(name=tag_name).first()
            if not tag:
                tag = Tag(name=tag_name)
                session.add(tag)
            clip.tags.append(tag)

    # Add to session and commit
    session.add(clip)
    session.commit()

    return clip


def create_transcoded_clip(
    original_clip_id: str,
    file_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str,
    session: Session,
    crf: Optional[int] = None,
    preset: Optional[str] = None,
    has_audio: bool = True,
) -> TranscodedClip:
    """Create a new transcoded clip entry.

    Args:
        original_clip_id: ID of the original clip
        file_path: Path to the transcoded file
        width: Target width
        height: Target height
        fps: Target FPS
        codec: Target codec
        session: SQLAlchemy session
        crf: Optional Constant Rate Factor
        preset: Optional encoding preset
        has_audio: Whether audio was included

    Returns:
        TranscodedClip: The created transcoded clip entry
    """
    transcoded = TranscodedClip(
        original_clip_id=original_clip_id,
        file_path=file_path,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        crf=crf,
        preset=preset,
        has_audio=has_audio,
    )

    session.add(transcoded)
    session.flush()  # Get the ID without committing

    return transcoded


def update_transcode_status(
    transcode_id: str,
    status: TranscodeStatus,
    session: Session,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update the status of a transcoding operation.

    Args:
        transcode_id: ID of the transcoded clip
        status: New status
        session: SQLAlchemy session
        progress: Optional progress percentage (0-100)
        error_message: Optional error message if failed
    """
    transcoded = session.query(TranscodedClip).get(transcode_id)
    if transcoded:
        transcoded.status = status
        if progress is not None:
            transcoded.progress = progress
        if error_message is not None:
            transcoded.error_message = error_message
        session.flush()

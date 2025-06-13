"""Database utility functions."""

from typing import List, Optional

import bcrypt
from sqlalchemy.orm import Session

from .models import Clip, Tag, TranscodedClip, TranscodeStatus


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_clip(
    file_path: str,
    title: str,
    player_name: str,
    session: Session,
    tags: list[str] = None,
    description: Optional[str] = None,
) -> Clip:
    """Create a new clip in the database."""
    # Create clip
    clip = Clip(
        title=title,
        file_path=file_path,
        player_name=player_name,
        description=description,
    )
    session.add(clip)

    # Add tags
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


def create_transcoded_clip(
    original_clip_id: int,
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
    """Create a new transcoded clip in the database."""
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
        status=TranscodeStatus.PENDING,
    )
    session.add(transcoded)
    session.commit()
    return transcoded


def update_transcode_status(
    transcoded_id: int,
    status: TranscodeStatus,
    session: Session,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update the status of a transcoded clip."""
    transcoded = session.query(TranscodedClip).get(transcoded_id)
    if transcoded:
        transcoded.status = status
        if progress is not None:
            transcoded.progress = progress
        if error_message is not None:
            transcoded.error_message = error_message
        session.commit()

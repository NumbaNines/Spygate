"""Database models for Spygate."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from .database import Base

# Association table for clips and tags
clip_tags = Table(
    "clip_tags",
    Base.metadata,
    Column("clip_id", String, ForeignKey("clips.id")),
    Column("tag_id", String, ForeignKey("tags.id")),
)


class TranscodeStatus(enum.Enum):
    """Status of video transcoding."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisStatus(enum.Enum):
    """Status of video analysis."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clips = relationship("Clip", back_populates="user")


class Clip(Base):
    """Video clip model."""

    __tablename__ = "clips"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String)
    description = Column(String)
    file_path = Column(String)
    duration = Column(Float)
    source_clip_id = Column(String, ForeignKey("clips.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="clips")
    tags = relationship("Tag", secondary=clip_tags, back_populates="clips")
    transcoded_clips = relationship("TranscodedClip", back_populates="clip")
    analysis_jobs = relationship("AnalysisJob", back_populates="clip")


class TranscodedClip(Base):
    """Transcoded video clip model."""

    __tablename__ = "transcoded_clips"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    clip_id = Column(String, ForeignKey("clips.id"))
    file_path = Column(String)
    format = Column(String)
    resolution = Column(String)
    bitrate = Column(Integer)
    status = Column(Enum(TranscodeStatus), default=TranscodeStatus.PENDING)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clip = relationship("Clip", back_populates="transcoded_clips")


class Tag(Base):
    """Tag model for clips."""

    __tablename__ = "tags"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clips = relationship("Clip", secondary=clip_tags, back_populates="tags")


class AnalysisJob(Base):
    """Video analysis job model."""

    __tablename__ = "analysis_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    clip_id = Column(String, ForeignKey("clips.id"))
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    analysis_metadata = Column(JSON, default=dict)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clip = relationship("Clip", back_populates="analysis_jobs")


def create_clip(
    db, user_id, title, description, file_path, duration, source_clip_id=None
):
    """Create a new clip.

    Args:
        db: Database session
        user_id: ID of the user who owns the clip
        title: Clip title
        description: Clip description
        file_path: Path to the clip file
        duration: Duration in seconds
        source_clip_id: Optional ID of the source clip

    Returns:
        Clip: The created clip
    """
    clip = Clip(
        user_id=user_id,
        title=title,
        description=description,
        file_path=file_path,
        duration=duration,
        source_clip_id=source_clip_id,
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    return clip


def create_transcoded_clip(db, clip_id, file_path, format, resolution, bitrate):
    """Create a new transcoded clip.

    Args:
        db: Database session
        clip_id: ID of the original clip
        file_path: Path to the transcoded file
        format: Video format
        resolution: Video resolution
        bitrate: Video bitrate

    Returns:
        TranscodedClip: The created transcoded clip
    """
    transcoded = TranscodedClip(
        clip_id=clip_id,
        file_path=file_path,
        format=format,
        resolution=resolution,
        bitrate=bitrate,
    )
    db.add(transcoded)
    db.commit()
    db.refresh(transcoded)
    return transcoded


def update_transcode_status(db, clip_id, status, error_message=None):
    """Update the status of a transcoded clip.

    Args:
        db: Database session
        clip_id: ID of the transcoded clip
        status: New status
        error_message: Optional error message

    Returns:
        TranscodedClip: The updated transcoded clip
    """
    transcoded = (
        db.query(TranscodedClip).filter(TranscodedClip.clip_id == clip_id).first()
    )
    if transcoded:
        transcoded.status = status
        transcoded.error_message = error_message
        transcoded.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(transcoded)
    return transcoded

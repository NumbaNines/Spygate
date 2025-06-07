"""Database models for the application."""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database.base import Base

# Association tables for many-to-many relationships
clip_tags = Table(
    "clip_tags",
    Base.metadata,
    Column("clip_id", UUID(as_uuid=True), ForeignKey("clips.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)


class Player(Base):
    """Player model for storing player information."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team = Column(String(100))
    position = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    clips = relationship("Clip", back_populates="player")


class Clip(Base):
    """Clip model for storing video clip information."""

    __tablename__ = "clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    duration = Column(Integer)  # Duration in seconds
    player_id = Column(Integer, ForeignKey("players.id"))
    player_name = Column(String(100))  # "Self" or "Opponent: Name"
    thumbnail_path = Column(String(500))
    formation = Column(String(100))
    play_type = Column(String(100))
    situation = Column(String(200))
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)  # Used as upload_date
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    player = relationship("Player", back_populates="clips")
    tags = relationship("Tag", secondary=clip_tags, back_populates="clips")
    motion_events = relationship("MotionEvent", back_populates="clip")


class Tag(Base):
    """Tag model for categorizing clips."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    clips = relationship("Clip", secondary=clip_tags, back_populates="tags")


class MotionEvent(Base):
    """Motion event model for storing detected motion in clips."""

    __tablename__ = "motion_events"

    id = Column(Integer, primary_key=True)
    clip_id = Column(UUID(as_uuid=True), ForeignKey("clips.id"))
    start_time = Column(Integer)  # Start time in milliseconds
    end_time = Column(Integer)  # End time in milliseconds
    event_type = Column(String(100))
    confidence = Column(Integer)  # Confidence score (0-100)
    event_metadata = Column(String(1000))  # JSON string of additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    clip = relationship("Clip", back_populates="motion_events")


class ImportStatus(Enum):
    """Enum for video import status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    """Model for storing video file information."""

    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String, nullable=False)
    player_name = Column(String, nullable=False)  # "Self" or "Opponent: Name"
    import_status = Column(
        SQLEnum(ImportStatus), nullable=False, default=ImportStatus.PENDING
    )
    error_message = Column(String)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    video_metadata = relationship(
        "VideoMetadata",
        back_populates="video",
        uselist=False,
        cascade="all, delete-orphan",
    )


class VideoMetadata(Base):
    """Model for storing video metadata."""

    __tablename__ = "video_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    codec = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    video = relationship("Video", back_populates="video_metadata")

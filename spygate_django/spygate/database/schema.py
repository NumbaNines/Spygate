"""
Database schema for Spygate application.
Defines SQLite tables for storing video metadata, analysis results, and user data.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Association tables
video_tags = Table(
    "video_tags",
    Base.metadata,
    Column("video_id", Integer, ForeignKey("videos.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

video_players = Table(
    "video_players",
    Base.metadata,
    Column("video_id", Integer, ForeignKey("videos.id"), primary_key=True),
    Column("player_id", Integer, ForeignKey("players.id"), primary_key=True),
    Column("is_primary", Boolean, default=False),  # Indicates if this is the primary player
)


class Player(Base):
    """Player model for storing player information."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team = Column(String(100))
    is_self = Column(Boolean, default=False)
    gamertag = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    videos = relationship("Video", secondary="video_players", back_populates="players")


class Tag(Base):
    """Tag model for categorizing videos."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    videos = relationship("Video", secondary="video_tags", back_populates="tags")


class Video(Base):
    """Video model for storing video metadata and file information."""

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)

    # File information
    file_path = Column(String(255), nullable=False, unique=True)
    file_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hash
    file_size = Column(Integer, nullable=False)  # Size in bytes
    original_filename = Column(String(255), nullable=False)

    # Import metadata
    import_date = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Video properties
    duration = Column(Float, nullable=False)  # Duration in seconds
    frame_count = Column(Integer, nullable=False)
    fps = Column(Float, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    codec = Column(String(50), nullable=False)

    # Additional properties
    bit_rate = Column(Integer)  # Bits per second
    has_audio = Column(Boolean, default=False)
    audio_codec = Column(String(50))

    # Status and organization
    is_deleted = Column(Boolean, default=False)
    delete_date = Column(DateTime)
    notes = Column(Text)

    # Relationships
    analysis_jobs = relationship("AnalysisJob", back_populates="video")
    tags = relationship("Tag", secondary="video_tags", back_populates="videos")
    players = relationship("Player", secondary="video_players", back_populates="videos")


class AnalysisJob(Base):
    """Model for tracking video analysis jobs."""

    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    error_message = Column(Text)

    # Relationships
    video = relationship("Video", back_populates="analysis_jobs")


class ImportLog(Base):
    """Log of video import operations."""

    __tablename__ = "import_logs"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    operation = Column(String, nullable=False)
    status = Column(String, nullable=False)  # success, failed
    message = Column(Text)
    details = Column(JSON)  # Additional details as JSON


class Clip(Base):
    """Video clip segments with analysis results."""

    __tablename__ = "clips"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)  # End time in seconds
    situation = Column(String)  # e.g., "No Huddle", "Two-Minute Drill"
    formation = Column(String)  # e.g., "Shotgun", "I-Formation"
    play_type = Column(String)  # e.g., "Pass", "Run"
    notes = Column(String)  # User notes or analysis results
    video = relationship("Video", back_populates="clips")
    tags = relationship("Tag", secondary="clip_tags", back_populates="clips")


# Association table for many-to-many relationship between clips and tags
clip_tags = Table(
    "clip_tags",
    Base.metadata,
    Column("clip_id", Integer, ForeignKey("clips.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

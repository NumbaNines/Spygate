from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MotionEvent(Base):
    """Model for storing motion detection events."""

    __tablename__ = "motion_events"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # Seconds from video start
    event_time = Column(DateTime, default=datetime.utcnow)
    motion_score = Column(Float, nullable=False)
    hardware_tier = Column(String(50))
    processing_fps = Column(Float)
    regions = Column(JSON)  # Store motion regions as JSON
    metadata = Column(JSON)  # Additional metadata

    # Relationships
    video = relationship("Video", back_populates="motion_events")
    situations = relationship("Situation", back_populates="motion_event")


class Situation(Base):
    """Model for storing detected game situations."""

    __tablename__ = "situations"

    id = Column(Integer, primary_key=True)
    motion_event_id = Column(Integer, ForeignKey("motion_events.id"), nullable=False)
    type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    field_region = Column(String(50))
    direction = Column(String(50))
    speed = Column(Float)
    duration = Column(Float)  # For sustained situations
    details = Column(JSON)  # Additional situation-specific details

    # Relationships
    motion_event = relationship("MotionEvent", back_populates="situations")


class MotionHeatmap(Base):
    """Model for storing aggregated motion heatmaps."""

    __tablename__ = "motion_heatmaps"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)  # Seconds from video start
    end_time = Column(Float, nullable=False)  # Seconds from video start
    heatmap_data = Column(JSON, nullable=False)  # Compressed heatmap data
    resolution = Column(JSON, nullable=False)  # Original frame resolution
    metadata = Column(JSON)  # Additional metadata

    # Relationships
    video = relationship("Video", back_populates="heatmaps")


class MotionPattern(Base):
    """Model for storing identified motion patterns."""

    __tablename__ = "motion_patterns"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    pattern_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    field_region = Column(String(50))
    direction = Column(String(50))
    speed = Column(Float)
    duration = Column(Float)
    trajectory = Column(JSON)  # Store motion trajectory points
    metadata = Column(JSON)  # Additional metadata

    # Relationships
    video = relationship("Video", back_populates="motion_patterns")


# Add relationships to Video model
from .video import Video

Video.motion_events = relationship("MotionEvent", back_populates="video")
Video.heatmaps = relationship("MotionHeatmap", back_populates="video")
Video.motion_patterns = relationship("MotionPattern", back_populates="video")

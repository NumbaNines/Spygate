from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Table
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Association tables
clip_tags = Table(
    "clip_tags",
    Base.metadata,
    Column("clip_id", Integer, ForeignKey("clips.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

clip_collections = Table(
    "clip_collections",
    Base.metadata,
    Column("clip_id", Integer, ForeignKey("clips.id"), primary_key=True),
    Column("collection_id", Integer, ForeignKey("collections.id"), primary_key=True),
)


class BaseModel:
    """Base model with common fields"""

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + "s"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Clip(Base, BaseModel):
    """Model for video clips"""

    title = Column(String(255), nullable=False)
    description = Column(String(1000))
    filename = Column(String(255), nullable=False)
    duration = Column(Float, nullable=False)
    player_name = Column(String(100))
    view_count = Column(Integer, default=0)
    thumbnail_path = Column(String(255))

    # Watch progress tracking
    watch_progress = Column(Float, default=0.0)  # Progress as percentage
    last_watched = Column(DateTime)

    # Relationships
    tags = relationship("Tag", secondary=clip_tags, back_populates="clips")
    collections = relationship("Collection", secondary=clip_collections, back_populates="clips")
    situations = relationship("Situation", back_populates="clip")
    formations = relationship("Formation", back_populates="clip")


class Tag(Base, BaseModel):
    """Model for clip tags"""

    name = Column(String(50), nullable=False, unique=True)
    description = Column(String(255))

    # Relationships
    clips = relationship("Clip", secondary=clip_tags, back_populates="tags")


class Collection(Base, BaseModel):
    """Model for clip collections/playlists"""

    name = Column(String(100), nullable=False)
    description = Column(String(500))
    is_playlist = Column(Boolean, default=False)

    # Relationships
    clips = relationship("Clip", secondary=clip_collections, back_populates="collections")


class Situation(Base, BaseModel):
    """Model for detected game situations"""

    clip_id = Column(Integer, ForeignKey("clips.id"))
    type = Column(String(50), nullable=False)  # e.g., 'offense', 'defense'
    confidence = Column(Float)
    timestamp = Column(Float)  # Time in video where situation was detected

    # Relationships
    clip = relationship("Clip", back_populates="situations")


class Formation(Base, BaseModel):
    """Model for detected formations"""

    clip_id = Column(Integer, ForeignKey("clips.id"))
    name = Column(String(50), nullable=False)  # e.g., 'spread', '4-3'
    confidence = Column(Float)
    timestamp = Column(Float)  # Time in video where formation was detected

    # Relationships
    clip = relationship("Clip", back_populates="formations")


class WatchHistory(Base, BaseModel):
    """Model for detailed watch history"""

    clip_id = Column(Integer, ForeignKey("clips.id"))
    timestamp = Column(Float)  # Position in video
    action = Column(String(20))  # e.g., 'play', 'pause', 'seek'

    # Relationships
    clip = relationship("Clip")


class ShareHistory(Base, BaseModel):
    """Model for tracking clip shares"""

    clip_id = Column(Integer, ForeignKey("clips.id"))
    platform = Column(String(50))  # e.g., 'discord'
    channel = Column(String(255))  # e.g., discord channel ID
    shared_by = Column(String(100))

    # Relationships
    clip = relationship("Clip")

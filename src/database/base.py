"""SQLAlchemy declarative base configuration."""

from sqlalchemy.ext.declarative import declarative_base

# Create base class for declarative models
Base = declarative_base()

# Import all models to ensure they're registered with Base
from src.database.models import Video, VideoMetadata  # noqa

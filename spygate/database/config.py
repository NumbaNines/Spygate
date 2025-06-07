"""Database configuration for Spygate."""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from .models import Base

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Create data directory if it doesn't exist
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database file path
DB_PATH = PROJECT_ROOT / "spygate.db"

# Create database engine
engine = create_engine(f"sqlite:///{DB_PATH}", echo=True)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)


def init_db():
    """Initialize the database."""
    Base.metadata.create_all(engine)


def get_db():
    """Get a database session."""
    return Session()

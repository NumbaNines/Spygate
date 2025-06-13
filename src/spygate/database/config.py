"""
Database configuration and initialization for Spygate application.
"""

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .schema import Base

logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = Path("data/spygate.db")
DB_URL = f"sqlite:///{DB_PATH}"


def get_engine():
    """Create and return SQLAlchemy engine instance."""
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create engine with SQLite optimizations
    engine = create_engine(
        DB_URL,
        connect_args={"check_same_thread": False},  # Allow multi-threading
        echo=False,  # Set to True for SQL query logging
    )
    return engine


def init_db():
    """Initialize the database, creating tables if they don't exist."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


def get_session():
    """Create and return a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

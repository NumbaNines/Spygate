"""Database configuration module."""

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from .models import Base

# Load environment variables from .env.database if it exists
env_path = Path(".env.database")
if env_path.exists():
    load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    "postgresql": {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "spygate"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
    },
    "sqlite": {
        "database": os.getenv("SQLITE_PATH", "data/spygate.db"),
    },
}


def get_database_url(db_type: str = None) -> str:
    """Get database URL based on type."""
    if db_type is None:
        db_type = os.getenv("DB_TYPE", "postgresql")

    if db_type == "postgresql":
        config = DB_CONFIG["postgresql"]
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    elif db_type == "sqlite":
        return f"sqlite:///{DB_CONFIG['sqlite']['database']}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_db_config(db_type: str = None) -> Dict[str, Optional[str]]:
    """Get database configuration based on type."""
    if db_type is None:
        db_type = os.getenv("DB_TYPE", "postgresql")
    return DB_CONFIG.get(db_type, {})


def create_database_session(db_type: str = None):
    """Initialize the database connection and create session factory."""
    # Create database engine with connection pooling
    engine = create_engine(
        get_database_url(db_type),
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 minutes
        echo=False,  # Set to True for SQL query logging
    )

    # Create session factory
    Session = sessionmaker(bind=engine)
    return Session, engine


# Create global session factory
DatabaseSession, engine = create_database_session()

"""Tests for database configuration module."""

import os
import shutil
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from spygate.database.config import Base, DatabaseSession, engine, get_database_url, get_db, init_db


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Save original environment
    original_env = {
        "DB_TYPE": os.environ.get("DB_TYPE"),
        "SQLITE_PATH": os.environ.get("SQLITE_PATH"),
        "DB_HOST": os.environ.get("DB_HOST"),
        "DB_PORT": os.environ.get("DB_PORT"),
        "DB_NAME": os.environ.get("DB_NAME"),
        "DB_USER": os.environ.get("DB_USER"),
        "DB_PASSWORD": os.environ.get("DB_PASSWORD"),
    }

    # Set test values
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["SQLITE_PATH"] = ":memory:"

    yield

    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture
def test_db():
    """Create a test database."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    yield

    # Drop tables after tests
    Base.metadata.drop_all(bind=engine)


def test_get_database_url():
    """Test database URL generation."""
    # Test SQLite memory database
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["SQLITE_PATH"] = ":memory:"
    assert get_database_url() == "sqlite:///:memory:"

    # Test SQLite file database
    os.environ["SQLITE_PATH"] = "data/spygate.db"
    assert get_database_url() == "sqlite:///data/spygate.db"

    # Test PostgreSQL database
    os.environ["DB_TYPE"] = "postgresql"
    os.environ["DB_HOST"] = "testhost"
    os.environ["DB_PORT"] = "5433"
    os.environ["DB_NAME"] = "testdb"
    os.environ["DB_USER"] = "testuser"
    os.environ["DB_PASSWORD"] = "testpass"
    assert get_database_url() == "postgresql://testuser:testpass@testhost:5433/testdb"

    # Test unsupported database type
    os.environ["DB_TYPE"] = "mysql"
    with pytest.raises(ValueError, match="Unsupported database type: mysql"):
        get_database_url()


def test_database_session(test_db):
    """Test database session creation and usage."""
    # Create a session
    session = DatabaseSession()
    assert isinstance(session, Session)

    # Test connection with text()
    result = session.execute(text("SELECT 1"))
    assert result.scalar() == 1

    # Clean up
    session.close()


def test_get_db(test_db):
    """Test get_db context manager."""
    # Test successful transaction
    with get_db() as session:
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1

    # Test transaction rollback on error
    with pytest.raises(ValueError):
        with get_db() as session:
            session.execute(text("SELECT 1"))
            raise ValueError("Test error")


def test_init_db():
    """Test database initialization."""
    # Test with SQLite memory database
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["SQLITE_PATH"] = ":memory:"
    init_db()

    # Test with file database
    test_db_path = "data/test.db"
    os.environ["SQLITE_PATH"] = test_db_path
    init_db()

    # Verify data directory was created
    assert Path("data").exists()
    assert Path(test_db_path).exists()

    # Clean up
    Path(test_db_path).unlink()
    shutil.rmtree("data", ignore_errors=True)  # Remove directory and contents

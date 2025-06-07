"""Pytest configuration and shared fixtures."""

import os
import shutil

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..database import Base, get_db
from ..database.models import AnalysisJob, Clip, Tag, User


@pytest.fixture(scope="session")
def test_db():
    """Create a test database."""
    # Use an in-memory SQLite database for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables
    Base.metadata.create_all(engine)

    # Create a new session factory
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine, class_=Session
    )

    # Create a session
    session = TestingSessionLocal()

    # Override the get_db dependency
    def override_get_db():
        try:
            yield session
        finally:
            session.rollback()

    # Replace the original get_db with our test version
    get_db.__wrapped__ = override_get_db

    return session


@pytest.fixture(scope="session")
def test_user(test_db):
    """Create a test user."""
    user = User(id="test-user-id", email="test@example.com", is_active=True)
    test_db.add(user)
    test_db.commit()
    return user


@pytest.fixture(scope="function")
def mock_db_session():
    """Create a mock database session."""

    class MockContextManager:
        def __init__(self, session):
            self.session = session

        def __enter__(self):
            return self.session

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockDB:
        def __init__(self):
            self.added = []
            self.committed = False
            self.rolled_back = False

        def add(self, obj):
            self.added.append(obj)

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def query(self, *args):
            return self

        def filter(self, *args):
            return self

        def first(self):
            return None

    return MockContextManager(MockDB())


@pytest.fixture(scope="function")
def test_directories():
    """Create test directories."""
    dirs = ["uploads/videos", "uploads/clips"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    yield dirs

    # Cleanup
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

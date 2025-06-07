"""Test configuration for database tests."""

import pytest
from sqlalchemy.orm import Session

from spygate.database.config import Base, DatabaseSession, engine


@pytest.fixture
def db_session() -> Session:
    """Create a test database session.

    Yields:
        Session: SQLAlchemy session for testing
    """
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    session = DatabaseSession()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(bind=engine)

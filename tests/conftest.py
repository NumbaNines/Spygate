"""Test configuration and fixtures."""

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import scoped_session, sessionmaker

from src.database.base import Base
from src.database.models import Video, VideoMetadata  # Import the models explicitly
from src.database.session import Session


@pytest.fixture(scope="session")
def test_db():
    """Create a test database."""
    # Use SQLite for testing
    db_path = Path("tests/test.db")
    if db_path.exists():
        try:
            os.remove(db_path)
        except PermissionError:
            pass

    # Create test engine with echo for debugging
    engine = create_engine(f"sqlite:///{db_path}", echo=True)

    # Create all tables
    Base.metadata.drop_all(engine)  # Clean slate
    Base.metadata.create_all(engine)

    # Verify tables were created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "videos" in tables, f"Videos table not created. Available tables: {tables}"

    # Configure the session factory
    session_factory = sessionmaker(bind=engine)
    Session.configure(bind=engine)

    yield engine

    # Cleanup
    Session.remove()
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def test_session(test_db):
    """Create a new session for a test."""
    # Start with a clean session
    Session.remove()

    # Create new session
    session = Session()

    yield session

    # Cleanup
    session.rollback()
    session.close()
    Session.remove()


@pytest.fixture
def test_app(qtbot):
    """Create a test QApplication instance."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="session")
def app(qapp):
    """Create a Qt Application that persists for the entire test session."""
    return qapp


@pytest.fixture
def theme():
    """Provide a default theme configuration for testing."""
    return {
        # General
        "primary": "#007bff",
        "secondary": "#6c757d",
        "success": "#28a745",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        # Text
        "text_primary": "#212529",
        "text_secondary": "#6c757d",
        # Background
        "bg_primary": "#ffffff",
        "bg_secondary": "#f8f9fa",
        # Components
        "card_bg": "#ffffff",
        "card_border": "#dee2e6",
        "dialog_bg": "#ffffff",
        "dialog_border": "#dee2e6",
        "form_bg": "#ffffff",
        "form_border": "#ced4da",
        "nav_bg": "#ffffff",
        "nav_border": "#dee2e6",
        # States
        "hover": "#e9ecef",
        "active": "#007bff",
        "disabled": "#6c757d",
        # Elevation shadows
        "elevation_1": "0 2px 4px rgba(0,0,0,0.1)",
        "elevation_2": "0 4px 8px rgba(0,0,0,0.1)",
        "elevation_3": "0 8px 16px rgba(0,0,0,0.1)",
        "elevation_4": "0 16px 32px rgba(0,0,0,0.1)",
        "elevation_5": "0 32px 64px rgba(0,0,0,0.1)",
    }

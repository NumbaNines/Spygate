"""Pytest configuration and shared fixtures."""

import os
import shutil
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.config import Config
from ..core.hardware import HardwareDetector, HardwareTier
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


@pytest.fixture
def mock_hardware():
    """Create a mock hardware detector."""
    hardware = MagicMock(spec=HardwareDetector)

    # Mock GPU detection
    hardware.has_cuda = True
    hardware.gpu_count = 1
    hardware.get_gpu_memory_usage.return_value = 512.0
    hardware.get_system_memory.return_value = {
        "total": 16384.0,
        "available": 8192.0,
        "used": 8192.0,
        "percent": 50.0,
    }
    hardware.get_cpu_usage.return_value = 50.0

    # Mock hardware tier
    hardware.tier = HardwareTier.HIGH

    return hardware


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=Config)

    # Mock performance settings
    config.performance = {
        "target_fps": 30.0,
        "min_fps": 20.0,
        "max_memory_mb": 2048.0,
        "max_gpu_memory_mb": 1024.0,
        "memory_warning_threshold": 0.9,
        "gpu_warning_threshold": 0.9,
        "max_processing_time": 0.05,
        "max_batch_time": 0.2,
        "min_quality": 0.5,
        "max_quality": 1.0,
        "quality_step": 0.1,
        "fps_buffer_size": 100,
        "metrics_interval": 1.0,
        "cleanup_interval": 100,
    }

    # Mock tracking settings
    config.tracking = {
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
        "max_prediction_age": 5,
        "max_tracks": 100,
        "track_buffer_size": 30,
    }

    # Mock preprocessing settings
    config.preprocessing = {
        "target_size": (640, 640),
        "normalize": True,
        "batch_size": 32,
        "num_workers": 4,
    }

    return config

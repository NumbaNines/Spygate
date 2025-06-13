"""Pytest configuration and shared fixtures."""

import os
import shutil
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..core.config import Config
from ..core.hardware import HardwareDetector, HardwareTier
from ..database import Base, get_db
from ..database.models import AnalysisJob, Clip, Tag, User
from ..ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from ..ml.enhanced_ocr import EnhancedOCR
from ..ml.yolov8_model import EnhancedYOLOv8, OptimizationConfig


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


@pytest.fixture
def mock_hardware_detector():
    """Mock hardware detector returning HIGH tier."""
    detector = MagicMock(spec=HardwareDetector)
    detector.tier = HardwareTier.HIGH
    detector.get_gpu_info.return_value = {"name": "RTX 3080", "memory": 10240}
    detector.get_cpu_info.return_value = {"cores": 8, "threads": 16}
    detector.get_ram_info.return_value = {"total": 32768, "available": 16384}
    return detector


@pytest.fixture
def mock_yolo_model():
    """Mock YOLOv8 model for testing."""
    model = MagicMock(spec=EnhancedYOLOv8)
    model.detect_hud_elements.return_value = [
        {
            "class_name": "hud",
            "bbox": [0, 0, 1920, 100],
            "confidence": 0.95
        }
    ]
    return model


@pytest.fixture
def mock_ocr():
    """Mock OCR engine for testing."""
    ocr = MagicMock(spec=EnhancedOCR)
    ocr.read_text.return_value = ("Sample Text", 0.9)
    return ocr


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def game_analyzer(mock_hardware_detector, mock_yolo_model, mock_ocr):
    """Create a game analyzer instance with mocked components."""
    analyzer = EnhancedGameAnalyzer(
        hardware=mock_hardware_detector,
        optimization_config=OptimizationConfig(
            enable_dynamic_switching=True,
            enable_adaptive_batch_size=True,
            enable_performance_monitoring=True
        )
    )
    analyzer.model = mock_yolo_model
    analyzer.ocr = mock_ocr
    return analyzer


@pytest.fixture
def test_video_path():
    """Path to test video file."""
    return Path(__file__).parent / "data" / "test_clip.mp4"


@pytest.fixture
def mock_cv2():
    """Mock cv2 functions."""
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.read.return_value = (True, np.zeros((1080, 1920, 3)))
        mock_cap.return_value.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(x, 0)
        yield mock_cap


@pytest.fixture
def test_frame():
    """Create a test frame with HUD elements."""
    # Create base frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add HUD bar
    cv2.rectangle(frame, (100, 100), (500, 150), (128, 128, 128), -1)
    
    # Add possession triangle (left)
    pts = np.array([[50, 120], [80, 120], [65, 140]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (255, 0, 0))
    
    # Add territory triangle (right)
    pts = np.array([[520, 120], [550, 120], [535, 140]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    
    # Add text elements
    cv2.putText(frame, "1ST & 10", (120, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "OWN 25", (220, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


@pytest.fixture
def test_frame_sequence():
    """Create a sequence of test frames with changing game state."""
    frames = []
    for i in range(5):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add HUD bar
        cv2.rectangle(frame, (100, 100), (500, 150), (128, 128, 128), -1)
        
        # Add possession triangle (left)
        pts = np.array([[50, 120], [80, 120], [65, 140]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], (255, 0, 0))
        
        # Add territory triangle (right)
        pts = np.array([[520, 120], [550, 120], [535, 140]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        # Add text elements with changing down
        cv2.putText(frame, f"{i+1}ST & 10", (120, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "OWN 25", (220, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--show-plot",
        action="store_true",
        default=False,
        help="Show visualization plots during tests"
    )

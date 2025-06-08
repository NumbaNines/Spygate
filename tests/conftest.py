"""Test fixtures and configuration."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from collections.abc import Generator

import cv2
import numpy as np
import pytest

from spygate.utils.tracking_hardware import TrackingMode
from spygate.video.object_tracker import MultiObjectTracker, ObjectTracker
from tests.data_generator import TestDataGenerator
from tests.utils import BoundingBox, Color, Frame, Position

# Constants
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """Create and return a temporary test directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def data_generator(test_data_dir: Path) -> TestDataGenerator:
    """Return a test data generator instance."""
    return TestDataGenerator(test_data_dir)


@pytest.fixture
def mock_frame() -> Frame:
    """Create a mock video frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_frames() -> list[Frame]:
    """Create a sequence of mock video frames."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture
def mock_bounding_box() -> BoundingBox:
    """Create a mock bounding box."""
    return (100, 100, 200, 200)  # x1, y1, x2, y2


@pytest.fixture
def mock_bounding_boxes() -> list[BoundingBox]:
    """Create a sequence of mock bounding boxes."""
    return [(100 + i * 10, 100 + i * 10, 200 + i * 10, 200 + i * 10) for i in range(10)]


@pytest.fixture
def mock_position() -> Position:
    """Create a mock position."""
    return (150, 150)  # x, y


@pytest.fixture
def mock_positions() -> list[Position]:
    """Create a sequence of mock positions."""
    return [(150 + i * 10, 150 + i * 10) for i in range(10)]


@pytest.fixture
def mock_velocity() -> tuple[float, float]:
    """Create a mock velocity vector."""
    return (5.0, 5.0)  # vx, vy


@pytest.fixture
def mock_velocities() -> list[tuple[float, float]]:
    """Create a sequence of mock velocity vectors."""
    return [(5.0 + i * 0.5, 5.0 + i * 0.5) for i in range(10)]


@pytest.fixture
def mock_color() -> Color:
    """Create a mock color."""
    return (255, 0, 0)  # BGR format


@pytest.fixture
def mock_colors() -> list[Color]:
    """Create a sequence of mock colors."""
    return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


@pytest.fixture
def mock_object_tracker() -> ObjectTracker:
    """Create a mock object tracker instance."""
    return ObjectTracker()


@pytest.fixture
def mock_multi_object_tracker() -> MultiObjectTracker:
    """Create a mock multi-object tracker instance."""
    return MultiObjectTracker()


@pytest.fixture
def mock_tracking_config() -> dict[str, any]:
    """Create a mock tracking configuration."""
    return {
        "mode": TrackingMode.GPU,
        "max_objects": 10,
        "min_confidence": 0.5,
        "iou_threshold": 0.3,
        "max_age": 5,
        "min_hits": 3,
    }


@pytest.fixture
def mock_formation_config() -> dict[str, any]:
    """Create a mock formation tracking configuration."""
    return {
        "formation_type": "4-4-2",
        "num_players": 11,
        "field_dimensions": (100, 50),  # length, width in meters
        "position_tolerance": 2.0,  # meters
        "formation_tolerance": 5.0,  # meters
    }


@pytest.fixture
def mock_performance_metrics() -> dict[str, list[float]]:
    """Create mock performance metrics data."""
    return {
        "processing_times": [0.033 + i * 0.001 for i in range(10)],  # seconds
        "memory_usage": [100.0 + i * 10 for i in range(10)],  # MB
        "gpu_usage": [50.0 + i * 2 for i in range(10)],  # percentage
    }


@pytest.fixture
def mock_occlusion_data() -> dict[str, any]:
    """Create mock occlusion test data."""
    return {
        "frames": [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(20)],
        "bounding_boxes": [(100 + i * 10, 100, 200 + i * 10, 200) for i in range(20)],
        "occlusion_frames": [5, 6, 7, 8, 9],  # frames with occlusion
        "occlusion_objects": [(300, 50, 400, 250)],  # occluding objects
    }


@pytest.fixture
def mock_formation_data() -> dict[str, any]:
    """Create mock formation test data."""
    return {
        "formation_type": "4-4-2",
        "player_positions": [
            [(x * 10, y * 10) for x, y in zip(range(11), range(11))] for _ in range(10)  # 10 frames
        ],
        "player_velocities": [[(2.0, 2.0) for _ in range(11)] for _ in range(10)],  # 10 frames
    }


@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for each test."""
    # Setup
    cv2.setNumThreads(1)  # Ensure deterministic behavior
    np.random.seed(42)  # Set random seed for reproducibility

    yield  # Run test

    # Teardown
    cv2.destroyAllWindows()


@pytest.fixture
def benchmark_data(test_data_dir: Path) -> dict[str, any]:
    """Load or create benchmark comparison data."""
    benchmark_file = test_data_dir / "benchmark_data.json"

    if benchmark_file.exists():
        with open(benchmark_file) as f:
            return json.load(f)

    # Create default benchmark data
    data = {
        "baseline": {"fps": 30.0, "memory_usage": 100.0, "gpu_usage": 50.0},
        "thresholds": {"fps_min": 25.0, "memory_max": 150.0, "gpu_max": 75.0},
    }

    with open(benchmark_file, "w") as f:
        json.dump(data, f, indent=2)

    return data


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False

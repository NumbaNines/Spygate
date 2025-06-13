import pytest
import numpy as np
import cv2
from pathlib import Path
from spygate.ml.triangle_orientation_detector import (
    TriangleOrientationDetector,
    TriangleType,
    Direction,
    TriangleOrientation
)

@pytest.fixture
def detector():
    """Create a detector instance for testing."""
    return TriangleOrientationDetector()

@pytest.fixture
def debug_detector(tmp_path):
    """Create a detector instance with debug output enabled."""
    debug_dir = tmp_path / "debug_output"
    return TriangleOrientationDetector(debug_output_dir=debug_dir)

def create_arrow_contour(direction: str) -> np.ndarray:
    """Create a synthetic arrow contour for testing."""
    if direction == "right":
        points = np.array([
            [[0, 10]], [[30, 10]], [[20, 0]],
            [[30, 10]], [[20, 20]], [[0, 10]]
        ], dtype=np.int32)
    else:  # left
        points = np.array([
            [[30, 10]], [[0, 10]], [[10, 0]],
            [[0, 10]], [[10, 20]], [[30, 10]]
        ], dtype=np.int32)
    return points

def create_triangle_contour(direction: str) -> np.ndarray:
    """Create a synthetic triangle contour for testing."""
    if direction == "up":
        points = np.array([
            [[15, 0]], [[30, 30]], [[0, 30]]
        ], dtype=np.int32)
    else:  # down
        points = np.array([
            [[15, 30]], [[30, 0]], [[0, 0]]
        ], dtype=np.int32)
    return points

class TestTriangleOrientationDetector:
    """Test suite for TriangleOrientationDetector."""
    
    def test_possession_arrow_right(self, detector):
        """Test right-pointing possession arrow detection."""
        contour = create_arrow_contour("right")
        result = detector.analyze_possession_triangle(contour)
        
        assert result.is_valid
        assert result.direction == Direction.RIGHT
        assert result.confidence > detector.MIN_CONFIDENCE
        assert result.triangle_type == TriangleType.POSSESSION
    
    def test_possession_arrow_left(self, detector):
        """Test left-pointing possession arrow detection."""
        contour = create_arrow_contour("left")
        result = detector.analyze_possession_triangle(contour)
        
        assert result.is_valid
        assert result.direction == Direction.LEFT
        assert result.confidence > detector.MIN_CONFIDENCE
        assert result.triangle_type == TriangleType.POSSESSION
    
    def test_territory_triangle_up(self, detector):
        """Test upward-pointing territory triangle detection."""
        contour = create_triangle_contour("up")
        result = detector.analyze_territory_triangle(contour)
        
        assert result.is_valid
        assert result.direction == Direction.UP
        assert result.confidence > detector.MIN_CONFIDENCE
        assert result.triangle_type == TriangleType.TERRITORY
    
    def test_territory_triangle_down(self, detector):
        """Test downward-pointing territory triangle detection."""
        contour = create_triangle_contour("down")
        result = detector.analyze_territory_triangle(contour)
        
        assert result.is_valid
        assert result.direction == Direction.DOWN
        assert result.confidence > detector.MIN_CONFIDENCE
        assert result.triangle_type == TriangleType.TERRITORY
    
    def test_invalid_contour(self, detector):
        """Test handling of invalid contours."""
        invalid_contour = np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)  # Too few points
        
        possession_result = detector.analyze_possession_triangle(invalid_contour)
        assert not possession_result.is_valid
        assert possession_result.direction == Direction.UNKNOWN
        assert possession_result.confidence == 0.0
        
        territory_result = detector.analyze_territory_triangle(invalid_contour)
        assert not territory_result.is_valid
        assert territory_result.direction == Direction.UNKNOWN
        assert territory_result.confidence == 0.0
    
    def test_debug_visualization(self, debug_detector, tmp_path):
        """Test debug visualization output."""
        debug_dir = tmp_path / "debug_output"
        assert debug_dir.exists()
        
        # Create a test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test possession arrow visualization
        contour = create_arrow_contour("right")
        debug_detector.analyze_possession_triangle(contour, frame)
        
        # Test territory triangle visualization
        contour = create_triangle_contour("up")
        debug_detector.analyze_territory_triangle(contour, frame)
        
        # Check that visualization files were created
        viz_files = list(debug_dir.glob("*.png"))
        assert len(viz_files) == 2
        
        possession_files = list(debug_dir.glob("possession_*.png"))
        territory_files = list(debug_dir.glob("territory_*.png"))
        assert len(possession_files) == 1
        assert len(territory_files) == 1
    
    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
    def test_invalid_confidence_validation(self, confidence):
        """Test validation of confidence values."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            TriangleOrientation(
                is_valid=True,
                direction=Direction.UP,
                confidence=confidence,
                center=(0, 0),
                points=np.array([]),
                triangle_type=TriangleType.TERRITORY
            )
    
    def test_direction_enum_conversion(self):
        """Test automatic conversion of direction strings to enum."""
        orientation = TriangleOrientation(
            is_valid=True,
            direction="up",  # String instead of enum
            confidence=0.8,
            center=(0, 0),
            points=np.array([]),
            triangle_type=TriangleType.TERRITORY
        )
        assert isinstance(orientation.direction, Direction)
        assert orientation.direction == Direction.UP
    
    def test_invalid_triangle_type(self):
        """Test validation of triangle type."""
        with pytest.raises(ValueError, match="Invalid triangle type"):
            TriangleOrientation(
                is_valid=True,
                direction=Direction.UP,
                confidence=0.8,
                center=(0, 0),
                points=np.array([]),
                triangle_type="invalid"  # Invalid type
            ) 
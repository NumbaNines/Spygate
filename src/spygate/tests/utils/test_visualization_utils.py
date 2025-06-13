import cv2
import numpy as np
import pytest

from spygate.utils.visualization_utils import draw_bounding_box, draw_formation, draw_trajectory


@pytest.fixture
def test_frame():
    """Create a test frame for visualization."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_draw_bounding_box(test_frame):
    """Test bounding box drawing functionality."""
    bbox = (100, 100, 50, 50)  # x, y, w, h
    object_id = 1
    color = (0, 255, 0)  # Green

    # Draw bounding box
    draw_bounding_box(test_frame, bbox, object_id, color)

    # Check that pixels were modified
    assert np.any(test_frame[100:150, 100:150] != 0)  # Box area should have non-zero pixels
    assert np.any(test_frame[90:100, 100:150] != 0)  # ID text area should have non-zero pixels


def test_draw_trajectory(test_frame):
    """Test trajectory drawing functionality."""
    trajectory = [(100, 100), (150, 150), (200, 200)]
    color = (255, 0, 0)  # Red

    # Draw trajectory
    draw_trajectory(test_frame, trajectory, color)

    # Check that pixels were modified along the trajectory
    for i in range(len(trajectory) - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        # Check points along the line
        assert np.any(test_frame[y1:y2, x1:x2] != 0)


def test_draw_formation(test_frame):
    """Test formation drawing functionality."""
    formation = [(100, 100), (200, 100), (300, 100)]  # Line formation
    color = (0, 0, 255)  # Blue

    # Draw formation
    draw_formation(test_frame, formation, color)

    # Check that pixels were modified at formation points
    for x, y in formation:
        assert np.any(test_frame[y - 5 : y + 5, x - 5 : x + 5] != 0)  # Formation point markers

    # Check that lines were drawn between points
    for i in range(len(formation) - 1):
        x1, y1 = formation[i]
        x2, y2 = formation[i + 1]
        assert np.any(test_frame[y1:y2, x1:x2] != 0)


def test_draw_bounding_box_edge_cases(test_frame):
    """Test bounding box drawing with edge cases."""
    # Test box at frame edge
    edge_bbox = (0, 0, 50, 50)
    draw_bounding_box(test_frame, edge_bbox, 1, (255, 0, 0))
    assert np.any(test_frame[0:50, 0:50] != 0)

    # Test box near frame boundary
    boundary_bbox = (590, 430, 50, 50)
    draw_bounding_box(test_frame, boundary_bbox, 2, (0, 255, 0))
    assert np.any(test_frame[430:480, 590:640] != 0)


def test_draw_trajectory_empty_points(test_frame):
    """Test trajectory drawing with empty points list."""
    empty_points = []
    original_frame = test_frame.copy()

    # Draw trajectory with empty points
    draw_trajectory(test_frame, empty_points, (255, 0, 0), 2)

    # Frame should remain unchanged
    assert np.array_equal(test_frame, original_frame)


def test_draw_formation_single_player(test_frame):
    """Test formation drawing with single player."""
    single_player = [(320, 240)]  # Center of frame

    # Draw formation with single player
    draw_formation(test_frame, single_player, "Single Back", (0, 0, 255))

    # Check that player position was marked
    assert np.any(test_frame[235:245, 315:325] != 0)

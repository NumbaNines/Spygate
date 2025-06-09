"""Utility functions and helpers for testing."""

from collections.abc import Generator
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spygate.utils.tracking_hardware import TrackingMode
from spygate.video.object_tracker import MultiObjectTracker, ObjectTracker

# Type aliases
Frame = np.ndarray
BoundingBox = tuple[int, int, int, int]
Position = tuple[int, int]
Color = tuple[int, int, int]


def create_motion_sequence(
    start_pos: Position,
    velocity: tuple[float, float],
    num_frames: int,
    frame_size: tuple[int, int] = (640, 480),
    object_size: tuple[int, int] = (50, 50),
    color: Color = (255, 255, 255),
) -> list[Frame]:
    """Create a sequence of frames with an object moving at constant velocity.

    Args:
        start_pos: Initial (x, y) position
        velocity: (vx, vy) velocity in pixels per frame
        num_frames: Number of frames to generate
        frame_size: Size of frames (width, height)
        object_size: Size of moving object (width, height)
        color: BGR color of the object

    Returns:
        List of frames with moving object
    """
    frames = []
    x, y = start_pos
    vx, vy = velocity

    for i in range(num_frames):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        x_pos = int(x + vx * i)
        y_pos = int(y + vy * i)
        cv2.rectangle(
            frame, (x_pos, y_pos), (x_pos + object_size[0], y_pos + object_size[1]), color, -1
        )
        frames.append(frame)

    return frames


def create_occlusion_sequence(
    obj1_start: Position,
    obj2_start: Position,
    collision_frame: int,
    num_frames: int,
    frame_size: tuple[int, int] = (640, 480),
    object_size: tuple[int, int] = (50, 50),
) -> list[Frame]:
    """Create a sequence where two objects collide and then separate.

    Args:
        obj1_start: Starting position of first object
        obj2_start: Starting position of second object
        collision_frame: Frame number where objects collide
        num_frames: Total number of frames
        frame_size: Size of frames (width, height)
        object_size: Size of objects (width, height)

    Returns:
        List of frames showing collision sequence
    """
    frames = []
    x1, y1 = obj1_start
    x2, y2 = obj2_start

    # Calculate velocities to meet at collision_frame
    collision_x = (x1 + x2) // 2
    collision_y = (y1 + y2) // 2
    vx1 = (collision_x - x1) / collision_frame
    vy1 = (collision_y - y1) / collision_frame
    vx2 = (collision_x - x2) / collision_frame
    vy2 = (collision_y - y2) / collision_frame

    for i in range(num_frames):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # Pre-collision
        if i < collision_frame:
            x1_pos = int(x1 + vx1 * i)
            y1_pos = int(y1 + vy1 * i)
            x2_pos = int(x2 + vx2 * i)
            y2_pos = int(y2 + vy2 * i)
        # Collision
        elif i == collision_frame:
            x1_pos = x2_pos = collision_x
            y1_pos = y2_pos = collision_y
        # Post-collision
        else:
            x1_pos = int(collision_x - vx1 * (i - collision_frame))
            y1_pos = int(collision_y - vy1 * (i - collision_frame))
            x2_pos = int(collision_x - vx2 * (i - collision_frame))
            y2_pos = int(collision_y - vy2 * (i - collision_frame))

        # Draw objects
        cv2.rectangle(
            frame,
            (x1_pos, y1_pos),
            (x1_pos + object_size[0], y1_pos + object_size[1]),
            (255, 0, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (x2_pos, y2_pos),
            (x2_pos + object_size[0], y2_pos + object_size[1]),
            (0, 255, 0),
            -1,
        )
        frames.append(frame)

    return frames


def create_formation_sequence(
    num_players: int = 22, field_size: tuple[int, int] = (1024, 768), formation_type: str = "4-4-2"
) -> list[Frame]:
    """Create a sequence showing players in formation.

    Args:
        num_players: Number of players (default: 22 for full teams)
        field_size: Size of the field (width, height)
        formation_type: Type of formation (e.g., "4-4-2", "4-3-3")

    Returns:
        List of frames showing formation
    """
    frames = []
    formation_map = {
        "4-4-2": [
            # Defense
            [(200, y) for y in range(200, 601, 100)],
            # Midfield
            [(400, y) for y in range(200, 601, 100)],
            # Attack
            [(600, 300), (600, 500)],
        ],
        "4-3-3": [
            # Defense
            [(200, y) for y in range(200, 601, 100)],
            # Midfield
            [(400, y) for y in range(250, 551, 100)],
            # Attack
            [(600, y) for y in range(200, 601, 150)],
        ],
    }

    if formation_type not in formation_map:
        raise ValueError(f"Unsupported formation type: {formation_type}")

    formation = formation_map[formation_type]
    frame = np.zeros((field_size[1], field_size[0], 3), dtype=np.uint8)

    # Draw players
    for line in formation:
        for pos in line:
            cv2.circle(frame, pos, 10, (255, 255, 255), -1)

    # Mirror for opposing team
    frame_mirror = cv2.flip(frame.copy(), 1)
    cv2.circle(frame, (field_size[0] // 2, field_size[1] // 2), 5, (0, 255, 0), -1)  # Ball

    frames.append(cv2.addWeighted(frame, 1, frame_mirror, 1, 0))
    return frames


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box (x, y, w, h)
        box2: Second bounding box (x, y, w, h)

    Returns:
        IoU score between 0 and 1
    """
    # Convert to x1, y1, x2, y2 format
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union


def verify_tracking_consistency(
    tracker: ObjectTracker,
    sequence: list[Frame],
    initial_bbox: BoundingBox,
    max_deviation: float = 15.0,
) -> tuple[bool, list[float]]:
    """Verify tracking consistency across a sequence.

    Args:
        tracker: Initialized tracker
        sequence: List of frames
        initial_bbox: Initial bounding box
        max_deviation: Maximum allowed position deviation

    Returns:
        Tuple of (success, deviations)
    """
    success = tracker.init(sequence[0], initial_bbox)
    if not success:
        return False, []

    deviations = []
    last_pos = np.array([initial_bbox[0], initial_bbox[1]])

    for frame in sequence[1:]:
        success, bbox = tracker.update(frame)
        if not success:
            return False, deviations

        current_pos = np.array([bbox[0], bbox[1]])
        deviation = np.linalg.norm(current_pos - last_pos)
        deviations.append(deviation)
        last_pos = current_pos

        if deviation > max_deviation:
            return False, deviations

    return True, deviations


def mock_hardware_environment(
    has_gpu: bool = True, available_memory: float = 8192.0, gpu_memory: float = 4096.0
) -> dict:
    """Create a mock hardware environment configuration.

    Args:
        has_gpu: Whether GPU is available
        available_memory: Available system memory in MB
        gpu_memory: Available GPU memory in MB

    Returns:
        Dictionary with mock configuration
    """
    return {
        "has_gpu": has_gpu,
        "available_memory": available_memory,
        "gpu_memory": gpu_memory,
        "cpu_cores": 8,
        "gpu_compute_capability": 7.5 if has_gpu else None,
        "cpu_architecture": "x86_64",
        "os_platform": "linux",
    }


def generate_performance_report(
    timings: list[float], memory_usage: list[float], gpu_usage: Optional[list[float]] = None
) -> dict:
    """Generate a performance report from tracking metrics.

    Args:
        timings: List of processing times
        memory_usage: List of memory usage values
        gpu_usage: Optional list of GPU usage values

    Returns:
        Dictionary with performance statistics
    """
    report = {
        "timing": {
            "mean": np.mean(timings),
            "std": np.std(timings),
            "min": np.min(timings),
            "max": np.max(timings),
            "fps": 1.0 / np.mean(timings),
        },
        "memory": {
            "mean": np.mean(memory_usage),
            "peak": np.max(memory_usage),
            "baseline": memory_usage[0],
        },
    }

    if gpu_usage:
        report["gpu"] = {
            "mean": np.mean(gpu_usage),
            "peak": np.max(gpu_usage),
            "utilization": np.mean(gpu_usage) / 100.0,
        }

    return report

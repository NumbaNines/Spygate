"""
Visualization utilities for the Spygate application.

This module provides common visualization functions and utilities
used across different visualization components.
"""

from typing import List, Tuple, Union

import cv2
import numpy as np


def draw_bounding_box(
    frame: np.ndarray,
    bbox: list[float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw a bounding box on a frame.

    Args:
        frame: Input frame
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        color: Box color in BGR format
        thickness: Line thickness
        label: Optional label to draw above box
        font_scale: Font scale for label

    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Calculate text size and background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness,
        )

    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    draw_direction: bool = True,
) -> np.ndarray:
    """Draw a trajectory line through a series of points.

    Args:
        frame: Input frame
        points: List of (x, y) coordinates
        color: Line color in BGR format
        thickness: Line thickness
        draw_direction: Whether to draw direction arrows

    Returns:
        Frame with trajectory drawn
    """
    if len(points) < 2:
        return frame

    # Convert points to integer array
    points_array = np.array(points, dtype=np.int32)

    # Draw trajectory line
    cv2.polylines(
        frame,
        [points_array],
        False,
        color,
        thickness,
    )

    if draw_direction and len(points) >= 2:
        # Draw direction arrow at the end
        end_point = points[-1]
        prev_point = points[-2]

        # Calculate arrow properties
        angle = np.arctan2(
            end_point[1] - prev_point[1],
            end_point[0] - prev_point[0],
        )
        arrow_length = 20
        arrow_angle = np.pi / 6  # 30 degrees

        # Calculate arrow points
        p1 = (
            int(end_point[0] - arrow_length * np.cos(angle + arrow_angle)),
            int(end_point[1] - arrow_length * np.sin(angle + arrow_angle)),
        )
        p2 = (
            int(end_point[0] - arrow_length * np.cos(angle - arrow_angle)),
            int(end_point[1] - arrow_length * np.sin(angle - arrow_angle)),
        )

        # Draw arrow head
        cv2.line(
            frame,
            (int(end_point[0]), int(end_point[1])),
            p1,
            color,
            thickness,
        )
        cv2.line(
            frame,
            (int(end_point[0]), int(end_point[1])),
            p2,
            color,
            thickness,
        )

    return frame


def create_heat_map(
    frame_shape: tuple[int, int],
    points: list[tuple[float, float]],
    sigma: float = 30.0,
    opacity: float = 0.7,
) -> np.ndarray:
    """Create a heat map from a list of points.

    Args:
        frame_shape: Shape of the frame (height, width)
        points: List of (x, y) coordinates
        sigma: Gaussian blur sigma
        opacity: Heat map opacity (0-1)

    Returns:
        Heat map as a numpy array
    """
    heat_map = np.zeros(frame_shape, dtype=np.float32)

    # Add points to heat map
    for x, y in points:
        if 0 <= int(y) < frame_shape[0] and 0 <= int(x) < frame_shape[1]:
            heat_map[int(y), int(x)] += 1

    # Apply Gaussian blur
    kernel_size = int(6 * sigma + 1)  # 3 sigma on each side
    if kernel_size % 2 == 0:
        kernel_size += 1
    heat_map = cv2.GaussianBlur(
        heat_map,
        (kernel_size, kernel_size),
        sigma,
    )

    # Normalize and colorize
    if np.max(heat_map) > 0:
        heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map = cv2.applyColorMap(heat_map.astype(np.uint8), cv2.COLORMAP_JET)

    # Add alpha channel
    heat_map = cv2.addWeighted(
        heat_map,
        opacity,
        np.zeros_like(heat_map),
        1 - opacity,
        0,
    )

    return heat_map


def draw_formation_lines(
    frame: np.ndarray,
    positions: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    connection_threshold: float = 100.0,
) -> np.ndarray:
    """Draw formation analysis visualization.

    Args:
        frame: Input frame
        positions: List of player positions as (x, y) coordinates
        color: Line color in BGR format
        thickness: Line thickness
        connection_threshold: Maximum distance to connect players

    Returns:
        Frame with formation visualization
    """
    if len(positions) < 3:
        return frame

    # Convert positions to numpy array
    pos_array = np.array(positions, dtype=np.int32)

    # Calculate and draw convex hull
    hull = cv2.convexHull(pos_array)
    cv2.polylines(frame, [hull], True, color, thickness)

    # Draw connections between nearby players
    for i, pos1 in enumerate(positions):
        pos1 = np.array(pos1)
        for pos2 in positions[i + 1 :]:
            pos2 = np.array(pos2)
            dist = np.linalg.norm(pos1 - pos2)
            if dist < connection_threshold:
                cv2.line(
                    frame,
                    tuple(pos1.astype(int)),
                    tuple(pos2.astype(int)),
                    color,
                    max(1, thickness - 1),
                )

    return frame


def draw_motion_vector(
    frame: np.ndarray,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    """Draw a motion vector arrow.

    Args:
        frame: Input frame
        start_point: Starting point (x, y)
        end_point: Ending point (x, y)
        color: Arrow color in BGR format
        thickness: Line thickness
        arrow_scale: Scale factor for arrow size

    Returns:
        Frame with motion vector drawn
    """
    # Convert points to integer coordinates
    start = (int(start_point[0]), int(start_point[1]))
    end = (int(end_point[0]), int(end_point[1]))

    # Calculate vector properties
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    if magnitude < 1:
        return frame

    # Scale arrow head size with magnitude and scale factor
    arrow_size = min(0.3 * magnitude * arrow_scale, 20)

    # Draw arrow
    cv2.arrowedLine(
        frame,
        start,
        end,
        color,
        thickness,
        tipLength=arrow_size / magnitude,
    )

    return frame


def blend_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """Blend an overlay image with a background image.

    Args:
        background: Background image
        overlay: Overlay image (same size as background)
        opacity: Opacity of overlay (0-1)

    Returns:
        Blended image
    """
    if background.shape != overlay.shape:
        raise ValueError("Background and overlay must have same shape")

    return cv2.addWeighted(
        background,
        1 - opacity,
        overlay,
        opacity,
        0,
    )

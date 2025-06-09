import colorsys
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


def draw_bounding_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    object_id: Union[int, str],
    color: tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.8,
) -> None:
    """
    Draw a bounding box with object ID on the frame.

    Args:
        frame: The frame to draw on
        bbox: Bounding box coordinates (x, y, w, h)
        object_id: ID to display with the box
        color: RGB color tuple for the box and text
        thickness: Line thickness
        font_scale: Font size scale
    """
    x, y, w, h = bbox

    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # Draw ID text
    text = str(object_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Background rectangle for text
    cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), color, -1)

    # Text
    cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
    max_points: int = 30,
    fade: bool = True,
) -> None:
    """
    Draw an object's movement trajectory on the frame.

    Args:
        frame: The frame to draw on
        trajectory: List of (x, y) positions
        color: RGB color tuple for the line
        thickness: Line thickness
        max_points: Maximum number of trajectory points to show
        fade: Whether to fade out older trajectory points
    """
    if len(trajectory) < 2:
        return

    # Use only the most recent points
    if len(trajectory) > max_points:
        trajectory = trajectory[-max_points:]

    # Convert to numpy array for drawing
    points = np.array(trajectory, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    if fade:
        # Draw trajectory with fading effect
        num_segments = len(trajectory) - 1
        for i in range(num_segments):
            alpha = (i + 1) / num_segments
            segment_color = tuple(int(c * alpha) for c in color)
            cv2.line(
                frame,
                tuple(trajectory[i]),
                tuple(trajectory[i + 1]),
                segment_color,
                thickness,
                cv2.LINE_AA,
            )
    else:
        # Draw trajectory line
        cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def draw_formation(
    frame: np.ndarray,
    positions: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
    node_radius: int = 5,
) -> None:
    """
    Draw team formation by connecting player positions.

    Args:
        frame: The frame to draw on
        positions: List of player (x, y) positions
        color: RGB color tuple for the formation lines
        thickness: Line thickness
        node_radius: Radius of position nodes
    """
    if len(positions) < 2:
        return

    # Convert positions to numpy array
    points = np.array(positions, dtype=np.int32)

    # Draw position nodes
    for pos in positions:
        cv2.circle(frame, pos, node_radius, color, -1)

    # Find convex hull of positions
    hull = cv2.convexHull(points)

    # Draw formation outline
    cv2.polylines(frame, [hull], True, color, thickness)

    # Draw connections between nearby players
    max_dist = 150  # Maximum distance to draw connection

    for i, pos1 in enumerate(positions):
        for pos2 in positions[i + 1 :]:
            dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
            if dist < max_dist:
                cv2.line(frame, pos1, pos2, color, thickness)


@lru_cache(maxsize=32)
def _create_gradient_kernel(
    radius: int,
    gradient_stops: tuple[tuple[float, tuple[int, int, int]], ...],
) -> np.ndarray:
    """
    Create a cached gradient kernel for heat map generation.

    Args:
        radius: Radius of the kernel
        gradient_stops: Tuple of (position, color) tuples for gradient

    Returns:
        Pre-computed gradient kernel
    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x * x + y * y <= radius * radius

    # Calculate distance from center for each point
    distances = np.sqrt(x * x + y * y)
    distances[~mask] = radius

    # Normalize distances to 0-1 range
    normalized_distances = distances / radius

    # Create gradient based on stops
    gradient = np.zeros((2 * radius + 1, 2 * radius + 1, 3), dtype=np.float32)

    for i in range(len(gradient_stops) - 1):
        pos1, color1 = gradient_stops[i]
        pos2, color2 = gradient_stops[i + 1]

        # Find points in this gradient segment
        mask_segment = (normalized_distances >= pos1) & (normalized_distances <= pos2)

        # Calculate interpolation factor
        factor = (normalized_distances[mask_segment] - pos1) / (pos2 - pos1)

        # Vectorized color interpolation
        for c in range(3):
            gradient[mask_segment, c] = color1[c] + factor * (color2[c] - color1[c])

    gradient[~mask] = 0
    return gradient


def draw_heat_map(
    frame: np.ndarray,
    positions: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int = 30,
    alpha: float = 0.3,
    blur_size: int = 15,
    use_gradient: bool = True,
    gradient_stops: Optional[list[tuple[float, tuple[int, int, int]]]] = None,
) -> None:
    """
    Draw a heat map of player positions with advanced gradient coloring.

    Args:
        frame: The frame to draw on
        positions: List of (x, y) positions
        color: Base RGB color tuple for the heat map
        radius: Radius of influence for each position
        alpha: Transparency of the heat map
        blur_size: Size of Gaussian blur kernel
        use_gradient: Whether to use gradient coloring
        gradient_stops: List of (position, color) tuples for custom gradient
    """
    if not positions:
        return

    # Input validation
    radius = max(1, min(radius, min(frame.shape[:2]) // 2))
    alpha = max(0.0, min(alpha, 1.0))
    blur_size = max(1, min(blur_size, min(frame.shape[:2]) // 4))
    if blur_size % 2 == 0:
        blur_size += 1  # Ensure odd kernel size for Gaussian blur

    # Create heat map layer
    heat_map = np.zeros_like(frame, dtype=np.float32)

    if use_gradient:
        # Default gradient stops if none provided
        if gradient_stops is None:
            gradient_stops = [
                (0.0, (color[0], color[1], color[2])),
                (
                    0.5,
                    (min(color[0] * 1.5, 255), min(color[1] * 1.5, 255), min(color[2] * 1.5, 255)),
                ),
                (1.0, (255, 255, 255)),
            ]

        # Convert gradient stops to tuple for caching
        gradient_stops_tuple = tuple((pos, tuple(color)) for pos, color in gradient_stops)
        gradient = _create_gradient_kernel(radius, gradient_stops_tuple)

        # Vectorized addition of gradients
        for pos in positions:
            y1, y2 = max(0, pos[1] - radius), min(frame.shape[0], pos[1] + radius + 1)
            x1, x2 = max(0, pos[0] - radius), min(frame.shape[1], pos[0] + radius + 1)

            gy1 = max(0, radius - pos[1])
            gy2 = min(2 * radius + 1, radius + frame.shape[0] - pos[1])
            gx1 = max(0, radius - pos[0])
            gx2 = min(2 * radius + 1, radius + frame.shape[1] - pos[0])

            try:
                heat_map[y1:y2, x1:x2] += gradient[gy1:gy2, gx1:gx2]
            except ValueError:
                continue  # Skip if position is too close to frame edge
    else:
        # Simple circular heat map using vectorized operations
        y, x = np.ogrid[: frame.shape[0], : frame.shape[1]]
        for pos in positions:
            dist_sq = (x - pos[0]) ** 2 + (y - pos[1]) ** 2
            mask = dist_sq <= radius**2
            heat = 1 - np.sqrt(dist_sq[mask]) / radius
            for c in range(3):
                heat_map[mask, c] += heat * color[c]

    # Normalize and apply blur
    if np.max(heat_map) > 0:
        heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
    if blur_size > 1:
        heat_map = cv2.GaussianBlur(heat_map, (blur_size, blur_size), 0)

    # Blend with original frame using CUDA if available
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            frame_gpu = cv2.cuda_GpuMat(frame)
            heat_map_gpu = cv2.cuda_GpuMat(heat_map.astype(np.uint8))
            result_gpu = cv2.cuda.addWeighted(frame_gpu, 1, heat_map_gpu, alpha, 0)
            result_gpu.download(frame)
        else:
            cv2.addWeighted(frame, 1, heat_map.astype(np.uint8), alpha, 0, frame)
    except Exception:
        # Fallback to CPU if CUDA fails
        cv2.addWeighted(frame, 1, heat_map.astype(np.uint8), alpha, 0, frame)


@lru_cache(maxsize=16)
def _create_arrow_head(arrow_size: float, angle: float) -> np.ndarray:
    """
    Create a cached arrow head shape for motion vectors.

    Args:
        arrow_size: Size of the arrow head
        angle: Angle of the arrow in radians

    Returns:
        Arrow head points as numpy array
    """
    # Create arrow head shape
    pts = np.array(
        [[-arrow_size, -arrow_size / 2], [0, 0], [-arrow_size, arrow_size / 2]], dtype=np.float32
    )

    # Rotate points
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return np.dot(pts, rotation_matrix.T)


def draw_motion_vectors(
    frame: np.ndarray,
    start_points: list[tuple[int, int]],
    end_points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
    arrow_size: float = 10.0,
    min_magnitude: float = 5.0,
    max_magnitude: Optional[float] = None,
    normalize_color: bool = True,
    smooth_arrows: bool = True,
) -> None:
    """
    Draw motion vectors with magnitude-based coloring and smooth arrows.

    Args:
        frame: The frame to draw on
        start_points: List of vector start points (x, y)
        end_points: List of vector end points (x, y)
        color: Base RGB color tuple for vectors
        thickness: Line thickness
        arrow_size: Size of arrow heads
        min_magnitude: Minimum vector magnitude to draw
        max_magnitude: Maximum vector magnitude for color normalization
        normalize_color: Whether to color vectors based on magnitude
        smooth_arrows: Whether to draw smooth arrow heads
    """
    if not start_points or not end_points or len(start_points) != len(end_points):
        return

    # Convert points to numpy arrays for vectorized operations
    starts = np.array(start_points, dtype=np.float32)
    ends = np.array(end_points, dtype=np.float32)

    # Calculate vector properties in a vectorized way
    vectors = ends - starts
    magnitudes = np.sqrt(np.sum(vectors**2, axis=1))

    # Filter by magnitude
    valid_mask = magnitudes >= min_magnitude
    if not np.any(valid_mask):
        return

    starts = starts[valid_mask]
    ends = ends[valid_mask]
    vectors = vectors[valid_mask]
    magnitudes = magnitudes[valid_mask]

    # Normalize magnitudes for coloring if requested
    if normalize_color:
        if max_magnitude is None:
            max_magnitude = np.max(magnitudes)
        normalized_magnitudes = np.clip(magnitudes / max_magnitude, 0, 1)

    # Draw vectors
    for i, (start, end, magnitude) in enumerate(zip(starts, ends, magnitudes)):
        # Calculate color intensity based on magnitude
        if normalize_color:
            intensity = normalized_magnitudes[i]
            col = tuple(int(c * intensity) for c in color)
        else:
            col = color

        # Draw line
        cv2.line(frame, tuple(map(int, start)), tuple(map(int, end)), col, thickness, cv2.LINE_AA)

        if smooth_arrows:
            # Calculate arrow angle
            angle = np.arctan2(end[1] - start[1], end[0] - start[0])

            # Get cached arrow head
            arrow_pts = _create_arrow_head(arrow_size, angle)

            # Transform arrow head to end point
            arrow_pts = arrow_pts + end

            # Draw arrow head
            cv2.fillPoly(frame, [arrow_pts.astype(np.int32)], col, cv2.LINE_AA)
        else:
            # Draw simple arrow head
            cv2.arrowedLine(
                frame,
                tuple(map(int, start)),
                tuple(map(int, end)),
                tuple(map(int, col)),
                thickness,
                tipLength=0.2,
            )


def draw_ball_prediction(
    frame: np.ndarray,
    current_pos: tuple[int, int],
    predicted_positions: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 1,
    confidence: Optional[float] = None,
    fade_effect: bool = True,
) -> None:
    """
    Draw predicted ball trajectory with confidence indicators and fading effect.

    Args:
        frame: The frame to draw on
        current_pos: Current ball position (x, y)
        predicted_positions: List of predicted (x, y) positions
        color: RGB color tuple for the prediction line
        thickness: Line thickness
        confidence: Prediction confidence score (0.0 to 1.0)
        fade_effect: Whether to apply fading effect to predictions
    """
    if not predicted_positions:
        return

    # Draw current position
    cv2.circle(frame, current_pos, 5, color, -1)

    # Draw predicted trajectory
    num_predictions = len(predicted_positions)
    for i in range(num_predictions - 1):
        # Calculate alpha for fading effect
        alpha = 1.0 - (i / num_predictions) if fade_effect else 1.0

        # Adjust color based on confidence and fading
        segment_color = list(color)
        if confidence is not None:
            # Scale color intensity by confidence
            segment_color = [int(c * confidence) for c in color]

        # Apply fading effect
        if fade_effect:
            segment_color = [int(c * alpha) for c in segment_color]

        # Draw line segment
        cv2.line(
            frame,
            predicted_positions[i],
            predicted_positions[i + 1],
            tuple(segment_color),
            thickness,
            cv2.LINE_AA,
        )

        # Draw confidence indicator at each prediction point
        if confidence is not None:
            radius = int(3 * confidence)  # Radius based on confidence
            cv2.circle(
                frame, predicted_positions[i], max(1, radius), tuple(segment_color), 1, cv2.LINE_AA
            )

    # Draw final prediction point
    if predicted_positions:
        final_color = [int(c * (confidence or 1.0) * (0.3 if fade_effect else 1.0)) for c in color]
        cv2.circle(frame, predicted_positions[-1], 3, tuple(final_color), -1, cv2.LINE_AA)


class VisualizationUtils:
    @staticmethod
    def visualize_ball_prediction(
        frame: np.ndarray,
        predictions: list[tuple[float, float]],
        confidences: list[float],
        fade_effect: bool = True,
        max_trail_length: int = 10,
        trail_thickness: int = 2,
        min_confidence: float = 0.3,
        uncertainty_radius: bool = True,
        smooth_curve: bool = True,
    ) -> np.ndarray:
        """
        Visualize ball predictions with confidence indicators and fading effects.

        Args:
            frame: Input frame to draw on
            predictions: List of (x, y) ball position predictions
            confidences: List of confidence scores for each prediction
            fade_effect: Whether to apply fading effect to older predictions
            max_trail_length: Maximum number of previous positions to show
            trail_thickness: Thickness of the prediction trail
            min_confidence: Minimum confidence to show prediction
            uncertainty_radius: Whether to show uncertainty radius based on confidence
            smooth_curve: Whether to use Bezier curve smoothing between predictions

        Returns:
            Frame with visualized ball predictions
        """
        result = frame.copy()
        if not predictions or not confidences:
            return result

        # Filter by confidence and limit trail length
        valid_indices = [i for i, conf in enumerate(confidences) if conf >= min_confidence]
        if not valid_indices:
            return result

        predictions = [predictions[i] for i in valid_indices]
        confidences = [confidences[i] for i in valid_indices]

        # Limit trail length
        predictions = predictions[-max_trail_length:]
        confidences = confidences[-max_trail_length:]

        # Draw connecting lines with fading effect
        if len(predictions) > 1:
            if smooth_curve and len(predictions) >= 3:
                # Generate smooth curve points using Bezier interpolation
                curve_points = []
                for i in range(len(predictions) - 2):
                    p0 = np.array(predictions[i])
                    p1 = np.array(predictions[i + 1])
                    p2 = np.array(predictions[i + 2])

                    # Generate control points
                    ctrl1 = p0 + (p1 - p0) * 0.5
                    ctrl2 = p1 + (p2 - p1) * 0.5

                    # Generate points along the curve
                    num_points = 10
                    for t in np.linspace(0, 1, num_points):
                        # Quadratic Bezier formula
                        point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
                        curve_points.append(point.astype(np.int32))

                # Draw smooth curve
                if curve_points:
                    points = np.array(curve_points)
                    for i in range(len(points) - 1):
                        # Calculate alpha based on position in curve
                        alpha = 1.0 if not fade_effect else (i + 1) / len(points)
                        avg_conf = np.mean(confidences[i : i + 2])

                        color = (int(255 * (1 - avg_conf)), int(255 * avg_conf), 0)  # B  # G  # R

                        # Apply alpha blending
                        overlay = result.copy()
                        cv2.line(
                            overlay,
                            tuple(points[i]),
                            tuple(points[i + 1]),
                            tuple(int(c * alpha) for c in color),
                            trail_thickness,
                            cv2.LINE_AA,
                        )
                        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            else:
                # Original line-based drawing
                for i in range(len(predictions) - 1):
                    start = tuple(map(int, predictions[i]))
                    end = tuple(map(int, predictions[i + 1]))

                    # Calculate alpha based on position in trail
                    alpha = 1.0 if not fade_effect else (i + 1) / len(predictions)
                    avg_conf = np.mean(confidences[i : i + 1])

                    color = (int(255 * (1 - avg_conf)), int(255 * avg_conf), 0)  # B  # G  # R

                    # Draw line with alpha blending
                    overlay = result.copy()
                    cv2.line(overlay, start, end, color, trail_thickness, cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw prediction points with uncertainty radius
        for i, ((x, y), conf) in enumerate(zip(predictions, confidences)):
            pos = (int(x), int(y))

            # Calculate alpha based on position in trail
            alpha = 1.0 if not fade_effect else (i + 1) / len(predictions)

            # Calculate radius based on confidence
            radius = int(max(3, min(10, conf * 15)))

            # Calculate color based on confidence (green to red)
            color = (int(255 * (1 - conf)), int(255 * conf), 0)  # B  # G  # R

            # Draw uncertainty radius if enabled
            if uncertainty_radius:
                uncertainty = int(
                    radius * (1 + (1 - conf) * 2)
                )  # Larger radius for lower confidence
                overlay = result.copy()
                cv2.circle(overlay, pos, uncertainty, color, 1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha * 0.5, result, 1 - alpha * 0.5, 0, result)

            # Draw prediction point
            overlay = result.copy()
            cv2.circle(overlay, pos, radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

            # Draw confidence value for most recent prediction
            if i == len(predictions) - 1:
                conf_text = f"{conf:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = pos[0] + 15
                text_y = pos[1] - 15

                # Draw text background
                bg_pts = np.array(
                    [
                        [text_x - 2, text_y - text_size[1] - 2],
                        [text_x + text_size[0] + 2, text_y - text_size[1] - 2],
                        [text_x + text_size[0] + 2, text_y + 2],
                        [text_x - 2, text_y + 2],
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(result, [bg_pts], (32, 32, 32))

                # Draw text
                cv2.putText(
                    result,
                    conf_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return result

    @staticmethod
    def create_heatmap(
        positions: list[tuple[float, float]],
        frame_shape: tuple[int, int],
        sigma: float = 30.0,
        gradient_stops: Optional[list[tuple[float, tuple[int, int, int]]]] = None,
    ) -> np.ndarray:
        """
        Create a heat map visualization with advanced gradient coloring and Gaussian blur.

        Args:
            positions: List of (x, y) positions
            frame_shape: Shape of the frame (height, width)
            sigma: Standard deviation for Gaussian blur
            gradient_stops: List of (position, color) tuples for custom gradient

        Returns:
            Heat map visualization as numpy array
        """
        if not positions:
            return np.zeros((*frame_shape, 3), dtype=np.uint8)

        # Create accumulation map
        heatmap = np.zeros(frame_shape, dtype=np.float32)
        for x, y in positions:
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                heatmap[int(y), int(x)] += 1

        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)

        # Normalize heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

        # Default gradient if none provided
        if not gradient_stops:
            gradient_stops = [
                (0.0, (0, 0, 255)),  # Blue
                (0.5, (0, 255, 0)),  # Green
                (1.0, (255, 0, 0)),  # Red
            ]

        # Create colored heatmap
        colored = np.zeros((*frame_shape, 3), dtype=np.uint8)
        for y in range(frame_shape[0]):
            for x in range(frame_shape[1]):
                value = heatmap[y, x]

                # Find gradient colors to interpolate between
                for i in range(len(gradient_stops) - 1):
                    pos1, color1 = gradient_stops[i]
                    pos2, color2 = gradient_stops[i + 1]

                    if pos1 <= value <= pos2:
                        # Linear interpolation between colors
                        t = (value - pos1) / (pos2 - pos1)
                        color = tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
                        colored[y, x] = color
                        break

        return colored

    @staticmethod
    def visualize_motion_vectors(
        frame: np.ndarray,
        vectors: list[tuple[tuple[float, float], tuple[float, float]]],
        min_magnitude: float = 1.0,
        max_magnitude: Optional[float] = None,
    ) -> np.ndarray:
        """
        Visualize motion vectors with magnitude-based coloring and smooth arrows.

        Args:
            frame: Input frame to draw on
            vectors: List of ((start_x, start_y), (end_x, end_y)) motion vectors
            min_magnitude: Minimum vector magnitude to display
            max_magnitude: Maximum magnitude for color scaling (auto-calculated if None)

        Returns:
            Frame with visualized motion vectors
        """
        result = frame.copy()
        if not vectors:
            return result

        # Calculate vector magnitudes
        magnitudes = []
        for (start_x, start_y), (end_x, end_y) in vectors:
            dx = end_x - start_x
            dy = end_y - start_y
            magnitude = np.sqrt(dx * dx + dy * dy)
            magnitudes.append(magnitude)

        # Determine max magnitude for color scaling
        if max_magnitude is None:
            max_magnitude = max(magnitudes)

        for ((start_x, start_y), (end_x, end_y)), magnitude in zip(vectors, magnitudes):
            if magnitude < min_magnitude:
                continue

            # Calculate color based on magnitude (HSV color space)
            hue = (1.0 - min(magnitude / max_magnitude, 1.0)) * 0.66  # Blue to red
            color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0))

            # Draw arrow with anti-aliasing
            cv2.arrowedLine(
                result,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                color,
                thickness=2,
                line_type=cv2.LINE_AA,
                tipLength=0.2,
            )

            # Add magnitude text for significant vectors
            if magnitude > min_magnitude * 2:
                text_pos = (int((start_x + end_x) / 2), int((start_y + end_y) / 2))
                cv2.putText(
                    result,
                    f"{magnitude:.1f}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return result

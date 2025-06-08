"""
Real-time visualization manager for tracking data.

This module provides a centralized manager for visualizing tracking data,
including player positions, ball tracking, and formation analysis in real-time.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer

logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """Available visualization modes."""

    TRACKING_ONLY = auto()  # Show only tracking boxes/labels
    MOTION_VECTORS = auto()  # Show motion vectors
    HEAT_MAP = auto()  # Show player movement heat map
    FORMATION = auto()  # Show formation analysis
    FULL = auto()  # Show all visualizations


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    mode: VisualizationMode = VisualizationMode.FULL
    show_player_ids: bool = True
    show_confidence: bool = True
    show_ball_trajectory: bool = True
    show_formation_lines: bool = True
    show_motion_trails: bool = True
    trail_length: int = 30
    heat_map_opacity: float = 0.4
    vector_scale: float = 1.0
    line_thickness: int = 2
    font_scale: float = 0.5
    enable_gpu: bool = True


class VisualizationManager(QObject):
    """Manages real-time visualization of tracking data."""

    # Signals for UI updates
    frame_ready = pyqtSignal(QImage)
    stats_updated = pyqtSignal(dict)

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
    ):
        """Initialize visualization manager.

        Args:
            config: Visualization configuration
        """
        super().__init__()
        self.config = config or VisualizationConfig()

        # Initialize hardware optimization
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize GPU context if available and requested
        self.use_gpu = False
        if self.config.enable_gpu:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    cv2.cuda.setDevice(0)
                    self.use_gpu = True
                    logger.info("GPU acceleration enabled for visualization")
                else:
                    logger.info("No GPU available, using CPU for visualization")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU: {e}")

        # Initialize visualization buffers
        self.position_history = {}  # Player ID -> List of positions
        self.heat_map = None
        self.frame_count = 0
        self.last_update = time.time()

        # Performance monitoring
        self.processing_times = []

        logger.info(f"Initialized VisualizationManager with {self.hardware.tier.name} tier")

    def update_frame(
        self,
        frame: np.ndarray,
        tracking_data: dict[str, Union[list, dict]],
        frame_number: int,
    ) -> np.ndarray:
        """Update visualization with new frame and tracking data.

        Args:
            frame: Current video frame
            tracking_data: Dictionary containing tracking results
            frame_number: Current frame number

        Returns:
            Frame with visualizations applied
        """
        start_time = time.time()

        # Create working copy of frame
        if self.use_gpu:
            vis_frame = cv2.cuda.GpuMat(frame)
        else:
            vis_frame = frame.copy()

        # Update tracking history
        self._update_position_history(tracking_data)

        # Apply visualizations based on mode
        if self.config.mode in [VisualizationMode.TRACKING_ONLY, VisualizationMode.FULL]:
            vis_frame = self._draw_tracking_boxes(vis_frame, tracking_data)

        if self.config.mode in [VisualizationMode.MOTION_VECTORS, VisualizationMode.FULL]:
            vis_frame = self._draw_motion_vectors(vis_frame, tracking_data)

        if self.config.mode in [VisualizationMode.HEAT_MAP, VisualizationMode.FULL]:
            vis_frame = self._draw_heat_map(vis_frame)

        if self.config.mode in [VisualizationMode.FORMATION, VisualizationMode.FULL]:
            vis_frame = self._draw_formation_analysis(vis_frame, tracking_data)

        # Convert back from GPU if needed
        if self.use_gpu:
            vis_frame = vis_frame.download()

        # Update performance stats
        end_time = time.time()
        self.processing_times.append(end_time - start_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        # Emit stats update every second
        if end_time - self.last_update >= 1.0:
            self.stats_updated.emit(self.get_performance_stats())
            self.last_update = end_time

        # Convert to QImage and emit
        height, width = vis_frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(
            vis_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        self.frame_ready.emit(q_img)

        return vis_frame

    def _update_position_history(
        self,
        tracking_data: dict[str, Union[list, dict]],
    ) -> None:
        """Update position history buffers."""
        for player_id, data in tracking_data.get("players", {}).items():
            if "position" not in data:
                continue

            if player_id not in self.position_history:
                self.position_history[player_id] = []

            self.position_history[player_id].append(data["position"])

            # Maintain trail length
            if len(self.position_history[player_id]) > self.config.trail_length:
                self.position_history[player_id].pop(0)

    def _draw_tracking_boxes(
        self,
        frame: Union[np.ndarray, cv2.cuda.GpuMat],
        tracking_data: dict[str, Union[list, dict]],
    ) -> Union[np.ndarray, cv2.cuda.GpuMat]:
        """Draw bounding boxes and labels for tracked objects."""
        # Draw player boxes
        for player_id, data in tracking_data.get("players", {}).items():
            if "bbox" not in data:
                continue

            bbox = data["bbox"]
            confidence = data.get("confidence", 0.0)

            # Draw box
            if self.use_gpu:
                cv2.cuda.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    self.config.line_thickness,
                )
            else:
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    self.config.line_thickness,
                )

            # Add label if enabled
            if self.config.show_player_ids:
                label = f"Player {player_id}"
                if self.config.show_confidence:
                    label += f" ({confidence:.2f})"

                if self.use_gpu:
                    frame = frame.download()
                    cv2.putText(
                        frame,
                        label,
                        (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        (0, 255, 0),
                        self.config.line_thickness,
                    )
                    frame = cv2.cuda.GpuMat(frame)
                else:
                    cv2.putText(
                        frame,
                        label,
                        (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        (0, 255, 0),
                        self.config.line_thickness,
                    )

        # Draw ball if present
        if "ball" in tracking_data and "position" in tracking_data["ball"]:
            ball_pos = tracking_data["ball"]["position"]
            if self.use_gpu:
                cv2.cuda.circle(
                    frame,
                    (int(ball_pos[0]), int(ball_pos[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )
            else:
                cv2.circle(
                    frame,
                    (int(ball_pos[0]), int(ball_pos[1])),
                    5,
                    (0, 0, 255),
                    -1,
                )

        return frame

    def _draw_motion_vectors(
        self,
        frame: Union[np.ndarray, cv2.cuda.GpuMat],
        tracking_data: dict[str, Union[list, dict]],
    ) -> Union[np.ndarray, cv2.cuda.GpuMat]:
        """Draw motion vectors for tracked objects."""
        for player_id, history in self.position_history.items():
            if len(history) < 2:
                continue

            # Draw motion trail
            if self.config.show_motion_trails:
                points = np.array(history, dtype=np.int32)
                if self.use_gpu:
                    frame = frame.download()
                    cv2.polylines(
                        frame,
                        [points],
                        False,
                        (255, 0, 0),
                        self.config.line_thickness,
                    )
                    frame = cv2.cuda.GpuMat(frame)
                else:
                    cv2.polylines(
                        frame,
                        [points],
                        False,
                        (255, 0, 0),
                        self.config.line_thickness,
                    )

            # Draw motion vector
            if len(history) >= 2:
                start = history[-2]
                end = history[-1]
                if self.use_gpu:
                    frame = frame.download()
                    cv2.arrowedLine(
                        frame,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        (255, 0, 0),
                        self.config.line_thickness,
                    )
                    frame = cv2.cuda.GpuMat(frame)
                else:
                    cv2.arrowedLine(
                        frame,
                        (int(start[0]), int(start[1])),
                        (int(end[0]), int(end[1])),
                        (255, 0, 0),
                        self.config.line_thickness,
                    )

        return frame

    def _draw_heat_map(
        self,
        frame: Union[np.ndarray, cv2.cuda.GpuMat],
    ) -> Union[np.ndarray, cv2.cuda.GpuMat]:
        """Draw player movement heat map."""
        if not self.position_history:
            return frame

        # Initialize heat map if needed
        if self.heat_map is None:
            if self.use_gpu:
                shape = frame.size()[:2]
            else:
                shape = frame.shape[:2]
            self.heat_map = np.zeros(shape, dtype=np.float32)

        # Update heat map with current positions
        for history in self.position_history.values():
            if not history:
                continue

            pos = history[-1]
            y, x = int(pos[1]), int(pos[0])
            if 0 <= y < self.heat_map.shape[0] and 0 <= x < self.heat_map.shape[1]:
                self.heat_map[y, x] += 1

        # Apply Gaussian blur
        if self.use_gpu:
            heat_map_gpu = cv2.cuda.GpuMat(self.heat_map)
            heat_map_gpu = cv2.cuda.GaussianBlur(heat_map_gpu, (15, 15), 0)
            heat_map = heat_map_gpu.download()
        else:
            heat_map = cv2.GaussianBlur(self.heat_map, (15, 15), 0)

        # Normalize and colorize
        heat_map = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map = cv2.applyColorMap(
            heat_map.astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        # Blend with frame
        if self.use_gpu:
            frame = frame.download()
            frame = cv2.addWeighted(
                frame,
                1 - self.config.heat_map_opacity,
                heat_map,
                self.config.heat_map_opacity,
                0,
            )
            frame = cv2.cuda.GpuMat(frame)
        else:
            frame = cv2.addWeighted(
                frame,
                1 - self.config.heat_map_opacity,
                heat_map,
                self.config.heat_map_opacity,
                0,
            )

        return frame

    def _draw_formation_analysis(
        self,
        frame: Union[np.ndarray, cv2.cuda.GpuMat],
        tracking_data: dict[str, Union[list, dict]],
    ) -> Union[np.ndarray, cv2.cuda.GpuMat]:
        """Draw formation analysis visualization."""
        if not self.config.show_formation_lines:
            return frame

        # Extract player positions
        positions = []
        for data in tracking_data.get("players", {}).values():
            if "position" in data:
                positions.append(data["position"])

        if len(positions) < 3:  # Need at least 3 players for formation
            return frame

        # Convert to numpy array
        positions = np.array(positions)

        # Calculate convex hull
        hull = cv2.convexHull(positions.astype(np.int32))

        # Draw formation outline
        if self.use_gpu:
            frame = frame.download()
            cv2.polylines(
                frame,
                [hull],
                True,
                (0, 255, 255),
                self.config.line_thickness,
            )
            frame = cv2.cuda.GpuMat(frame)
        else:
            cv2.polylines(
                frame,
                [hull],
                True,
                (0, 255, 255),
                self.config.line_thickness,
            )

        # Draw lines between nearby players
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 100:  # Threshold for connecting players
                    if self.use_gpu:
                        frame = frame.download()
                        cv2.line(
                            frame,
                            tuple(pos1.astype(int)),
                            tuple(pos2.astype(int)),
                            (0, 255, 255),
                            1,
                        )
                        frame = cv2.cuda.GpuMat(frame)
                    else:
                        cv2.line(
                            frame,
                            tuple(pos1.astype(int)),
                            tuple(pos2.astype(int)),
                            (0, 255, 255),
                            1,
                        )

        return frame

    def get_performance_stats(self) -> dict[str, float]:
        """Get visualization performance statistics."""
        if not self.processing_times:
            return {
                "avg_processing_time": 0.0,
                "fps": 0.0,
                "gpu_enabled": self.use_gpu,
            }

        avg_time = np.mean(self.processing_times)
        return {
            "avg_processing_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "gpu_enabled": self.use_gpu,
        }

    def reset(self) -> None:
        """Reset visualization state."""
        self.position_history.clear()
        self.heat_map = None
        self.frame_count = 0
        self.processing_times.clear()

"""
Performance optimization and tier-based feature adaptation for SpygateAI.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .hardware import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)


@dataclass
class ProcessingParams:
    """Container for video processing parameters."""

    target_fps: float
    max_fps: float
    resolution_scale: float
    frame_skip: int
    batch_size: int
    use_gpu: bool
    yolo_model: str


class TierOptimizer:
    """Optimizes processing parameters based on hardware tier."""

    # YOLO model selection by tier
    YOLO_MODELS = {
        HardwareTier.MINIMUM: None,  # CV-only mode
        HardwareTier.STANDARD: "yolov11-nano",
        HardwareTier.PREMIUM: "yolov11-medium",
        HardwareTier.PROFESSIONAL: "yolov11-large",
    }

    # Batch size by tier
    BATCH_SIZES = {
        HardwareTier.MINIMUM: 1,
        HardwareTier.STANDARD: 2,
        HardwareTier.PREMIUM: 4,
        HardwareTier.PROFESSIONAL: 8,
    }

    def __init__(self, hardware_detector: HardwareDetector):
        """Initialize the optimizer with hardware detection."""
        self.hardware = hardware_detector
        self.current_params = self._initialize_params()
        self.performance_history = []
        logger.info(f"Initialized optimizer with params: {self.current_params}")

    def _initialize_params(self) -> ProcessingParams:
        """Initialize processing parameters based on hardware tier."""
        capabilities = self.hardware.get_tier_capabilities()
        features = self.hardware.get_tier_features()

        return ProcessingParams(
            target_fps=capabilities["target_fps"],
            max_fps=capabilities["max_fps"],
            resolution_scale=capabilities["resolution_scale"],
            frame_skip=capabilities["frame_skip"],
            batch_size=self.BATCH_SIZES[self.hardware.tier],
            use_gpu=features["yolo_detection"],
            yolo_model=self.YOLO_MODELS[self.hardware.tier],
        )

    def optimize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimize a frame based on current parameters."""
        metrics = {}

        # Apply resolution scaling
        if self.current_params.resolution_scale < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.current_params.resolution_scale)
            new_h = int(h * self.current_params.resolution_scale)
            frame = cv2.resize(frame, (new_w, new_h))
            metrics["resolution_scale"] = self.current_params.resolution_scale

        # Track processing metrics
        metrics.update(
            {
                "target_fps": self.current_params.target_fps,
                "frame_skip": self.current_params.frame_skip,
                "batch_size": self.current_params.batch_size,
            }
        )

        return frame, metrics

    def should_process_frame(self, frame_count: int) -> bool:
        """Determine if a frame should be processed based on frame skip."""
        return frame_count % self.current_params.frame_skip == 0

    def update_performance(self, processing_time: float) -> None:
        """Update performance history and adjust parameters if needed."""
        achieved_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.performance_history.append(achieved_fps)

        # Keep last 30 seconds of history
        if len(self.performance_history) > 30:
            self.performance_history.pop(0)

        # Calculate average FPS
        avg_fps = sum(self.performance_history) / len(self.performance_history)

        # Adjust parameters if performance is outside target range
        if avg_fps < self.current_params.target_fps * 0.8:
            self._degrade_performance()
        elif avg_fps > self.current_params.max_fps * 1.2:
            self._improve_performance()

    def _degrade_performance(self) -> None:
        """Gradually degrade performance to meet target FPS."""
        if self.current_params.resolution_scale > 0.75:
            self.current_params.resolution_scale = max(
                0.75, self.current_params.resolution_scale - 0.1
            )
            logger.info(
                f"Reduced resolution scale to {self.current_params.resolution_scale}"
            )
            return

        if self.current_params.frame_skip < 3:
            self.current_params.frame_skip += 1
            logger.info(f"Increased frame skip to {self.current_params.frame_skip}")
            return

        if self.current_params.batch_size > 1:
            self.current_params.batch_size = max(1, self.current_params.batch_size // 2)
            logger.info(f"Reduced batch size to {self.current_params.batch_size}")

    def _improve_performance(self) -> None:
        """Gradually improve performance if resources allow."""
        if self.current_params.resolution_scale < 1.0:
            self.current_params.resolution_scale = min(
                1.0, self.current_params.resolution_scale + 0.1
            )
            logger.info(
                f"Increased resolution scale to {self.current_params.resolution_scale}"
            )
            return

        if self.current_params.frame_skip > 1:
            self.current_params.frame_skip -= 1
            logger.info(f"Decreased frame skip to {self.current_params.frame_skip}")
            return

        max_batch = self.BATCH_SIZES[self.hardware.tier]
        if self.current_params.batch_size < max_batch:
            self.current_params.batch_size = min(
                max_batch, self.current_params.batch_size * 2
            )
            logger.info(f"Increased batch size to {self.current_params.batch_size}")

    def get_current_params(self) -> ProcessingParams:
        """Get current processing parameters."""
        return self.current_params

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {"avg_fps": 0.0, "stability": 0.0}

        avg_fps = sum(self.performance_history) / len(self.performance_history)
        stability = 1.0 - (
            np.std(self.performance_history) / avg_fps if avg_fps > 0 else 0
        )

        return {
            "avg_fps": avg_fps,
            "stability": stability,
            "resolution_scale": self.current_params.resolution_scale,
            "frame_skip": self.current_params.frame_skip,
            "batch_size": self.current_params.batch_size,
        }

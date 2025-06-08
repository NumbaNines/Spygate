"""
Performance optimization and tier-based feature adaptation for SpygateAI.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    """
    Optimizes performance parameters based on hardware tier.
    Provides adaptive settings for memory usage, thread counts,
    batch sizes, and other performance-critical parameters.
    """

    # Base configuration for each tier
    TIER_CONFIGS = {
        "minimum": {
            "cache_size": 500,  # frames in memory
            "thread_multiplier": 0.5,  # percentage of available threads
            "batch_size": 4,  # frames per batch
            "resolution_scale": 0.75,  # scale factor for frame size
            "prefetch_frames": 10,  # number of frames to prefetch
            "max_parallel_tasks": 2,  # maximum concurrent tasks
            "memory_limit": 0.5,  # fraction of available RAM to use
            "gpu_memory_limit": 0.5,  # fraction of available VRAM to use
            "frame_skip": 2,  # process every Nth frame
        },
        "standard": {
            "cache_size": 1000,
            "thread_multiplier": 0.75,
            "batch_size": 8,
            "resolution_scale": 0.9,
            "prefetch_frames": 20,
            "max_parallel_tasks": 4,
            "memory_limit": 0.6,
            "gpu_memory_limit": 0.7,
            "frame_skip": 1,
        },
        "premium": {
            "cache_size": 2000,
            "thread_multiplier": 0.8,
            "batch_size": 16,
            "resolution_scale": 1.0,
            "prefetch_frames": 30,
            "max_parallel_tasks": 6,
            "memory_limit": 0.7,
            "gpu_memory_limit": 0.8,
            "frame_skip": 1,
        },
        "professional": {
            "cache_size": 4000,
            "thread_multiplier": 0.9,
            "batch_size": 32,
            "resolution_scale": 1.0,
            "prefetch_frames": 50,
            "max_parallel_tasks": 8,
            "memory_limit": 0.8,
            "gpu_memory_limit": 0.9,
            "frame_skip": 1,
        },
    }

    def __init__(self, hardware: HardwareDetector):
        """Initialize the optimizer with hardware information."""
        self.hardware = hardware
        self.current_config = self._get_base_config()
        self._adapt_to_hardware()
        logger.info(f"Initialized {self.hardware.performance_tier} tier optimizer")

    def _get_base_config(self) -> Dict[str, Any]:
        """Get the base configuration for the current hardware tier."""
        tier = self.hardware.performance_tier
        return self.TIER_CONFIGS[tier].copy()

    def _adapt_to_hardware(self):
        """Adapt base configuration to actual hardware capabilities."""
        # Adjust thread count based on available CPU threads
        max_threads = self.hardware.system_info.cpu_threads
        self.current_config["thread_count"] = max(
            2,
            min(
                int(max_threads * self.current_config["thread_multiplier"]),
                max_threads - 1,
            ),
        )

        # Adjust memory limits based on available RAM
        ram_info = self.hardware.get_ram_info()
        self.current_config["memory_limit_mb"] = int(
            ram_info["free"] * self.current_config["memory_limit"]
        )

        # Adjust GPU memory limits if available
        if self.hardware.has_cuda:
            vram_info = self.hardware.get_vram_info()
            self.current_config["gpu_memory_limit_mb"] = int(
                vram_info["free"] * self.current_config["gpu_memory_limit"]
            )
        else:
            self.current_config["gpu_memory_limit_mb"] = 0

        # Adjust cache size based on available memory
        self.current_config["cache_size"] = min(
            self.current_config["cache_size"],
            self.current_config["memory_limit_mb"] // 10,  # Assume ~10MB per frame
        )

    def get_optimal_thread_count(self) -> int:
        """Get the optimal number of threads for parallel processing."""
        return self.current_config["thread_count"]

    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for processing."""
        return self.current_config["batch_size"]

    def get_optimal_cache_size(self) -> int:
        """Get the optimal number of frames to keep in memory cache."""
        return self.current_config["cache_size"]

    def get_resolution_scale(self) -> float:
        """Get the optimal resolution scale factor."""
        return self.current_config["resolution_scale"]

    def get_prefetch_count(self) -> int:
        """Get the optimal number of frames to prefetch."""
        return self.current_config["prefetch_frames"]

    def get_max_parallel_tasks(self) -> int:
        """Get the maximum number of parallel tasks."""
        return self.current_config["max_parallel_tasks"]

    def get_frame_skip(self) -> int:
        """Get the number of frames to skip during processing."""
        return self.current_config["frame_skip"]

    def get_memory_limits(self) -> Dict[str, int]:
        """Get memory limits in megabytes."""
        return {
            "ram": self.current_config["memory_limit_mb"],
            "vram": self.current_config["gpu_memory_limit_mb"],
        }

    def should_use_gpu(self, operation: str) -> bool:
        """
        Determine if GPU should be used for a specific operation.

        Args:
            operation: Operation type (e.g., 'resize', 'normalize', 'detect')

        Returns:
            bool: Whether to use GPU acceleration
        """
        if not self.hardware.has_cuda:
            return False

        # Operations that benefit from GPU acceleration
        gpu_operations = {
            "resize": True,
            "normalize": True,
            "detect": True,
            "transform": True,
            "filter": True,
        }

        # Check if operation benefits from GPU
        if operation not in gpu_operations:
            return False

        # Only use GPU for supported operations on higher tiers
        return self.hardware.performance_tier in ["premium", "professional"]

    def get_operation_batch_size(self, operation: str) -> int:
        """
        Get optimal batch size for specific operations.

        Args:
            operation: Operation type

        Returns:
            int: Optimal batch size
        """
        # Operation-specific batch size multipliers
        multipliers = {
            "detect": 0.5,  # Detection is more memory intensive
            "transform": 0.75,
            "filter": 1.0,
            "resize": 1.5,  # Resizing can handle larger batches
            "normalize": 1.5,
        }

        multiplier = multipliers.get(operation, 1.0)
        return max(1, int(self.current_config["batch_size"] * multiplier))

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.current_config.update(kwargs)
        self._adapt_to_hardware()
        logger.info("Updated optimizer configuration")

    def optimize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Optimize a frame based on current parameters."""
        metrics = {}

        # Apply resolution scaling
        if self.current_config["resolution_scale"] < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.current_config["resolution_scale"])
            new_h = int(h * self.current_config["resolution_scale"])
            frame = cv2.resize(frame, (new_w, new_h))
            metrics["resolution_scale"] = self.current_config["resolution_scale"]

        # Track processing metrics
        metrics.update(
            {
                "target_fps": self.current_config["target_fps"],
                "frame_skip": self.current_config["frame_skip"],
                "batch_size": self.current_config["batch_size"],
            }
        )

        return frame, metrics

    def should_process_frame(self, frame_count: int) -> bool:
        """Determine if a frame should be processed based on frame skip."""
        return frame_count % self.current_config["frame_skip"] == 0

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
        if avg_fps < self.current_config["target_fps"] * 0.8:
            self._degrade_performance()
        elif avg_fps > self.current_config["max_fps"] * 1.2:
            self._improve_performance()

    def _degrade_performance(self) -> None:
        """Gradually degrade performance to meet target FPS."""
        if self.current_config["resolution_scale"] > 0.75:
            self.current_config["resolution_scale"] = max(
                0.75, self.current_config["resolution_scale"] - 0.1
            )
            logger.info(
                f"Reduced resolution scale to {self.current_config['resolution_scale']}"
            )
            return

        if self.current_config["frame_skip"] < 3:
            self.current_config["frame_skip"] += 1
            logger.info(f"Increased frame skip to {self.current_config['frame_skip']}")
            return

        if self.current_config["batch_size"] > 1:
            self.current_config["batch_size"] = max(
                1, self.current_config["batch_size"] // 2
            )
            logger.info(f"Reduced batch size to {self.current_config['batch_size']}")

    def _improve_performance(self) -> None:
        """Gradually improve performance if resources allow."""
        if self.current_config["resolution_scale"] < 1.0:
            self.current_config["resolution_scale"] = min(
                1.0, self.current_config["resolution_scale"] + 0.1
            )
            logger.info(
                f"Increased resolution scale to {self.current_config['resolution_scale']}"
            )
            return

        if self.current_config["frame_skip"] > 1:
            self.current_config["frame_skip"] -= 1
            logger.info(f"Decreased frame skip to {self.current_config['frame_skip']}")
            return

        max_batch = self.TIER_CONFIGS[self.hardware.performance_tier]["batch_size"]
        if self.current_config["batch_size"] < max_batch:
            self.current_config["batch_size"] = min(
                max_batch, self.current_config["batch_size"] * 2
            )
            logger.info(f"Increased batch size to {self.current_config['batch_size']}")

    def get_current_params(self) -> Dict[str, Any]:
        """Get current processing parameters."""
        return self.current_config

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
            "resolution_scale": self.current_config["resolution_scale"],
            "frame_skip": self.current_config["frame_skip"],
            "batch_size": self.current_config["batch_size"],
        }

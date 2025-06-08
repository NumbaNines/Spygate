"""
Hardware requirements and capabilities module for object tracking.

This module defines the hardware requirements and capabilities for different
tracking algorithms and provides utilities to determine the best tracking
approach based on available hardware.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .hardware_monitor import HardwareMonitor


class TrackingAlgorithm(Enum):
    """Available tracking algorithms."""

    CSRT = (
        auto()
    )  # Discriminative Correlation Filter with Channel and Spatial Reliability
    KCF = auto()  # Kernelized Correlation Filters
    MOSSE = auto()  # Minimum Output Sum of Squared Error
    MEDIANFLOW = auto()  # Forward-Backward Error
    GOTURN = auto()  # Generic Object Tracking Using Regression Networks
    DEEPSORT = auto()  # Deep Learning based SORT
    SORT = auto()  # Simple Online and Realtime Tracking


class TrackingMode(Enum):
    """Tracking modes based on hardware capabilities."""

    BASIC = auto()  # CPU-only, basic algorithms
    STANDARD = auto()  # CPU + basic GPU support
    ADVANCED = auto()  # Full GPU support with CUDA
    PROFESSIONAL = auto()  # High-end GPU with multiple models


class TrackingHardwareManager:
    """
    Manages hardware requirements and capabilities for object tracking.

    This class determines the best tracking algorithms and configurations
    based on the available hardware resources.
    """

    # Minimum hardware requirements for each tracking mode
    MODE_REQUIREMENTS = {
        TrackingMode.BASIC: {
            "cpu_cores": 2,
            "ram_gb": 4,
            "gpu_required": False,
            "min_vram_gb": 0,
            "cuda_required": False,
        },
        TrackingMode.STANDARD: {
            "cpu_cores": 4,
            "ram_gb": 8,
            "gpu_required": True,
            "min_vram_gb": 2,
            "cuda_required": True,
        },
        TrackingMode.ADVANCED: {
            "cpu_cores": 6,
            "ram_gb": 16,
            "gpu_required": True,
            "min_vram_gb": 6,
            "cuda_required": True,
        },
        TrackingMode.PROFESSIONAL: {
            "cpu_cores": 8,
            "ram_gb": 32,
            "gpu_required": True,
            "min_vram_gb": 8,
            "cuda_required": True,
        },
    }

    # Algorithm requirements and capabilities
    ALGORITHM_REQUIREMENTS = {
        TrackingAlgorithm.CSRT: {
            "min_mode": TrackingMode.BASIC,
            "gpu_accelerated": False,
            "accuracy": 0.85,
            "speed": 0.6,
            "occlusion_handling": 0.7,
            "recovery": 0.75,
        },
        TrackingAlgorithm.KCF: {
            "min_mode": TrackingMode.BASIC,
            "gpu_accelerated": False,
            "accuracy": 0.75,
            "speed": 0.85,
            "occlusion_handling": 0.6,
            "recovery": 0.65,
        },
        TrackingAlgorithm.MOSSE: {
            "min_mode": TrackingMode.BASIC,
            "gpu_accelerated": False,
            "accuracy": 0.7,
            "speed": 0.9,
            "occlusion_handling": 0.5,
            "recovery": 0.6,
        },
        TrackingAlgorithm.MEDIANFLOW: {
            "min_mode": TrackingMode.BASIC,
            "gpu_accelerated": False,
            "accuracy": 0.7,
            "speed": 0.8,
            "occlusion_handling": 0.6,
            "recovery": 0.7,
        },
        TrackingAlgorithm.GOTURN: {
            "min_mode": TrackingMode.STANDARD,
            "gpu_accelerated": True,
            "accuracy": 0.8,
            "speed": 0.75,
            "occlusion_handling": 0.75,
            "recovery": 0.8,
        },
        TrackingAlgorithm.DEEPSORT: {
            "min_mode": TrackingMode.ADVANCED,
            "gpu_accelerated": True,
            "accuracy": 0.9,
            "speed": 0.7,
            "occlusion_handling": 0.85,
            "recovery": 0.85,
        },
        TrackingAlgorithm.SORT: {
            "min_mode": TrackingMode.STANDARD,
            "gpu_accelerated": True,
            "accuracy": 0.85,
            "speed": 0.8,
            "occlusion_handling": 0.8,
            "recovery": 0.8,
        },
    }

    def __init__(self):
        """Initialize the hardware manager."""
        self.hardware_monitor = HardwareMonitor()
        self.tracking_mode = self._determine_tracking_mode()
        self.available_algorithms = self._get_available_algorithms()

    def _determine_tracking_mode(self) -> TrackingMode:
        """Determine the tracking mode based on hardware capabilities."""
        # Get system info
        system_info = self.hardware_monitor.get_system_info()
        cpu_cores = system_info.get("cpu_count", 0)
        total_ram = system_info.get("total_memory", 0) / (1024**3)  # Convert to GB
        has_gpu = system_info.get("gpu_available", False)
        gpu_vram = system_info.get("gpu_memory", 0) / (1024**3)  # Convert to GB
        has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if has_gpu else False

        # Check requirements from highest to lowest mode
        for mode in reversed(list(TrackingMode)):
            reqs = self.MODE_REQUIREMENTS[mode]
            if (
                cpu_cores >= reqs["cpu_cores"]
                and total_ram >= reqs["ram_gb"]
                and (not reqs["gpu_required"] or has_gpu)
                and (not reqs["cuda_required"] or has_cuda)
                and gpu_vram >= reqs["min_vram_gb"]
            ):
                return mode

        # Default to BASIC mode if no other requirements are met
        return TrackingMode.BASIC

    def _get_available_algorithms(self) -> List[TrackingAlgorithm]:
        """Get list of available tracking algorithms based on tracking mode."""
        available = []
        for algo, reqs in self.ALGORITHM_REQUIREMENTS.items():
            if self._mode_level(reqs["min_mode"]) <= self._mode_level(
                self.tracking_mode
            ) and (
                not reqs["gpu_accelerated"] or self.hardware_monitor.has_gpu_support()
            ):
                available.append(algo)
        return available

    def _mode_level(self, mode: TrackingMode) -> int:
        """Get numeric level for a tracking mode."""
        mode_levels = {
            TrackingMode.BASIC: 0,
            TrackingMode.STANDARD: 1,
            TrackingMode.ADVANCED: 2,
            TrackingMode.PROFESSIONAL: 3,
        }
        return mode_levels[mode]

    def get_recommended_algorithm(
        self,
        priority_accuracy: float = 0.5,
        priority_speed: float = 0.3,
        priority_occlusion: float = 0.1,
        priority_recovery: float = 0.1,
    ) -> TrackingAlgorithm:
        """
        Get the recommended tracking algorithm based on priorities.

        Args:
            priority_accuracy: Weight for accuracy (0-1)
            priority_speed: Weight for speed (0-1)
            priority_occlusion: Weight for occlusion handling (0-1)
            priority_recovery: Weight for recovery capability (0-1)

        Returns:
            The recommended tracking algorithm
        """
        if not self.available_algorithms:
            return TrackingAlgorithm.CSRT  # Default to CSRT if no algorithms available

        # Normalize priorities
        total = (
            priority_accuracy + priority_speed + priority_occlusion + priority_recovery
        )
        priority_accuracy /= total
        priority_speed /= total
        priority_occlusion /= total
        priority_recovery /= total

        # Calculate weighted scores
        best_score = -1
        best_algo = None

        for algo in self.available_algorithms:
            reqs = self.ALGORITHM_REQUIREMENTS[algo]
            score = (
                reqs["accuracy"] * priority_accuracy
                + reqs["speed"] * priority_speed
                + reqs["occlusion_handling"] * priority_occlusion
                + reqs["recovery"] * priority_recovery
            )
            if score > best_score:
                best_score = score
                best_algo = algo

        return best_algo

    def get_tracking_config(self) -> Dict:
        """Get the current tracking configuration."""
        return {
            "tracking_mode": self.tracking_mode,
            "available_algorithms": self.available_algorithms,
            "hardware_tier": self.hardware_monitor.get_performance_tier(),
            "gpu_available": self.hardware_monitor.has_gpu_support(),
            "cuda_available": cv2.cuda.getCudaEnabledDeviceCount() > 0,
        }

    def can_run_algorithm(self, algorithm: TrackingAlgorithm) -> bool:
        """Check if a specific algorithm can run on the current hardware."""
        return algorithm in self.available_algorithms

    def get_algorithm_requirements(self, algorithm: TrackingAlgorithm) -> Dict:
        """Get the requirements for a specific algorithm."""
        return self.ALGORITHM_REQUIREMENTS[algorithm].copy()

    def get_mode_requirements(self, mode: TrackingMode) -> Dict:
        """Get the requirements for a specific tracking mode."""
        return self.MODE_REQUIREMENTS[mode].copy()

"""
Algorithm selection module for video tracking.

This module provides dynamic selection of tracking algorithms based on scene
complexity, hardware capabilities, and performance requirements.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numba import cuda, jit

from ..core.hardware import HardwareDetector, HardwareTier
from ..core.optimizer import TierOptimizer
from ..utils.tracking_hardware import TrackingMode

logger = logging.getLogger(__name__)


class SceneComplexity(Enum):
    """Scene complexity levels for algorithm selection."""

    LOW = auto()  # Simple scenes, few objects, stable lighting
    MEDIUM = auto()  # Moderate complexity, some occlusions
    HIGH = auto()  # Complex scenes, many objects, dynamic lighting
    EXTREME = auto()  # Very complex scenes, heavy occlusions


@dataclass
class AlgorithmProfile:
    """Profile for tracking algorithm capabilities."""

    name: str
    min_fps: float
    max_objects: int
    occlusion_handling: bool
    lighting_invariant: bool
    gpu_accelerated: bool
    memory_usage: float  # MB per object
    accuracy_score: float  # 0-1 scale
    hardware_requirements: HardwareTier
    warmup_frames: int = 10  # Number of frames needed for warmup
    adaptive_params: bool = False  # Whether algorithm supports adaptive parameters


@jit(nopython=True, parallel=True)
def calculate_scene_metrics(frame: np.ndarray) -> tuple[float, float, float, float]:
    """Calculate scene complexity metrics using parallel Numba acceleration.

    Args:
        frame: Input frame

    Returns:
        Tuple of (edge_density, intensity_var, texture_complexity, motion_intensity)
    """
    # Calculate edge density in parallel
    gray = np.mean(frame, axis=2).astype(np.float32)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    edge_density = (np.mean(np.abs(dx)) + np.mean(np.abs(dy))) / 2

    # Calculate intensity variance
    intensity_var = np.var(gray)

    # Calculate texture complexity
    texture = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    texture_complexity = np.mean(texture)

    # Calculate motion intensity (using frame differences)
    motion = np.abs(dx[1:, :] - dx[:-1, :]) + np.abs(dy[:, 1:] - dy[:, :-1])
    motion_intensity = np.mean(motion)

    return edge_density, intensity_var, texture_complexity, motion_intensity


@cuda.jit
def calculate_scene_metrics_gpu(frame, metrics):
    """CUDA kernel for scene complexity metrics calculation."""
    x, y = cuda.grid(2)
    if x < frame.shape[0] - 1 and y < frame.shape[1] - 1:
        # Calculate local metrics
        dx = abs(float(frame[x, y + 1]) - float(frame[x, y]))
        dy = abs(float(frame[x + 1, y]) - float(frame[x, y]))
        texture = abs(float(frame[x + 1, y + 1]) - float(frame[x, y]))

        # Atomic add to global metrics
        cuda.atomic.add(metrics, 0, dx)  # Edge density X
        cuda.atomic.add(metrics, 1, dy)  # Edge density Y
        cuda.atomic.add(metrics, 2, texture)  # Texture complexity


class AlgorithmSelector:
    """Selects optimal tracking algorithms based on scene analysis."""

    def __init__(self):
        """Initialize the algorithm selector."""
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize algorithm profiles with enhanced parameters
        self.algorithms = {
            "CSRT": AlgorithmProfile(
                name="CSRT",
                min_fps=15,
                max_objects=10,
                occlusion_handling=True,
                lighting_invariant=True,
                gpu_accelerated=False,
                memory_usage=50,
                accuracy_score=0.9,
                hardware_requirements=HardwareTier.LOW,
                warmup_frames=15,
                adaptive_params=True,
            ),
            "KCF": AlgorithmProfile(
                name="KCF",
                min_fps=30,
                max_objects=20,
                occlusion_handling=False,
                lighting_invariant=False,
                gpu_accelerated=True,
                memory_usage=30,
                accuracy_score=0.8,
                hardware_requirements=HardwareTier.MEDIUM,
                warmup_frames=5,
                adaptive_params=False,
            ),
            "MOSSE": AlgorithmProfile(
                name="MOSSE",
                min_fps=60,
                max_objects=30,
                occlusion_handling=False,
                lighting_invariant=False,
                gpu_accelerated=True,
                memory_usage=20,
                accuracy_score=0.7,
                hardware_requirements=HardwareTier.LOW,
                warmup_frames=3,
                adaptive_params=False,
            ),
            "DeepSORT": AlgorithmProfile(
                name="DeepSORT",
                min_fps=20,
                max_objects=50,
                occlusion_handling=True,
                lighting_invariant=True,
                gpu_accelerated=True,
                memory_usage=200,
                accuracy_score=0.95,
                hardware_requirements=HardwareTier.HIGH,
                warmup_frames=20,
                adaptive_params=True,
            ),
        }

        # Initialize performance monitoring
        self.scene_history = deque(maxlen=30)  # Store last 30 scene complexities
        self.performance_history = {}  # Algorithm -> performance metrics
        self.algorithm_switches = deque(maxlen=10)  # Track recent algorithm switches
        self.last_switch_time = time.time()
        self.min_switch_interval = 1.0  # Minimum time between algorithm switches

        # Initialize GPU context if available
        self.use_gpu = False
        if self.hardware.has_cuda:
            try:
                cv2.cuda.setDevice(0)
                self.use_gpu = True
                # Preallocate GPU memory
                self.metrics_buffer = cuda.device_array(4, dtype=np.float32)
                logger.info("GPU acceleration enabled for algorithm selection")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU: {e}")

        logger.info(f"Initialized AlgorithmSelector with {self.hardware.tier.name} tier")

    @lru_cache(maxsize=1000)
    def _get_cached_metrics(self, frame_hash: int) -> tuple[float, float, float, float]:
        """Get cached scene metrics for a frame hash."""
        return calculate_scene_metrics(frame)

    def analyze_frame(self, frame: np.ndarray) -> SceneComplexity:
        """Analyze scene complexity of a single frame.

        Args:
            frame: Input frame

        Returns:
            Scene complexity level
        """
        # Try to use cached metrics
        frame_hash = hash(frame.tobytes())
        try:
            metrics = self._get_cached_metrics(frame_hash)
        except Exception:
            # Calculate metrics based on hardware
            if self.use_gpu:
                metrics = self._calculate_metrics_gpu(frame)
            else:
                metrics = calculate_scene_metrics(frame)

        # Calculate complexity score with weighted metrics
        edge_density, intensity_var, texture_complexity, motion_intensity = metrics

        complexity_score = (
            0.3 * edge_density
            + 0.2 * np.sqrt(intensity_var) / 255
            + 0.3 * texture_complexity
            + 0.2 * motion_intensity
        )

        # Update scene history
        self.scene_history.append(complexity_score)

        # Use moving average for stability
        avg_score = np.mean(self.scene_history)

        # Map score to complexity level with hysteresis
        if len(self.scene_history) >= 2:
            prev_score = self.scene_history[-2]
            # Add hysteresis to prevent rapid switching
            threshold_offset = 0.05 if avg_score > prev_score else -0.05
        else:
            threshold_offset = 0

        if avg_score < 0.3 + threshold_offset:
            return SceneComplexity.LOW
        elif avg_score < 0.5 + threshold_offset:
            return SceneComplexity.MEDIUM
        elif avg_score < 0.7 + threshold_offset:
            return SceneComplexity.HIGH
        else:
            return SceneComplexity.EXTREME

    def _calculate_metrics_gpu(self, frame: np.ndarray) -> tuple[float, float, float, float]:
        """Calculate scene metrics using GPU acceleration."""
        # Convert frame to grayscale and float32
        if len(frame.shape) == 3:
            gray = cv2.cuda.cvtColor(cv2.cuda_GpuMat(frame), cv2.COLOR_BGR2GRAY).download()
        else:
            gray = frame

        # Reset metrics buffer
        self.metrics_buffer.copy_to_device(np.zeros(4, dtype=np.float32))

        # Calculate grid dimensions
        block_dim = (16, 16)
        grid_dim = (
            (gray.shape[0] + block_dim[0] - 1) // block_dim[0],
            (gray.shape[1] + block_dim[1] - 1) // block_dim[1],
        )

        # Launch kernel
        calculate_scene_metrics_gpu[grid_dim, block_dim](gray, self.metrics_buffer)

        # Get results
        metrics = self.metrics_buffer.copy_to_host()

        # Normalize metrics
        total_pixels = (gray.shape[0] - 1) * (gray.shape[1] - 1)
        edge_density = (metrics[0] + metrics[1]) / (2 * total_pixels)
        texture_complexity = metrics[2] / total_pixels

        # Calculate intensity variance on CPU (more efficient for this metric)
        intensity_var = np.var(gray)

        # Calculate motion intensity
        if len(self.scene_history) > 0:
            motion_intensity = metrics[3] / total_pixels
        else:
            motion_intensity = 0.0

        return edge_density, intensity_var, texture_complexity, motion_intensity

    def analyze_batch(self, frames: list[np.ndarray]) -> SceneComplexity:
        """Analyze scene complexity for a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            Overall scene complexity level
        """
        complexities = []

        # Process frames in parallel using thread pool
        if self.use_gpu:
            # Process on GPU in batches
            batch_size = min(len(frames), 4)  # Process up to 4 frames at once
            for i in range(0, len(frames), batch_size):
                batch = frames[i : i + batch_size]
                batch_metrics = [self._calculate_metrics_gpu(f) for f in batch]
                batch_complexities = [self._metrics_to_complexity(m) for m in batch_metrics]
                complexities.extend(batch_complexities)
        else:
            # Process on CPU using Numba
            complexities = [self._metrics_to_complexity(calculate_scene_metrics(f)) for f in frames]

        # Count occurrences of each complexity level
        complexity_counts = {
            SceneComplexity.LOW: 0,
            SceneComplexity.MEDIUM: 0,
            SceneComplexity.HIGH: 0,
            SceneComplexity.EXTREME: 0,
        }

        for c in complexities:
            complexity_counts[c] += 1

        # Return highest complexity that occurs in at least 25% of frames
        threshold = len(frames) * 0.25
        for complexity in [
            SceneComplexity.EXTREME,
            SceneComplexity.HIGH,
            SceneComplexity.MEDIUM,
            SceneComplexity.LOW,
        ]:
            if complexity_counts[complexity] >= threshold:
                return complexity

        return SceneComplexity.LOW  # Default to LOW if no clear majority

    def _metrics_to_complexity(self, metrics: tuple[float, float, float, float]) -> SceneComplexity:
        """Convert scene metrics to complexity level."""
        edge_density, intensity_var, texture_complexity, motion_intensity = metrics

        complexity_score = (
            0.3 * edge_density
            + 0.2 * np.sqrt(intensity_var) / 255
            + 0.3 * texture_complexity
            + 0.2 * motion_intensity
        )

        if complexity_score < 0.3:
            return SceneComplexity.LOW
        elif complexity_score < 0.5:
            return SceneComplexity.MEDIUM
        elif complexity_score < 0.7:
            return SceneComplexity.HIGH
        else:
            return SceneComplexity.EXTREME

    def select_algorithm(
        self,
        frame: np.ndarray,
        tracking_mode: TrackingMode,
        n_objects: Optional[int] = None,
        min_fps: Optional[float] = None,
    ) -> tuple[str, dict]:
        """Select the optimal tracking algorithm for the current scene.

        Args:
            frame: Current video frame
            tracking_mode: Desired tracking mode
            n_objects: Number of objects to track (optional)
            min_fps: Minimum required FPS (optional)

        Returns:
            Tuple of (selected algorithm name, configuration parameters)
        """
        # Check if we should allow algorithm switching
        current_time = time.time()
        if (current_time - self.last_switch_time) < self.min_switch_interval:
            # Return last selected algorithm if available
            if self.algorithm_switches:
                last_algo = self.algorithm_switches[-1]
                return last_algo, self._get_algorithm_params(
                    last_algo,
                    self._get_cached_complexity(),
                    tracking_mode,
                )

        # Analyze scene complexity
        complexity = self.analyze_frame(frame)

        # Get hardware constraints
        hw_constraints = self._get_hardware_constraints()

        # Filter algorithms based on requirements
        candidates = self._filter_algorithms(
            complexity,
            tracking_mode,
            n_objects,
            min_fps,
            hw_constraints,
        )

        if not candidates:
            # Fall back to most compatible algorithm
            logger.warning("No ideal algorithm found, using fallback")
            selected = self._get_fallback_algorithm(tracking_mode)
        else:
            # Select best algorithm based on scoring
            candidates_scores = self._score_algorithms(candidates, complexity, tracking_mode)
            selected = candidates_scores[0][0]

        # Update algorithm switch history
        self.algorithm_switches.append(selected)
        self.last_switch_time = current_time

        # Get optimal parameters
        params = self._get_algorithm_params(
            selected,
            complexity,
            tracking_mode,
        )

        # Update performance history
        self._update_performance_history(selected, complexity)

        logger.info(f"Selected {selected} algorithm for {complexity.name} complexity scene")
        return selected, params

    def _get_cached_complexity(self) -> SceneComplexity:
        """Get the most recent scene complexity."""
        if self.scene_history:
            recent_scores = list(self.scene_history)[-5:]  # Last 5 scores
            avg_score = np.mean(recent_scores)

            if avg_score < 0.3:
                return SceneComplexity.LOW
            elif avg_score < 0.5:
                return SceneComplexity.MEDIUM
            elif avg_score < 0.7:
                return SceneComplexity.HIGH
            else:
                return SceneComplexity.EXTREME
        return SceneComplexity.LOW

    def _get_hardware_constraints(self) -> dict[str, Union[float, bool]]:
        """Get current hardware constraints with caching."""
        return {
            "gpu_available": self.hardware.has_cuda,
            "memory_available": self.hardware.available_memory,
            "cpu_cores": self.hardware.cpu_cores,
            "tier": self.hardware.tier,
            "gpu_memory": self.hardware.gpu_memory if self.hardware.has_cuda else 0,
        }

    def _filter_algorithms(
        self,
        complexity: SceneComplexity,
        tracking_mode: TrackingMode,
        n_objects: Optional[int],
        min_fps: Optional[float],
        hw_constraints: dict[str, Union[float, bool]],
    ) -> list[str]:
        """Filter algorithms based on requirements with enhanced criteria."""
        candidates = []

        for name, profile in self.algorithms.items():
            # Check hardware requirements
            if profile.hardware_requirements > hw_constraints["tier"]:
                continue

            # Check GPU requirement
            if profile.gpu_accelerated and not hw_constraints["gpu_available"]:
                continue

            # Check memory requirement
            if n_objects:
                total_memory = profile.memory_usage * n_objects
                if profile.gpu_accelerated:
                    if total_memory > hw_constraints["gpu_memory"]:
                        continue
                else:
                    if total_memory > hw_constraints["memory_available"]:
                        continue

            # Check FPS requirement
            if min_fps and profile.min_fps < min_fps:
                continue

            # Check tracking mode compatibility
            if tracking_mode == TrackingMode.PROFESSIONAL:
                if not (profile.occlusion_handling and profile.lighting_invariant):
                    continue
            elif tracking_mode == TrackingMode.ADVANCED:
                if not profile.occlusion_handling:
                    continue

            # Check scene complexity requirements
            if complexity == SceneComplexity.EXTREME:
                if not (profile.occlusion_handling and profile.lighting_invariant):
                    continue
            elif complexity == SceneComplexity.HIGH:
                if not profile.occlusion_handling:
                    continue

            candidates.append(name)

        return candidates

    def _score_algorithms(
        self,
        candidates: list[str],
        complexity: SceneComplexity,
        tracking_mode: TrackingMode,
    ) -> list[tuple[str, float]]:
        """Score algorithms based on multiple criteria."""
        scores = []

        for name in candidates:
            profile = self.algorithms[name]

            # Base score from profile
            base_score = profile.accuracy_score

            # Adjust based on complexity match
            if complexity == SceneComplexity.EXTREME:
                if profile.occlusion_handling and profile.lighting_invariant:
                    base_score *= 1.2
            elif complexity == SceneComplexity.HIGH:
                if profile.occlusion_handling:
                    base_score *= 1.1

            # Adjust based on tracking mode
            if tracking_mode == TrackingMode.PROFESSIONAL:
                if profile.occlusion_handling and profile.lighting_invariant:
                    base_score *= 1.2
            elif tracking_mode == TrackingMode.ADVANCED:
                if profile.occlusion_handling:
                    base_score *= 1.1

            # Adjust based on hardware match
            if profile.gpu_accelerated and self.hardware.has_cuda:
                base_score *= 1.1

            # Adjust based on historical performance
            if name in self.performance_history:
                perf_score = self.performance_history[name].get("success_rate", 0.8)
                base_score *= 0.7 + 0.3 * perf_score  # Weight history at 30%

            scores.append((name, base_score))

        # Sort by score in descending order
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _get_algorithm_params(
        self,
        algorithm: str,
        complexity: SceneComplexity,
        tracking_mode: TrackingMode,
    ) -> dict:
        """Get optimal parameters for the selected algorithm."""
        profile = self.algorithms[algorithm]

        # Base parameters
        params = {
            "max_objects": profile.max_objects,
            "min_confidence": 0.5,
            "max_age": 30,
            "min_hits": 3,
        }

        # Adjust based on complexity
        if complexity == SceneComplexity.EXTREME:
            params.update(
                {
                    "min_confidence": 0.7,
                    "max_age": 45,
                    "min_hits": 5,
                }
            )
        elif complexity == SceneComplexity.HIGH:
            params.update(
                {
                    "min_confidence": 0.6,
                    "max_age": 35,
                    "min_hits": 4,
                }
            )

        # Adjust based on tracking mode
        if tracking_mode == TrackingMode.PROFESSIONAL:
            params.update(
                {
                    "min_confidence": params["min_confidence"] + 0.1,
                    "min_hits": params["min_hits"] + 1,
                }
            )

        # Add algorithm-specific parameters
        if algorithm == "CSRT":
            params.update(
                {
                    "psr_threshold": 0.8 if complexity == SceneComplexity.EXTREME else 0.7,
                    "num_iterations": 10 if tracking_mode == TrackingMode.PROFESSIONAL else 8,
                }
            )
        elif algorithm == "KCF":
            params.update(
                {
                    "detect_thresh": 0.7 if complexity == SceneComplexity.EXTREME else 0.6,
                    "sigma": 0.2 if tracking_mode == TrackingMode.PROFESSIONAL else 0.1,
                }
            )
        elif algorithm == "DeepSORT":
            params.update(
                {
                    "nn_budget": 100 if tracking_mode == TrackingMode.PROFESSIONAL else 75,
                    "max_cosine_distance": 0.3 if complexity == SceneComplexity.EXTREME else 0.4,
                }
            )

        return params

    def _get_fallback_algorithm(self, tracking_mode: TrackingMode) -> str:
        """Get the most suitable fallback algorithm."""
        if tracking_mode == TrackingMode.PROFESSIONAL and self.hardware.has_cuda:
            return "DeepSORT"
        elif tracking_mode == TrackingMode.ADVANCED:
            return "CSRT"
        else:
            return "KCF"

    def _update_performance_history(self, algorithm: str, complexity: SceneComplexity):
        """Update historical performance metrics for an algorithm."""
        if algorithm not in self.performance_history:
            self.performance_history[algorithm] = {
                "success_count": 0,
                "failure_count": 0,
                "total_frames": 0,
                "complexity_distribution": {
                    SceneComplexity.LOW: 0,
                    SceneComplexity.MEDIUM: 0,
                    SceneComplexity.HIGH: 0,
                    SceneComplexity.EXTREME: 0,
                },
            }

        history = self.performance_history[algorithm]
        history["total_frames"] += 1
        history["complexity_distribution"][complexity] += 1

        # Calculate success rate
        if history["total_frames"] > 0:
            history["success_rate"] = (
                history["success_count"] / (history["success_count"] + history["failure_count"])
                if history["success_count"] + history["failure_count"] > 0
                else 0.8  # Default success rate
            )

    def update_algorithm_performance(
        self,
        algorithm: str,
        success: bool,
        tracking_time: float,
    ):
        """Update performance metrics for an algorithm.

        Args:
            algorithm: Name of the algorithm
            success: Whether tracking was successful
            tracking_time: Time taken for tracking
        """
        if algorithm not in self.performance_history:
            self._update_performance_history(algorithm, SceneComplexity.LOW)

        history = self.performance_history[algorithm]

        if success:
            history["success_count"] += 1
        else:
            history["failure_count"] += 1

        # Update success rate
        history["success_rate"] = history["success_count"] / (
            history["success_count"] + history["failure_count"]
        )

        # Update average tracking time
        if "avg_tracking_time" not in history:
            history["avg_tracking_time"] = tracking_time
        else:
            history["avg_tracking_time"] = (
                0.9 * history["avg_tracking_time"]
                + 0.1 * tracking_time  # Exponential moving average
            )

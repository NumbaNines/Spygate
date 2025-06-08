"""
Formation analysis module for video tracking.

This module provides optimized formation detection and analysis capabilities
with support for parallel processing and GPU acceleration.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Callable
from collections import deque
import time

import cv2
import numpy as np
from numba import cuda, jit
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer

logger = logging.getLogger(__name__)


class FormationAnalysisError(Exception):
    """Base exception for formation analysis errors."""
    pass


class ConfigurationError(FormationAnalysisError):
    """Exception for invalid configuration parameters."""
    pass


class ProcessingError(FormationAnalysisError):
    """Exception for processing-related errors."""
    pass


class Formation(Enum):
    """Common football formations."""

    F_4_4_2 = auto()
    F_4_3_3 = auto()
    F_3_5_2 = auto()
    F_5_3_2 = auto()
    F_4_2_3_1 = auto()
    UNKNOWN = auto()


@dataclass
class FormationConfig:
    """Configuration for formation analysis."""

    min_players: int = 8
    max_distance: float = 50.0  # meters
    temporal_smoothing: bool = True
    smoothing_window: int = 10
    enable_parallel: bool = True
    max_thread_workers: int = 4
    gpu_acceleration: bool = True
    prediction_buffer_size: int = 30
    confidence_threshold: float = 0.7
    
    # Validation ranges
    _min_players_range: Tuple[int, int] = field(default=(4, 11), repr=False)
    _max_distance_range: Tuple[float, float] = field(default=(10.0, 100.0), repr=False)
    _smoothing_window_range: Tuple[int, int] = field(default=(5, 30), repr=False)
    _buffer_size_range: Tuple[int, int] = field(default=(10, 60), repr=False)
    _confidence_range: Tuple[float, float] = field(default=(0.3, 1.0), repr=False)

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_params()
        
    def _validate_params(self):
        """Check if parameters are within valid ranges."""
        if not (self._min_players_range[0] <= self.min_players <= self._min_players_range[1]):
            raise ConfigurationError(
                f"min_players must be between {self._min_players_range[0]} and "
                f"{self._min_players_range[1]}"
            )
            
        if not (self._max_distance_range[0] <= self.max_distance <= self._max_distance_range[1]):
            raise ConfigurationError(
                f"max_distance must be between {self._max_distance_range[0]} and "
                f"{self._max_distance_range[1]} meters"
            )
            
        if not (self._smoothing_window_range[0] <= self.smoothing_window <= self._smoothing_window_range[1]):
            raise ConfigurationError(
                f"smoothing_window must be between {self._smoothing_window_range[0]} and "
                f"{self._smoothing_window_range[1]} frames"
            )
            
        if not (self._buffer_size_range[0] <= self.prediction_buffer_size <= self._buffer_size_range[1]):
            raise ConfigurationError(
                f"prediction_buffer_size must be between {self._buffer_size_range[0]} and "
                f"{self._buffer_size_range[1]} frames"
            )
            
        if not (self._confidence_range[0] <= self.confidence_threshold <= self._confidence_range[1]):
            raise ConfigurationError(
                f"confidence_threshold must be between {self._confidence_range[0]} and "
                f"{self._confidence_range[1]}"
            )


@jit(nopython=True)
def calculate_distances(positions):
    """Calculate pairwise distances between player positions."""
    n = len(positions)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(
                (positions[i][0] - positions[j][0])**2 +
                (positions[i][1] - positions[j][1])**2
            )
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


@jit(nopython=True)
def calculate_formation_metrics(positions):
    """Calculate key metrics for formation analysis."""
    # Calculate centroid
    centroid = np.mean(positions, axis=0)
    
    # Calculate spread (average distance from centroid)
    spread = np.mean(
        np.sqrt(
            np.sum((positions - centroid)**2, axis=1)
        )
    )
    
    # Calculate density (average pairwise distance)
    distances = calculate_distances(positions)
    density = np.mean(distances)
    
    return centroid, spread, density


@cuda.jit
def cuda_calculate_distances(positions, distances):
    """CUDA kernel for calculating pairwise distances."""
    i, j = cuda.grid(2)
    if i < positions.shape[0] and j < positions.shape[0]:
        if i != j:
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distances[i, j] = np.sqrt(dx*dx + dy*dy)


class FormationAnalyzer:
    """Analyzes player formations in video frames."""

    def __init__(
        self,
        config: Optional[FormationConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Initialize the formation analyzer.

        Args:
            config: Formation analysis configuration
            progress_callback: Optional callback for progress updates
                             Function(status_message: str, progress: float)
        """
        try:
            self.config = config or FormationConfig()
            self.progress_callback = progress_callback
            
            # Initialize hardware optimization
            self._update_progress("Detecting hardware capabilities...", 0.1)
            self.hardware = HardwareDetector()
            self.optimizer = TierOptimizer(self.hardware)
            
            # Initialize GPU context if available and requested
            self.use_gpu = False
            if self.config.gpu_acceleration:
                self._update_progress("Initializing GPU acceleration...", 0.3)
                try:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        cv2.cuda.setDevice(0)
                        self.use_gpu = True
                        logger.info("GPU acceleration enabled for formation analysis")
                    else:
                        logger.info("No GPU available, using CPU acceleration")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU: {e}")
                    
            # Initialize analysis buffers
            self._update_progress("Initializing analysis buffers...", 0.6)
            self.position_history = deque(maxlen=self.config.smoothing_window)
            self.formation_history = deque(maxlen=self.config.prediction_buffer_size)
            
            # Initialize clustering
            self._update_progress("Setting up clustering algorithm...", 0.8)
            self.clustering = DBSCAN(
                eps=self.config.max_distance,
                min_samples=3,
                n_jobs=-1 if self.config.enable_parallel else 1,
            )
            
            # Performance monitoring
            self.processing_times = deque(maxlen=100)
            
            self._update_progress("Initialization complete", 1.0)
            logger.info(
                f"Initialized FormationAnalyzer with {self.hardware.tier.name} tier"
            )
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise ProcessingError(f"Failed to initialize FormationAnalyzer: {e}")

    def _update_progress(self, status: str, progress: float) -> None:
        """Update progress through callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(status, progress)
            except Exception as e:
                logger.warning(f"Failed to update progress: {e}")

    def analyze_formation(
        self,
        positions: List[Tuple[float, float]],
        frame_number: int,
        confidence_scores: Optional[List[float]] = None,
    ) -> Dict[str, Union[Formation, Dict]]:
        """Analyze player formation from positions.

        Args:
            positions: List of player positions (x, y)
            frame_number: Current frame number
            confidence_scores: Optional confidence scores for positions

        Returns:
            Dictionary containing formation analysis results

        Raises:
            ProcessingError: If analysis fails
        """
        try:
            start_time = time.time()
            
            self._update_progress("Starting formation analysis...", 0.0)
            
            # Filter positions by confidence if scores provided
            if confidence_scores is not None:
                self._update_progress("Filtering positions by confidence...", 0.1)
                positions = [
                    pos for pos, conf in zip(positions, confidence_scores)
                    if conf >= self.config.confidence_threshold
                ]
                
            # Check minimum players requirement
            if len(positions) < self.config.min_players:
                logger.warning(
                    f"Insufficient players ({len(positions)}) for formation analysis"
                )
                return {
                    "formation": Formation.UNKNOWN,
                    "confidence": 0.0,
                    "metrics": {},
                    "error": "Insufficient players for analysis"
                }
                
            # Convert positions to numpy array
            pos_array = np.array(positions)
            
            # Update position history
            self._update_progress("Updating position history...", 0.2)
            self.position_history.append(pos_array)
            
            # Apply temporal smoothing if enabled
            if (self.config.temporal_smoothing and
                len(self.position_history) >= 2):
                self._update_progress("Applying temporal smoothing...", 0.3)
                pos_array = self._apply_temporal_smoothing()
                
            # Calculate formation metrics
            self._update_progress("Calculating formation metrics...", 0.5)
            metrics = self._calculate_metrics(pos_array)
            
            # Detect formation pattern
            self._update_progress("Detecting formation pattern...", 0.7)
            formation, confidence = self._detect_formation(pos_array, metrics)
            
            # Update formation history
            self.formation_history.append(formation)
            
            # Generate additional analysis
            self._update_progress("Generating detailed analysis...", 0.9)
            analysis = self._analyze_formation_details(pos_array, metrics)
            
            # Update performance metrics
            end_time = time.time()
            self.processing_times.append(end_time - start_time)
            
            self._update_progress("Analysis complete", 1.0)
            
            return {
                "formation": formation,
                "confidence": confidence,
                "metrics": metrics,
                "analysis": analysis,
            }
            
        except Exception as e:
            logger.error(f"Formation analysis failed: {e}")
            raise ProcessingError(f"Formation analysis failed: {e}")

    def _apply_temporal_smoothing(self) -> np.ndarray:
        """Apply temporal smoothing to position data."""
        # Calculate weighted average of recent positions
        weights = np.exp(
            -np.arange(len(self.position_history)) / 2
        )
        weights = weights / np.sum(weights)
        
        smoothed = np.zeros_like(self.position_history[-1])
        for i, pos in enumerate(self.position_history):
            smoothed += weights[i] * pos
            
        return smoothed

    def _calculate_metrics(
        self,
        positions: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate formation metrics."""
        metrics = {}
        
        # Calculate basic metrics
        centroid, spread, density = calculate_formation_metrics(positions)
        metrics.update({
            "centroid": centroid,
            "spread": spread,
            "density": density,
        })
        
        # Calculate convex hull
        hull = ConvexHull(positions)
        metrics["area"] = hull.area
        metrics["perimeter"] = hull.area / hull.volume if hull.volume > 0 else 0
        
        # Calculate clustering metrics
        clusters = self.clustering.fit_predict(positions)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        metrics["n_clusters"] = n_clusters
        
        # Calculate line structure metrics
        metrics.update(
            self._calculate_line_metrics(positions)
        )
        
        return metrics

    def _calculate_line_metrics(
        self,
        positions: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate metrics for line structures in formation."""
        metrics = {}
        
        # Sort positions by y-coordinate (assuming y is depth)
        sorted_pos = positions[positions[:, 1].argsort()]
        
        # Find potential lines
        lines = []
        current_line = [sorted_pos[0]]
        current_y = sorted_pos[0, 1]
        
        for pos in sorted_pos[1:]:
            if abs(pos[1] - current_y) < self.config.max_distance / 3:
                current_line.append(pos)
            else:
                if len(current_line) >= 2:
                    lines.append(np.array(current_line))
                current_line = [pos]
                current_y = pos[1]
                
        if len(current_line) >= 2:
            lines.append(np.array(current_line))
            
        # Calculate line metrics
        metrics["n_lines"] = len(lines)
        
        if lines:
            # Calculate average line straightness
            straightness = []
            for line in lines:
                if len(line) >= 2:
                    # Fit line and calculate R²
                    coeffs = np.polyfit(line[:, 0], line[:, 1], 1)
                    poly = np.poly1d(coeffs)
                    y_pred = poly(line[:, 0])
                    r2 = 1 - (np.sum((line[:, 1] - y_pred)**2) /
                             np.sum((line[:, 1] - np.mean(line[:, 1]))**2))
                    straightness.append(r2)
                    
            metrics["avg_line_straightness"] = (
                np.mean(straightness) if straightness else 0
            )
            
            # Calculate line spacing
            line_positions = [np.mean(line[:, 1]) for line in lines]
            if len(line_positions) >= 2:
                metrics["avg_line_spacing"] = (
                    np.mean(np.diff(sorted(line_positions)))
                )
            else:
                metrics["avg_line_spacing"] = 0
        else:
            metrics["avg_line_straightness"] = 0
            metrics["avg_line_spacing"] = 0
            
        return metrics

    def _detect_formation(
        self,
        positions: np.ndarray,
        metrics: Dict[str, Union[float, np.ndarray]],
    ) -> Tuple[Formation, float]:
        """Detect formation pattern from positions and metrics."""
        # Calculate formation features
        features = self._calculate_formation_features(positions, metrics)
        
        # Match against known formations
        best_match = Formation.UNKNOWN
        best_confidence = 0.0
        
        for formation in Formation:
            if formation == Formation.UNKNOWN:
                continue
                
            confidence = self._match_formation_pattern(
                formation,
                features,
                metrics,
            )
            
            if confidence > best_confidence:
                best_match = formation
                best_confidence = confidence
                
        return best_match, best_confidence

    def _calculate_formation_features(
        self,
        positions: np.ndarray,
        metrics: Dict[str, Union[float, np.ndarray]],
    ) -> Dict[str, float]:
        """Calculate features for formation matching."""
        features = {}
        
        # Number of players in each third
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        y_range = y_max - y_min
        thirds = [
            y_min + i * y_range / 3
            for i in range(4)
        ]
        
        for i in range(3):
            mask = (
                (positions[:, 1] >= thirds[i]) &
                (positions[:, 1] < thirds[i + 1])
            )
            features[f"players_third_{i}"] = np.sum(mask)
            
        # Width utilization
        x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
        features["width_utilization"] = x_range / metrics["spread"]
        
        # Clustering features
        features["cluster_ratio"] = (
            metrics["n_clusters"] / len(positions)
            if len(positions) > 0 else 0
        )
        
        # Line structure features
        features["line_score"] = (
            metrics["avg_line_straightness"] *
            metrics["n_lines"] /
            (metrics["avg_line_spacing"] + 1e-6)
        )
        
        return features

    def _match_formation_pattern(
        self,
        formation: Formation,
        features: Dict[str, float],
        metrics: Dict[str, Union[float, np.ndarray]],
    ) -> float:
        """Match features against a known formation pattern."""
        # Define expected patterns
        patterns = {
            Formation.F_4_4_2: {
                "players_third_0": (4, 0.5),  # (expected value, weight)
                "players_third_1": (4, 0.3),
                "players_third_2": (2, 0.2),
                "width_utilization": (0.8, 0.4),
                "cluster_ratio": (0.3, 0.3),
                "line_score": (0.7, 0.4),
            },
            Formation.F_4_3_3: {
                "players_third_0": (4, 0.5),
                "players_third_1": (3, 0.3),
                "players_third_2": (3, 0.2),
                "width_utilization": (0.9, 0.4),
                "cluster_ratio": (0.35, 0.3),
                "line_score": (0.65, 0.4),
            },
            Formation.F_3_5_2: {
                "players_third_0": (3, 0.5),
                "players_third_1": (5, 0.3),
                "players_third_2": (2, 0.2),
                "width_utilization": (0.85, 0.4),
                "cluster_ratio": (0.4, 0.3),
                "line_score": (0.75, 0.4),
            },
            Formation.F_5_3_2: {
                "players_third_0": (5, 0.5),
                "players_third_1": (3, 0.3),
                "players_third_2": (2, 0.2),
                "width_utilization": (0.75, 0.4),
                "cluster_ratio": (0.35, 0.3),
                "line_score": (0.8, 0.4),
            },
            Formation.F_4_2_3_1: {
                "players_third_0": (4, 0.5),
                "players_third_1": (5, 0.3),
                "players_third_2": (1, 0.2),
                "width_utilization": (0.85, 0.4),
                "cluster_ratio": (0.45, 0.3),
                "line_score": (0.7, 0.4),
            },
        }
        
        if formation not in patterns:
            return 0.0
            
        pattern = patterns[formation]
        confidence = 0.0
        total_weight = 0.0
        
        for feature, (expected, weight) in pattern.items():
            if feature in features:
                # Calculate feature match score
                value = features[feature]
                score = 1.0 - min(abs(value - expected) / expected, 1.0)
                confidence += score * weight
                total_weight += weight
                
        return confidence / total_weight if total_weight > 0 else 0.0

    def _analyze_formation_details(
        self,
        positions: np.ndarray,
        metrics: Dict[str, Union[float, np.ndarray]],
    ) -> Dict[str, Union[float, List]]:
        """Generate detailed formation analysis."""
        analysis = {}
        
        # Calculate team shape characteristics
        analysis["shape"] = {
            "width": np.max(positions[:, 0]) - np.min(positions[:, 0]),
            "depth": np.max(positions[:, 1]) - np.min(positions[:, 1]),
            "area": metrics["area"],
            "compactness": metrics["perimeter"] / (metrics["area"] + 1e-6),
        }
        
        # Analyze defensive line
        defensive_line = positions[positions[:, 1] <= np.percentile(positions[:, 1], 25)]
        if len(defensive_line) >= 2:
            analysis["defensive_line"] = {
                "width": np.max(defensive_line[:, 0]) - np.min(defensive_line[:, 0]),
                "straightness": self._calculate_line_straightness(defensive_line),
                "height": np.mean(defensive_line[:, 1]),
            }
            
        # Analyze attacking structure
        attacking_line = positions[positions[:, 1] >= np.percentile(positions[:, 1], 75)]
        if len(attacking_line) >= 2:
            analysis["attacking_line"] = {
                "width": np.max(attacking_line[:, 0]) - np.min(attacking_line[:, 0]),
                "straightness": self._calculate_line_straightness(attacking_line),
                "depth": np.mean(attacking_line[:, 1]),
            }
            
        # Calculate formation stability
        if len(self.formation_history) >= 2:
            analysis["stability"] = {
                "formation_changes": self._count_formation_changes(),
                "position_variance": self._calculate_position_variance(),
            }
            
        return analysis

    def _calculate_line_straightness(self, positions: np.ndarray) -> float:
        """Calculate how straight a line of players is."""
        if len(positions) < 2:
            return 0.0
            
        # Fit line to positions
        coeffs = np.polyfit(positions[:, 0], positions[:, 1], 1)
        poly = np.poly1d(coeffs)
        
        # Calculate R² score
        y_pred = poly(positions[:, 0])
        r2 = 1 - (np.sum((positions[:, 1] - y_pred)**2) /
                 np.sum((positions[:, 1] - np.mean(positions[:, 1]))**2))
        
        return max(0.0, min(1.0, r2))

    def _count_formation_changes(self) -> int:
        """Count number of formation changes in history."""
        changes = 0
        prev_formation = None
        
        for formation in self.formation_history:
            if prev_formation is not None and formation != prev_formation:
                changes += 1
            prev_formation = formation
            
        return changes

    def _calculate_position_variance(self) -> float:
        """Calculate variance in player positions over time."""
        if len(self.position_history) < 2:
            return 0.0
            
        # Calculate average position change
        changes = []
        for i in range(1, len(self.position_history)):
            if (self.position_history[i].shape ==
                self.position_history[i-1].shape):
                change = np.mean(
                    np.sqrt(
                        np.sum(
                            (self.position_history[i] -
                             self.position_history[i-1])**2,
                            axis=1
                        )
                    )
                )
                changes.append(change)
                
        return np.mean(changes) if changes else 0.0

    def reset(self) -> None:
        """Reset analyzer state."""
        self.position_history.clear()
        self.formation_history.clear()
        self.processing_times.clear()

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.processing_times:
            return {
                "avg_processing_time": 0.0,
                "fps": 0.0,
            }
            
        avg_time = np.mean(self.processing_times)
        return {
            "avg_processing_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
        } 
"""
Formation analysis module for video tracking.

This module provides optimized formation detection and analysis capabilities
with support for parallel processing and GPU acceleration.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numba import cuda, jit
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from ..core.game_detector import GameVersion
from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .player_detector import PlayerDetector

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


class FormationType(Enum):
    """Football formation types for offensive and defensive schemes."""

    # Offensive formations
    SPREAD = "spread"
    I_FORMATION = "i_formation"
    SHOTGUN = "shotgun"
    PISTOL = "pistol"
    UNDER_CENTER = "under_center"
    WILDCAT = "wildcat"
    TRIPS = "trips"
    BUNCH = "bunch"
    EMPTY = "empty"
    GOAL_LINE = "goal_line"

    # Defensive formations
    FOUR_THREE = "4-3"
    THREE_FOUR = "3-4"
    NICKEL = "nickel"
    DIME = "dime"
    QUARTER = "quarter"
    SIX_ONE = "6-1"
    FIVE_TWO = "5-2"
    FOUR_FOUR = "4-4"
    BEAR = "bear"
    COVER_TWO = "cover_2"
    COVER_THREE = "cover_3"
    COVER_FOUR = "cover_4"

    # Special formations
    SPECIAL_TEAMS = "special_teams"
    PUNT = "punt"
    KICK_RETURN = "kick_return"
    FIELD_GOAL = "field_goal"

    # Unknown/Unrecognized
    UNKNOWN = "unknown"


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
    clustering_eps: float = 0.15
    clustering_min_samples: int = 2

    # Validation ranges
    _min_players_range: tuple[int, int] = field(default=(4, 11), repr=False)
    _max_distance_range: tuple[float, float] = field(default=(10.0, 100.0), repr=False)
    _smoothing_window_range: tuple[int, int] = field(default=(5, 30), repr=False)
    _buffer_size_range: tuple[int, int] = field(default=(10, 60), repr=False)
    _confidence_range: tuple[float, float] = field(default=(0.3, 1.0), repr=False)

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

        if not (
            self._smoothing_window_range[0]
            <= self.smoothing_window
            <= self._smoothing_window_range[1]
        ):
            raise ConfigurationError(
                f"smoothing_window must be between {self._smoothing_window_range[0]} and "
                f"{self._smoothing_window_range[1]} frames"
            )

        if not (
            self._buffer_size_range[0] <= self.prediction_buffer_size <= self._buffer_size_range[1]
        ):
            raise ConfigurationError(
                f"prediction_buffer_size must be between {self._buffer_size_range[0]} and "
                f"{self._buffer_size_range[1]} frames"
            )

        if not (
            self._confidence_range[0] <= self.confidence_threshold <= self._confidence_range[1]
        ):
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
                (positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2
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
    spread = np.mean(np.sqrt(np.sum((positions - centroid) ** 2, axis=1)))

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
            distances[i, j] = np.sqrt(dx * dx + dy * dy)


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

            # Initialize player detector
            self._update_progress("Initializing player detector...", 0.85)
            self.player_detector = PlayerDetector(
                confidence_threshold=self.config.confidence_threshold
            )

            # Initialize formation templates
            self._update_progress("Loading formation templates...", 0.9)
            self._initialize_formation_templates()

            self._update_progress("Initialization complete", 1.0)
            logger.info(f"Initialized FormationAnalyzer with {self.hardware.tier.name} tier")

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

    def _initialize_formation_templates(self):
        """Initialize formation templates for pattern matching."""
        # Offensive formation templates (normalized coordinates 0-1)
        self.offensive_templates = {
            FormationType.SPREAD: [
                # QB
                (0.5, 0.2),
                # O-Line (5 players)
                (0.45, 0.3),
                (0.475, 0.3),
                (0.5, 0.3),
                (0.525, 0.3),
                (0.55, 0.3),
                # WRs (3 players)
                (0.15, 0.3),
                (0.85, 0.3),
                (0.3, 0.4),
                # RB
                (0.5, 0.15),
                # TE
                (0.6, 0.3),
            ],
            FormationType.I_FORMATION: [
                # QB
                (0.5, 0.2),
                # O-Line (5 players)
                (0.45, 0.3),
                (0.475, 0.3),
                (0.5, 0.3),
                (0.525, 0.3),
                (0.55, 0.3),
                # WRs (2 players)
                (0.15, 0.3),
                (0.85, 0.3),
                # FB
                (0.5, 0.15),
                # RB
                (0.5, 0.1),
                # TE
                (0.6, 0.3),
            ],
            FormationType.SHOTGUN: [
                # QB
                (0.5, 0.15),
                # O-Line (5 players)
                (0.45, 0.3),
                (0.475, 0.3),
                (0.5, 0.3),
                (0.525, 0.3),
                (0.55, 0.3),
                # WRs (3 players)
                (0.2, 0.35),
                (0.8, 0.35),
                (0.35, 0.4),
                # RB
                (0.45, 0.2),
                # TE
                (0.6, 0.3),
            ],
        }

        # Defensive formation templates (normalized coordinates 0-1)
        self.defensive_templates = {
            FormationType.FOUR_THREE: [
                # D-Line (4 players)
                (0.4, 0.6),
                (0.47, 0.6),
                (0.53, 0.6),
                (0.6, 0.6),
                # LBs (3 players)
                (0.35, 0.7),
                (0.5, 0.7),
                (0.65, 0.7),
                # DBs (4 players)
                (0.2, 0.85),
                (0.4, 0.8),
                (0.6, 0.8),
                (0.8, 0.85),
            ],
            FormationType.THREE_FOUR: [
                # D-Line (3 players)
                (0.4, 0.6),
                (0.5, 0.6),
                (0.6, 0.6),
                # LBs (4 players)
                (0.3, 0.7),
                (0.45, 0.7),
                (0.55, 0.7),
                (0.7, 0.7),
                # DBs (4 players)
                (0.2, 0.85),
                (0.4, 0.8),
                (0.6, 0.8),
                (0.8, 0.85),
            ],
        }

    def analyze_formation(
        self, frame: np.ndarray, game_version: GameVersion, is_offense: bool = True
    ) -> dict:
        """
        Analyze formation from a video frame.

        Args:
            frame: Video frame as numpy array
            game_version: Game version for context
            is_offense: Whether to analyze offensive (True) or defensive (False) formation

        Returns:
            Dictionary containing formation analysis results
        """
        try:
            # Detect players in the frame
            detections = self.player_detector.detect_players(frame)

            # Extract player positions from detections
            player_positions = []
            for detection in detections:
                bbox = detection["bbox"]
                # Calculate center of bounding box
                center_x = (bbox[0] + bbox[2]) / 2 / frame.shape[1]  # Normalize to 0-1
                center_y = (bbox[1] + bbox[3]) / 2 / frame.shape[0]  # Normalize to 0-1
                player_positions.append((center_x, center_y))

            # Check if we have enough players
            if len(player_positions) < self.config.min_players:
                return {
                    "formation_type": None,
                    "confidence": 0.0,
                    "player_positions": player_positions,
                    "clusters": [],
                }

            # Choose appropriate templates based on offense/defense
            templates = self.offensive_templates if is_offense else self.defensive_templates

            # Find best matching formation
            best_formation = None
            best_confidence = 0.0

            for formation_type, template in templates.items():
                similarity = self._calculate_template_similarity(player_positions, template)
                confidence = self._adjust_confidence_for_game(
                    similarity, formation_type, game_version
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_formation = formation_type

            # Cluster player positions
            clusters = self._cluster_positions(player_positions)

            return {
                "formation_type": (
                    best_formation if best_confidence >= self.config.confidence_threshold else None
                ),
                "confidence": best_confidence,
                "player_positions": player_positions,
                "clusters": clusters,
            }

        except Exception as e:
            logger.error(f"Formation analysis failed: {e}")
            return {
                "formation_type": None,
                "confidence": 0.0,
                "player_positions": [],
                "clusters": [],
            }

    def _cluster_positions(self, positions: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Cluster player positions using DBSCAN.

        Args:
            positions: List of (x, y) player positions

        Returns:
            List of cluster centers
        """
        if len(positions) < self.config.clustering_min_samples:
            return []

        try:
            # Convert to numpy array
            pos_array = np.array(positions)

            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.clustering_eps, min_samples=self.config.clustering_min_samples
            ).fit(pos_array)

            # Calculate cluster centers
            clusters = []
            unique_labels = set(clustering.labels_)

            for label in unique_labels:
                if label == -1:  # Noise points
                    continue

                cluster_points = pos_array[clustering.labels_ == label]
                cluster_center = np.mean(cluster_points, axis=0)
                clusters.append((float(cluster_center[0]), float(cluster_center[1])))

            return clusters

        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

    def _calculate_template_similarity(
        self, positions: list[tuple[float, float]], template: list[tuple[float, float]]
    ) -> float:
        """
        Calculate similarity between detected positions and a formation template.

        Args:
            positions: Detected player positions
            template: Formation template positions

        Returns:
            Similarity score between 0 and 1
        """
        if len(positions) != len(template):
            # Penalize for different number of players
            size_penalty = abs(len(positions) - len(template)) / max(len(positions), len(template))
            base_similarity = max(0, 1 - size_penalty)
        else:
            base_similarity = 1.0

        # Calculate minimum distance matching between positions and template
        pos_array = np.array(positions)
        template_array = np.array(template)

        # Use Hungarian algorithm-like approach for optimal matching
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        # Calculate distance matrix
        distances = cdist(pos_array, template_array)

        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(distances)

        # Calculate average minimum distance
        total_distance = distances[row_indices, col_indices].sum()
        avg_distance = total_distance / len(positions)

        # Convert distance to similarity (lower distance = higher similarity)
        distance_similarity = max(0, 1 - avg_distance * 5)  # Scale factor of 5

        return base_similarity * distance_similarity

    def _adjust_confidence_for_game(
        self, base_confidence: float, formation_type: FormationType, game_version: GameVersion
    ) -> float:
        """
        Adjust confidence based on game version characteristics.

        Args:
            base_confidence: Base confidence score
            formation_type: Type of formation
            game_version: Game version

        Returns:
            Adjusted confidence score
        """
        adjustment = 1.0

        if game_version == GameVersion.MADDEN_25:
            # Madden 25 typically has clearer formation patterns
            adjustment = 1.1
        elif game_version == GameVersion.CFB_25:
            # College football might have more variation
            adjustment = 0.9

        # Clamp to valid range
        return min(1.0, max(0.0, base_confidence * adjustment))

    def _update_formation_history(self, formation_type: FormationType, confidence: float):
        """Update formation history for statistics tracking."""
        self.formation_history.append((formation_type, confidence))

    def get_formation_stats(self) -> dict:
        """
        Get formation analysis statistics.

        Returns:
            Dictionary containing formation statistics
        """
        if not self.formation_history:
            return {"total_detections": 0, "avg_confidence": 0.0, "formation_types": {}}

        formations, confidences = zip(*self.formation_history)

        # Count formation types
        formation_counts = {}
        for formation in formations:
            formation_counts[formation] = formation_counts.get(formation, 0) + 1

        return {
            "total_detections": len(self.formation_history),
            "avg_confidence": np.mean(confidences),
            "formation_types": formation_counts,
        }

    def reset(self) -> None:
        """Reset analyzer state."""
        self.position_history.clear()
        self.formation_history.clear()
        self.processing_times.clear()

    def get_performance_stats(self) -> dict[str, float]:
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

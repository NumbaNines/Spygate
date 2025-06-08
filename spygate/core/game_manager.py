"""Game management and integration for multi-game support."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .game_detector import GameDetector, GameProfile, GameVersion
from .hardware import HardwareDetector, PerformanceTier

logger = logging.getLogger(__name__)


@dataclass
class GameSettings:
    """Game-specific settings based on hardware capabilities."""

    resolution: Tuple[int, int]  # Width, height
    frame_rate: int  # Target FPS
    analysis_features: List[str]  # Enabled analysis features
    performance_mode: str  # "quality" or "performance"
    gpu_enabled: bool  # Whether GPU acceleration is enabled


class GameManager:
    """
    Manages game detection, hardware capabilities, and settings optimization
    for multi-game support. Integrates HardwareDetector and GameDetector
    to provide a seamless experience across different games.
    """

    def __init__(self):
        """Initialize the game manager."""
        self.hardware = HardwareDetector()
        self.game_detector = GameDetector()
        self._current_settings: Optional[GameSettings] = None
        self._performance_profiles = self._initialize_performance_profiles()

    def _initialize_performance_profiles(
        self,
    ) -> Dict[Tuple[GameVersion, PerformanceTier], GameSettings]:
        """Initialize performance profiles for each game and hardware tier combination."""
        return {
            (GameVersion.MADDEN_25, PerformanceTier.MINIMUM): GameSettings(
                resolution=(1280, 720),
                frame_rate=30,
                analysis_features=["hud_analysis", "situation_detection"],
                performance_mode="performance",
                gpu_enabled=False,
            ),
            (GameVersion.MADDEN_25, PerformanceTier.STANDARD): GameSettings(
                resolution=(1920, 1080),
                frame_rate=30,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "situation_detection",
                ],
                performance_mode="balanced",
                gpu_enabled=True,
            ),
            (GameVersion.MADDEN_25, PerformanceTier.PREMIUM): GameSettings(
                resolution=(1920, 1080),
                frame_rate=60,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                ],
                performance_mode="quality",
                gpu_enabled=True,
            ),
            (GameVersion.MADDEN_25, PerformanceTier.PROFESSIONAL): GameSettings(
                resolution=(2560, 1440),
                frame_rate=60,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                    "advanced_analytics",
                ],
                performance_mode="quality",
                gpu_enabled=True,
            ),
            (GameVersion.CFB_25, PerformanceTier.MINIMUM): GameSettings(
                resolution=(1280, 720),
                frame_rate=30,
                analysis_features=["hud_analysis", "situation_detection"],
                performance_mode="performance",
                gpu_enabled=False,
            ),
            (GameVersion.CFB_25, PerformanceTier.STANDARD): GameSettings(
                resolution=(1920, 1080),
                frame_rate=30,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "situation_detection",
                ],
                performance_mode="balanced",
                gpu_enabled=True,
            ),
            (GameVersion.CFB_25, PerformanceTier.PREMIUM): GameSettings(
                resolution=(1920, 1080),
                frame_rate=60,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                ],
                performance_mode="quality",
                gpu_enabled=True,
            ),
            (GameVersion.CFB_25, PerformanceTier.PROFESSIONAL): GameSettings(
                resolution=(2560, 1440),
                frame_rate=60,
                analysis_features=[
                    "hud_analysis",
                    "formation_recognition",
                    "player_tracking",
                    "situation_detection",
                    "advanced_analytics",
                ],
                performance_mode="quality",
                gpu_enabled=True,
            ),
        }

    def detect_and_configure(
        self, frame: np.ndarray
    ) -> Tuple[GameVersion, GameSettings]:
        """
        Detect the game from a frame and configure optimal settings.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Tuple of (detected_version, optimal_settings)
        """
        # Detect game version
        version, confidence = self.game_detector.detect_game_version(frame)
        logger.info(
            f"Detected game version {version.value} with confidence {confidence:.2f}"
        )

        # Get hardware tier
        tier = self.hardware.get_performance_tier()
        logger.info(f"Current hardware performance tier: {tier.name}")

        # Get optimal settings
        settings = self.get_optimal_settings(version, tier)
        self._current_settings = settings

        return version, settings

    def get_optimal_settings(
        self, game_version: GameVersion, performance_tier: PerformanceTier
    ) -> GameSettings:
        """
        Get optimal settings for the current game and hardware combination.

        Args:
            game_version: Detected game version
            performance_tier: Current hardware performance tier

        Returns:
            GameSettings with optimal configuration
        """
        # Get base settings for the game/tier combination
        base_settings = self._performance_profiles[(game_version, performance_tier)]

        # Check if all features are supported by the game
        supported_features = []
        for feature in base_settings.analysis_features:
            if self.game_detector.is_feature_supported(feature, game_version):
                supported_features.append(feature)
            else:
                logger.warning(
                    f"Feature {feature} not supported by {game_version.value}"
                )

        # Create optimized settings
        optimized_settings = GameSettings(
            resolution=base_settings.resolution,
            frame_rate=base_settings.frame_rate,
            analysis_features=supported_features,
            performance_mode=base_settings.performance_mode,
            gpu_enabled=base_settings.gpu_enabled and self.hardware.has_gpu_support(),
        )

        return optimized_settings

    def get_interface_mapping(self) -> GameProfile:
        """
        Get the interface mapping for the current game.

        Returns:
            GameProfile with interface mapping information

        Raises:
            ValueError: If no game is currently detected
        """
        return self.game_detector.get_interface_mapping()

    def update_settings(self, frame: Optional[np.ndarray] = None) -> GameSettings:
        """
        Update settings based on current conditions or new frame.

        Args:
            frame: Optional new frame to analyze for game detection

        Returns:
            Updated GameSettings
        """
        if frame is not None:
            # Re-detect game and configure if new frame provided
            _, settings = self.detect_and_configure(frame)
        else:
            # Update settings based on current game and hardware
            version = self.game_detector.current_game
            if not version:
                raise ValueError("No game currently detected")
            tier = self.hardware.get_performance_tier()
            settings = self.get_optimal_settings(version, tier)

        self._current_settings = settings
        return settings

    @property
    def current_settings(self) -> Optional[GameSettings]:
        """Get the current game settings."""
        return self._current_settings

    @property
    def current_game(self) -> Optional[GameVersion]:
        """Get the currently detected game version."""
        return self.game_detector.current_game

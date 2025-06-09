"""Strategy mapping and analysis for multi-game support."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .game_detector import GameVersion
from .game_manager import GameManager

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of football strategies that can be mapped across games."""

    FORMATION = "formation"
    PLAY = "play"
    CONCEPT = "concept"
    SITUATION = "situation"


@dataclass
class UniversalStrategy:
    """Represents a strategy that can be mapped across different games."""

    strategy_type: StrategyType
    name: str
    core_principle: str
    game_implementations: dict[GameVersion, str]
    effectiveness_data: dict[GameVersion, float]
    metadata: dict[str, Any]


class StrategyMapper:
    """
    Maps and analyzes strategies across different game versions.
    Provides functionality to translate strategies between games and
    track their effectiveness.
    """

    def __init__(self, game_manager: GameManager):
        """Initialize the strategy mapper."""
        self.game_manager = game_manager
        self._strategy_db: dict[str, UniversalStrategy] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the StrategyMapper."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def register_strategy(
        self,
        strategy_type: StrategyType,
        name: str,
        core_principle: str,
        game_implementations: dict[GameVersion, str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> UniversalStrategy:
        """
        Register a new universal strategy that can be mapped across games.

        Args:
            strategy_type: Type of the strategy (formation, play, etc.)
            name: Name of the strategy
            core_principle: Core football principle this strategy represents
            game_implementations: Mapping of game versions to specific implementations
            metadata: Additional strategy metadata

        Returns:
            The created UniversalStrategy object
        """
        if name in self._strategy_db:
            self.logger.warning(f"Strategy {name} already exists, updating...")

        strategy = UniversalStrategy(
            strategy_type=strategy_type,
            name=name,
            core_principle=core_principle,
            game_implementations=game_implementations,
            effectiveness_data={version: 0.0 for version in game_implementations},
            metadata=metadata or {},
        )

        self._strategy_db[name] = strategy
        self.logger.info(f"Registered strategy: {name}")
        return strategy

    def get_strategy(self, name: str) -> Optional[UniversalStrategy]:
        """
        Retrieve a strategy by name.

        Args:
            name: Name of the strategy to retrieve

        Returns:
            UniversalStrategy if found, None otherwise
        """
        return self._strategy_db.get(name)

    def translate_strategy(
        self, strategy_name: str, source_game: GameVersion, target_game: GameVersion
    ) -> Optional[str]:
        """
        Translate a strategy from one game version to another.

        Args:
            strategy_name: Name of the strategy to translate
            source_game: Source game version
            target_game: Target game version

        Returns:
            The strategy implementation for the target game, or None if not available
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"Strategy not found: {strategy_name}")
            return None

        if target_game not in strategy.game_implementations:
            self.logger.warning(f"Strategy {strategy_name} not implemented for {target_game}")
            return None

        return strategy.game_implementations[target_game]

    def update_effectiveness(
        self, strategy_name: str, game_version: GameVersion, effectiveness: float
    ) -> None:
        """
        Update the effectiveness rating for a strategy in a specific game.

        Args:
            strategy_name: Name of the strategy
            game_version: Game version to update
            effectiveness: New effectiveness rating (0.0 to 1.0)
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"Strategy not found: {strategy_name}")
            return

        if game_version not in strategy.game_implementations:
            self.logger.warning(f"Strategy {strategy_name} not implemented for {game_version}")
            return

        strategy.effectiveness_data[game_version] = effectiveness
        self.logger.info(
            f"Updated effectiveness for {strategy_name} in {game_version}: {effectiveness:.2f}"
        )

    def get_similar_strategies(
        self, strategy_name: str, threshold: float = 0.7
    ) -> list[UniversalStrategy]:
        """
        Find strategies similar to the given one based on core principles.

        Args:
            strategy_name: Name of the strategy to find similar ones for
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            List of similar strategies
        """
        base_strategy = self.get_strategy(strategy_name)
        if not base_strategy:
            self.logger.warning(f"Strategy not found: {strategy_name}")
            return []

        similar_strategies = []
        for strategy in self._strategy_db.values():
            if strategy.name == strategy_name:
                continue

            # TODO: Implement more sophisticated similarity comparison
            # For now, just check if they share the same strategy type
            if strategy.strategy_type == base_strategy.strategy_type:
                similar_strategies.append(strategy)

        return similar_strategies

    def analyze_cross_game_effectiveness(self, strategy_name: str) -> dict[GameVersion, float]:
        """
        Analyze the effectiveness of a strategy across different games.

        Args:
            strategy_name: Name of the strategy to analyze

        Returns:
            Dictionary mapping game versions to effectiveness ratings
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"Strategy not found: {strategy_name}")
            return {}

        return strategy.effectiveness_data.copy()

    def get_strategies_by_type(self, strategy_type: StrategyType) -> list[UniversalStrategy]:
        """
        Get all strategies of a specific type.

        Args:
            strategy_type: Type of strategies to retrieve

        Returns:
            List of strategies matching the specified type
        """
        return [
            strategy
            for strategy in self._strategy_db.values()
            if strategy.strategy_type == strategy_type
        ]

    def export_strategy_data(self) -> dict[str, Any]:
        """
        Export all strategy data for storage or analysis.

        Returns:
            Dictionary containing all strategy data
        """
        return {
            name: {
                "type": strategy.strategy_type.value,
                "core_principle": strategy.core_principle,
                "implementations": {
                    version.value: impl for version, impl in strategy.game_implementations.items()
                },
                "effectiveness": {
                    version.value: score for version, score in strategy.effectiveness_data.items()
                },
                "metadata": strategy.metadata,
            }
            for name, strategy in self._strategy_db.items()
        }

    def import_strategy_data(self, data: dict[str, Any]) -> None:
        """
        Import strategy data from storage.

        Args:
            data: Strategy data to import
        """
        for name, strategy_data in data.items():
            game_implementations = {
                GameVersion(version): impl
                for version, impl in strategy_data["implementations"].items()
            }

            strategy = UniversalStrategy(
                strategy_type=StrategyType(strategy_data["type"]),
                name=name,
                core_principle=strategy_data["core_principle"],
                game_implementations=game_implementations,
                effectiveness_data={
                    GameVersion(version): score
                    for version, score in strategy_data["effectiveness"].items()
                },
                metadata=strategy_data["metadata"],
            )

            self._strategy_db[name] = strategy

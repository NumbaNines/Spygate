"""Tracking data models for game analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TrackingData:
    """Represents tracking data for game analysis."""
    frame_number: int
    timestamp: float
    game_state: Dict[str, any]
    detections: List[Dict[str, any]]
    confidence: float
    metadata: Optional[Dict[str, any]] = None

    def get_team_players(self, team_id: int) -> dict[int, tuple[int, int]]:
        """
        Get positions of all players in a specific team.

        Args:
            team_id: The team's ID (1 or 2)

        Returns:
            Dictionary mapping player IDs to their positions
        """
        return {
            player_id: pos
            for player_id, pos in self.player_positions.items()
            if self.player_teams.get(player_id) == team_id
        }

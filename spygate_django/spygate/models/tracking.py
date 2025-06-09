from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TrackingData:
    """
    Data class for storing tracking information for a single frame.
    """

    frame_id: int
    player_positions: dict[int, tuple[int, int]]  # player_id -> (x, y)
    player_teams: dict[int, int]  # player_id -> team_id
    ball_position: Optional[tuple[int, int]] = None  # (x, y) or None if not detected
    frame_timestamp: Optional[float] = None  # timestamp in seconds

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

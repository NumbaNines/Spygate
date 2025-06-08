from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class TrackingData:
    """Class for storing object tracking data."""
    
    # Ball tracking data
    ball_positions: List[Tuple[float, float]]
    ball_confidences: List[float]
    
    # Player tracking data
    player_positions: List[List[Tuple[float, float]]]  # List of positions for each player
    player_ids: List[int]
    player_confidences: List[List[float]]  # List of confidence values for each player
    
    # Team data
    team_formations: List[List[Tuple[float, float]]]  # List of formations for each team
    team_possession: List[float]  # Possession percentage for each team
    team_formation_stability: List[float]  # Formation stability score for each team
    
    # Frame metadata
    frame_width: int
    frame_height: int
    frame_number: int
    timestamp: float
    
    # Optional tracking data
    ball_velocities: Optional[List[Tuple[float, float]]] = None
    player_velocities: Optional[List[List[Tuple[float, float]]]] = None
    
    @property
    def ball_prediction_quality(self) -> float:
        """Calculate the overall quality of ball predictions."""
        if not self.ball_confidences:
            return 0.0
        return float(np.mean(self.ball_confidences))
    
    @property
    def player_tracking_quality(self) -> List[float]:
        """Calculate the tracking quality for each player."""
        if not self.player_confidences:
            return []
        return [float(np.mean(conf)) for conf in self.player_confidences]
    
    @property
    def team_formation_quality(self) -> List[float]:
        """Calculate the formation quality for each team."""
        if not self.team_formation_stability:
            return []
        return self.team_formation_stability
    
    def get_motion_vectors(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get all motion vectors for visualization."""
        vectors = []
        
        # Add ball motion if available
        if self.ball_velocities and len(self.ball_positions) > 1:
            for i in range(len(self.ball_positions) - 1):
                vectors.append((
                    self.ball_positions[i],
                    self.ball_positions[i + 1]
                ))
        
        # Add player motion if available
        if self.player_velocities:
            for player_positions in self.player_positions:
                if len(player_positions) > 1:
                    for i in range(len(player_positions) - 1):
                        vectors.append((
                            player_positions[i],
                            player_positions[i + 1]
                        ))
        
        return vectors
    
    def get_heatmap_positions(self, team_index: int) -> List[Tuple[float, float]]:
        """Get all player positions for a team's heat map."""
        if team_index < 0 or team_index >= len(self.team_formations):
            return []
        
        positions = []
        for formation in self.team_formations[team_index]:
            positions.extend(formation)
        return positions
    
    def get_ball_prediction_data(self) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Get ball position and confidence data for prediction visualization."""
        return self.ball_positions, self.ball_confidences 
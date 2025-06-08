from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.tracking_pipeline import TrackingPipeline
from ..models.tracking import TrackingData

class TrackingService:
    """
    Service class for managing tracking data and coordinating between tracking components.
    """
    
    def __init__(self):
        self._tracking_data: Dict[int, TrackingData] = {}
        self._current_frame_id: Optional[int] = None
        
    def update_tracking_data(self, frame_id: int, tracking_data: TrackingData) -> None:
        """
        Update tracking data for a specific frame.
        
        Args:
            frame_id: The frame ID
            tracking_data: New tracking data for the frame
        """
        self._tracking_data[frame_id] = tracking_data
        self._current_frame_id = frame_id
        
    def get_tracking_data(self, frame_id: Optional[int] = None) -> Optional[TrackingData]:
        """
        Get tracking data for a specific frame or current frame if not specified.
        
        Args:
            frame_id: Optional frame ID to get data for
            
        Returns:
            TrackingData if available, None otherwise
        """
        if frame_id is None:
            frame_id = self._current_frame_id
            
        if frame_id is None:
            return None
            
        return self._tracking_data.get(frame_id)
        
    def clear_tracking_data(self) -> None:
        """Clear all tracking data."""
        self._tracking_data.clear()
        self._current_frame_id = None
        
    def get_player_trajectory(self, player_id: int, frame_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Get trajectory points for a specific player.
        
        Args:
            player_id: The player's ID
            frame_range: Optional tuple of (start_frame, end_frame)
            
        Returns:
            List of (x, y) coordinates
        """
        trajectory = []
        frames = self._tracking_data.keys()
        
        if frame_range:
            start_frame, end_frame = frame_range
            frames = [f for f in frames if start_frame <= f <= end_frame]
            
        for frame_id in sorted(frames):
            data = self._tracking_data[frame_id]
            if player_id in data.player_positions:
                x, y = data.player_positions[player_id]
                trajectory.append((x, y))
                
        return trajectory
        
    def get_team_formation(self, team_id: int, frame_id: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Get current formation points for a team.
        
        Args:
            team_id: The team's ID (1 or 2)
            frame_id: Optional specific frame ID
            
        Returns:
            List of (x, y) coordinates for team players
        """
        data = self.get_tracking_data(frame_id)
        if not data:
            return []
            
        formation = []
        for player_id, pos in data.player_positions.items():
            if data.player_teams.get(player_id) == team_id:
                formation.append(pos)
                
        return formation 
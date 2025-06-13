"""
Game state tracking for SpygateAI.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class GameState:
    """Tracks the current state of the game."""
    
    # Core game state
    down: Optional[int] = None
    distance: Optional[int] = None
    territory: Optional[str] = None  # "OWN" or "OPPONENT"
    possession_team: Optional[str] = None  # "home" or "away"
    
    # Detection confidence
    confidence: float = 0.0
    last_update_frame: int = 0
    
    # Additional state (for future use)
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    quarter: Optional[int] = None
    time_remaining: Optional[str] = None
    timeouts_home: Optional[int] = None
    timeouts_away: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Check if the current state is valid."""
        return all([
            self.down is not None,
            self.distance is not None,
            self.territory is not None,
            self.possession_team is not None,
            self.confidence > 0.4  # Minimum confidence threshold
        ])
        
    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            "down": self.down,
            "distance": self.distance,
            "territory": self.territory,
            "possession_team": self.possession_team,
            "confidence": self.confidence,
            "last_update_frame": self.last_update_frame,
            "score_home": self.score_home,
            "score_away": self.score_away,
            "quarter": self.quarter,
            "time_remaining": self.time_remaining,
            "timeouts_home": self.timeouts_home,
            "timeouts_away": self.timeouts_away
        }
        
    def update_from_dict(self, data: Dict) -> None:
        """Update state from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_situation_description(self) -> str:
        """Get a human-readable description of the current situation."""
        if not self.is_valid():
            return "Invalid game state"
            
        down_map = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        down_str = down_map.get(self.down, str(self.down))
        
        description = f"{down_str} & {self.distance}"
        
        if self.territory:
            description += f" in {self.territory} territory"
            
        if self.possession_team:
            description += f" ({self.possession_team} possession)"
            
        if all([self.score_home is not None, self.score_away is not None]):
            description += f" | Score: {self.score_home}-{self.score_away}"
            
        if self.quarter:
            description += f" | Q{self.quarter}"
            
        if self.time_remaining:
            description += f" | {self.time_remaining}"
            
        return description 
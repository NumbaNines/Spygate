"""Annotation model for video timeline."""

import uuid
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtGui import QColor


@dataclass
class Annotation:
    """Represents a video annotation."""

    timestamp: float  # Time in seconds
    text: str  # Annotation text
    color: str  # Color in hex format (#RRGGBB)
    player_name: str  # Name of the player ("Self" or "Opponent: Name")
    duration: float = 5.0  # Duration in seconds (default: 5 seconds)
    id: Optional[str] = None  # Unique identifier

    def __post_init__(self):
        """Initialize the ID if not provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())

    def duration(self) -> float:
        """Get the duration of the annotation in seconds."""
        return self.duration

    def overlaps(self, other: "Annotation") -> bool:
        """Check if this annotation overlaps with another one."""
        return (
            (self.timestamp <= other.timestamp <= self.timestamp + self.duration)
            or (
                self.timestamp <= other.timestamp + other.duration <= self.timestamp + self.duration
            )
            or (other.timestamp <= self.timestamp <= other.timestamp + other.duration)
        )

    def to_dict(self) -> dict:
        """Convert the annotation to a dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "text": self.text,
            "color": self.color,
            "player_name": self.player_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Create an annotation from a dictionary."""
        return cls(
            timestamp=data["timestamp"],
            text=data["text"],
            color=data["color"],
            player_name=data["player_name"],
            duration=data.get("duration", 5.0),
            id=data.get("id"),
        )

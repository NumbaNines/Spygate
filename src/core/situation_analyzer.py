"""
Game situation analyzer for processing OCR results and determining game state.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SituationType(Enum):
    """Types of game situations that can be detected."""

    NORMAL = auto()
    RED_ZONE = auto()
    GOAL_LINE = auto()
    TWO_MINUTE_WARNING = auto()
    HURRY_UP = auto()
    THIRD_AND_LONG = auto()
    THIRD_AND_SHORT = auto()
    FOURTH_DOWN = auto()
    FIRST_AND_GOAL = auto()
    TWO_POINT_CONVERSION = auto()
    FIELD_GOAL_RANGE = auto()
    GARBAGE_TIME = auto()
    CLOSE_GAME = auto()
    BLOWOUT = auto()


@dataclass
class GameSituation:
    """Represents a detected game situation."""

    # Basic game state
    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[int] = None
    quarter: Optional[int] = None
    time_remaining: Optional[str] = None
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    possession: Optional[str] = None  # 'home' or 'away'
    is_no_huddle: bool = False
    confidence: float = 0.0

    # Field position indicators
    in_red_zone: bool = False
    in_goal_line: bool = False
    in_field_goal_range: bool = False

    # Time-based situations
    in_two_minute: bool = False
    is_hurry_up: bool = False
    is_garbage_time: bool = False

    # Score-based situations
    is_close_game: bool = False
    is_blowout: bool = False
    point_differential: Optional[int] = None

    # Down & distance specific situations
    is_third_and_long: bool = False
    is_third_and_short: bool = False
    is_fourth_down: bool = False
    is_first_and_goal: bool = False
    is_two_point_attempt: bool = False

    # Detected situation types
    situation_types: set[SituationType] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Initialize the set of situation types after instance creation."""
        if not isinstance(self.situation_types, set):
            self.situation_types = set()


class SituationAnalyzer:
    """Analyzes OCR results to determine game situations."""

    def __init__(self) -> None:
        """Initialize the situation analyzer."""
        # Regular expressions for parsing OCR text
        self.down_distance_pattern = re.compile(r"(\d)[THND]{2}\s*&\s*(\d{1,2})")
        self.game_clock_pattern = re.compile(r"(\d{2}):(\d{2})")
        self.score_pattern = re.compile(r"\d{1,2}")

        # Temporal consistency tracking
        self.last_situation: Optional[GameSituation] = None
        self.situation_history: list[GameSituation] = []
        self.consistency_threshold = 0.8  # Minimum confidence for temporal consistency

        # Situation thresholds
        self.THIRD_AND_LONG_THRESHOLD = 7
        self.THIRD_AND_SHORT_THRESHOLD = 2
        self.RED_ZONE_THRESHOLD = 20
        self.GOAL_LINE_THRESHOLD = 5
        self.FIELD_GOAL_RANGE_THRESHOLD = 35
        self.CLOSE_GAME_THRESHOLD = 8
        self.BLOWOUT_THRESHOLD = 21
        self.TWO_MINUTE_WARNING_TIME = "2:00"
        self.GARBAGE_TIME_THRESHOLD = 28  # Points difference for garbage time

    def parse_down_distance(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse down and distance from OCR text.

        Args:
            text: OCR text from down & distance display

        Returns:
            Tuple of (down, distance)
        """
        match = self.down_distance_pattern.search(text)
        if match:
            try:
                down = int(match.group(1))
                distance = int(match.group(2))
                if 1 <= down <= 4 and 1 <= distance <= 99:
                    return down, distance
            except ValueError:
                pass
        return None, None

    def parse_game_clock(self, text: str) -> Optional[str]:
        """Parse game clock from OCR text.

        Args:
            text: OCR text from game clock display

        Returns:
            Formatted time string or None
        """
        match = self.game_clock_pattern.search(text)
        if match:
            try:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                if 0 <= minutes <= 15 and 0 <= seconds <= 59:
                    return f"{minutes:02d}:{seconds:02d}"
            except ValueError:
                pass
        return None

    def parse_score(self, text: str) -> Optional[int]:
        """Parse score from OCR text.

        Args:
            text: OCR text from score display

        Returns:
            Score as integer or None
        """
        match = self.score_pattern.search(text)
        if match:
            try:
                score = int(match.group())
                if 0 <= score <= 99:
                    return score
            except ValueError:
                pass
        return None

    def check_temporal_consistency(self, current: GameSituation) -> GameSituation:
        """Check temporal consistency with previous situations.

        Args:
            current: Current game situation

        Returns:
            Validated game situation
        """
        if not self.last_situation:
            self.last_situation = current
            return current

        # Rules for temporal consistency
        if self.last_situation.down is not None and current.down is not None and current.down not in [1, self.last_situation.down + 1] and current.confidence < self.consistency_threshold:
            current.down = self.last_situation.down

        if self.last_situation.score_home is not None and current.score_home is not None and abs(current.score_home - self.last_situation.score_home) > 8 and current.confidence < self.consistency_threshold:
            current.score_home = self.last_situation.score_home

        if self.last_situation.score_away is not None and current.score_away is not None and abs(current.score_away - self.last_situation.score_away) > 8 and current.confidence < self.consistency_threshold:
            current.score_away = self.last_situation.score_away

        # Update history
        self.last_situation = current
        self.situation_history.append(current)
        if len(self.situation_history) > 300:  # Keep last ~10 seconds at 30fps
            self.situation_history.pop(0)

        return current

    def determine_situation_types(self, situation: GameSituation) -> set[SituationType]:
        """Determine the types of situations present.

        Args:
            situation: Current game situation

        Returns:
            Set of detected situation types
        """
        types = set()

        # Field position situations
        if situation.yard_line is not None:
            if situation.yard_line <= self.RED_ZONE_THRESHOLD:
                types.add(SituationType.RED_ZONE)
            if situation.yard_line <= self.GOAL_LINE_THRESHOLD:
                types.add(SituationType.GOAL_LINE)
            if situation.yard_line <= self.FIELD_GOAL_RANGE_THRESHOLD:
                types.add(SituationType.FIELD_GOAL_RANGE)

        # Down & distance situations
        if situation.down == 3:
            if situation.distance is not None:
                if situation.distance >= self.THIRD_AND_LONG_THRESHOLD:
                    types.add(SituationType.THIRD_AND_LONG)
                elif situation.distance <= self.THIRD_AND_SHORT_THRESHOLD:
                    types.add(SituationType.THIRD_AND_SHORT)
        elif situation.down == 4:
            types.add(SituationType.FOURTH_DOWN)

        if situation.down == 1 and situation.distance == 0:
            types.add(SituationType.FIRST_AND_GOAL)

        # Time-based situations
        if situation.time_remaining == self.TWO_MINUTE_WARNING_TIME:
            types.add(SituationType.TWO_MINUTE_WARNING)
        if situation.is_hurry_up:
            types.add(SituationType.HURRY_UP)

        # Score-based situations
        if situation.point_differential is not None:
            if abs(situation.point_differential) >= self.GARBAGE_TIME_THRESHOLD:
                types.add(SituationType.GARBAGE_TIME)
                types.add(SituationType.BLOWOUT)
            elif abs(situation.point_differential) <= self.CLOSE_GAME_THRESHOLD:
                types.add(SituationType.CLOSE_GAME)

        # Special situations
        if situation.is_two_point_attempt:
            types.add(SituationType.TWO_POINT_CONVERSION)

        # If no special situations detected, mark as normal
        if not types:
            types.add(SituationType.NORMAL)

        return types

    def analyze_frame(
        self, ocr_results: dict[str, tuple[str, float]], detections: list[dict[str, Any]]
    ) -> GameSituation:
        """Analyze a frame's OCR results and detections to determine the game situation.

        Args:
            ocr_results: Dictionary mapping HUD elements to (text, confidence) tuples
            detections: List of object detections with class and bbox information

        Returns:
            Current game situation
        """
        situation = GameSituation()

        # Process down & distance
        if "down_distance" in ocr_results:
            text, conf = ocr_results["down_distance"]
            down, distance = self.parse_down_distance(text)
            situation.down = down
            situation.distance = distance
            situation.confidence = conf

        # Process game clock
        if "game_clock" in ocr_results:
            text, conf = ocr_results["game_clock"]
            situation.time_remaining = self.parse_game_clock(text)
            situation.confidence = max(situation.confidence, conf)

        # Process scores
        if "score" in ocr_results:
            text, conf = ocr_results["score"]
            score = self.parse_score(text)
            if score is not None:
                if situation.score_home is None:
                    situation.score_home = score
                else:
                    situation.score_away = score
                situation.confidence = max(situation.confidence, conf)

        # Calculate point differential if both scores available
        if situation.score_home is not None and situation.score_away is not None:
            situation.point_differential = situation.score_home - situation.score_away

        # Check for no-huddle based on detections
        situation.is_no_huddle = any(d["class"] == "no_huddle" for d in detections)

        # Apply temporal consistency checks
        situation = self.check_temporal_consistency(situation)

        # Determine situation types
        situation.situation_types = self.determine_situation_types(situation)

        return situation

    def get_situation_description(self, situation: GameSituation) -> str:
        """Get a human-readable description of the game situation.

        Args:
            situation: Current game situation

        Returns:
            Description of the current situation
        """
        desc_parts = []

        # Basic game state
        if situation.down is not None and situation.distance is not None:
            desc_parts.append(f"{situation.down}th & {situation.distance}")

        if situation.yard_line is not None:
            desc_parts.append(f"at the {situation.yard_line} yard line")

        if situation.time_remaining:
            desc_parts.append(f"{situation.time_remaining} remaining")

        if situation.score_home is not None and situation.score_away is not None:
            desc_parts.append(f"Score: {situation.score_home}-{situation.score_away}")

        # Special situations
        if SituationType.RED_ZONE in situation.situation_types:
            desc_parts.append("RED ZONE")
        if SituationType.GOAL_LINE in situation.situation_types:
            desc_parts.append("GOAL LINE")
        if SituationType.TWO_MINUTE_WARNING in situation.situation_types:
            desc_parts.append("TWO MINUTE WARNING")
        if situation.is_no_huddle:
            desc_parts.append("NO HUDDLE")

        return " | ".join(desc_parts)

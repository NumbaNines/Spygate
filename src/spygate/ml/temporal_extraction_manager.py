#!/usr/bin/env python3
"""
Temporal Extraction Manager for SpygateAI
=========================================
Smart OCR extraction with temporal confidence voting to handle OCR errors
and optimize performance by extracting elements at appropriate frequencies.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Single OCR extraction result with metadata."""

    value: Any
    confidence: float
    timestamp: float
    raw_text: str = ""
    method: str = ""


@dataclass
class TemporalVote:
    """Temporal voting data for a specific value."""

    value: Any
    votes: int = 0
    total_confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    raw_texts: List[str] = field(default_factory=list)

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.votes if self.votes > 0 else 0.0

    @property
    def stability_score(self) -> float:
        """Higher score for values that appear consistently over time."""
        time_span = self.last_seen - self.first_seen
        consistency = self.votes / max(1, time_span)  # votes per second
        return self.avg_confidence * 0.7 + min(1.0, consistency) * 0.3


class TemporalExtractionManager:
    """
    Manages OCR extraction timing and temporal confidence voting.

    Key Features:
    1. Frequency-based extraction (clocks every frame, scores every 10s)
    2. Temporal confidence voting (best guess over time window)
    3. Smart triggers (only extract when change is possible)
    4. Performance optimization (75% reduction in OCR calls)
    """

    def __init__(self):
        # Extraction frequencies (seconds) - FIXED: More reasonable intervals for production
        self.extraction_intervals = {
            "game_clock": 3.0,  # Every 3 seconds (changes frequently, need regular updates)
            "play_clock": 3.0,  # Every 3 seconds (changes frequently, need regular updates)
            "down_distance": 4.0,  # Every 4 seconds (changes between plays, allow frequent detection)
            "scores": 10.0,  # Every 10 seconds (changes when scoring)
            "team_names": 60.0,  # Every minute (rarely changes)
        }

        # Temporal voting windows (seconds)
        self.voting_windows = {
            "game_clock": 2.0,  # 2 second window for clock voting
            "play_clock": 1.5,  # 1.5 second window for play clock
            "down_distance": 5.0,  # 5 second window for down/distance
            "scores": 15.0,  # 15 second window for scores
            "team_names": 30.0,  # 30 second window for team names
        }

        # Minimum votes required for confidence
        self.min_votes = {
            "game_clock": 2,  # Need 2+ votes for clock
            "play_clock": 2,  # Need 2+ votes for play clock
            "down_distance": 3,  # Need 3+ votes for down/distance
            "scores": 2,  # Need 2+ votes for scores
            "team_names": 1,  # Need 1+ vote for team names
        }

        # Last extraction timestamps
        self.last_extractions = {}

        # Temporal voting data
        self.voting_data = defaultdict(
            lambda: defaultdict(list)
        )  # element_type -> value -> [ExtractionResult]

        # Current best guesses
        self.current_values = {}

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "skipped_extractions": 0,
            "voting_decisions": 0,
        }

    def should_extract(
        self,
        element_type: str,
        current_time: float,
        game_state_changed: bool = False,
        possession_changed: bool = False,
    ) -> bool:
        """
        Determine if we should run OCR extraction for this element type.

        Args:
            element_type: Type of element to extract
            current_time: Current timestamp
            game_state_changed: Whether game state indicators changed
            possession_changed: Whether possession triangle flipped
        """
        # Check if enough time has passed since last extraction
        last_time = self.last_extractions.get(element_type, 0)
        time_since_last = current_time - last_time
        min_interval = self.extraction_intervals[element_type]

        if time_since_last < min_interval:
            self.extraction_stats["skipped_extractions"] += 1
            logger.debug(
                f"⏰ Temporal SKIP: {element_type} - {time_since_last:.1f}s < {min_interval}s interval"
            )
            return False

        # Smart triggers for specific elements
        if element_type in ["down_distance", "game_clock", "play_clock"]:
            # Extract when game state changes (new play, etc.) or interval reached
            should_extract = game_state_changed or time_since_last >= min_interval
            logger.debug(
                f"⏰ Temporal ALLOW: {element_type} - {time_since_last:.1f}s >= {min_interval}s interval"
            )
            return should_extract

        elif element_type == "scores":
            # Extract when possession changes (potential scoring) or timeout
            return possession_changed or time_since_last >= min_interval

        elif element_type == "team_names":
            # Extract rarely (team names don't change)
            return time_since_last >= min_interval

        return True

    def add_extraction_result(self, element_type: str, result: ExtractionResult) -> None:
        """Add a new OCR extraction result to temporal voting."""
        current_time = result.timestamp

        # Update last extraction time
        self.last_extractions[element_type] = current_time
        self.extraction_stats["total_extractions"] += 1

        # Add to voting data
        value_key = str(result.value)  # Convert to string for consistent keys
        self.voting_data[element_type][value_key].append(result)

        # Clean old data outside voting window
        self._clean_old_voting_data(element_type, current_time)

        # Update current best guess
        self._update_best_guess(element_type, current_time)

    def _clean_old_voting_data(self, element_type: str, current_time: float) -> None:
        """Remove voting data outside the temporal window."""
        window_size = self.voting_windows[element_type]
        cutoff_time = current_time - window_size

        for value_key in list(self.voting_data[element_type].keys()):
            # Filter out old results
            old_results = self.voting_data[element_type][value_key]
            new_results = [r for r in old_results if r.timestamp >= cutoff_time]

            if new_results:
                self.voting_data[element_type][value_key] = new_results
            else:
                # Remove empty entries
                del self.voting_data[element_type][value_key]

    def _update_best_guess(self, element_type: str, current_time: float) -> None:
        """Update the best guess for this element type using temporal voting."""
        voting_data = self.voting_data[element_type]

        if not voting_data:
            return

        # Calculate votes for each value
        candidates = []

        for value_key, results in voting_data.items():
            if not results:
                continue

            # Create temporal vote
            vote = TemporalVote(
                value=results[0].value,  # Use actual value from first result
                votes=len(results),
                total_confidence=sum(r.confidence for r in results),
                first_seen=min(r.timestamp for r in results),
                last_seen=max(r.timestamp for r in results),
                raw_texts=[r.raw_text for r in results],
            )

            candidates.append(vote)

        if not candidates:
            return

        # Find best candidate using stability score
        best_candidate = max(candidates, key=lambda v: v.stability_score)

        # Only update if we have enough votes and confidence
        min_votes_required = self.min_votes[element_type]
        if best_candidate.votes >= min_votes_required and best_candidate.avg_confidence >= 0.5:
            old_value = self.current_values.get(element_type)
            self.current_values[element_type] = {
                "value": best_candidate.value,
                "confidence": best_candidate.avg_confidence,
                "stability_score": best_candidate.stability_score,
                "votes": best_candidate.votes,
                "timestamp": current_time,
                "raw_texts": best_candidate.raw_texts,
            }

            # Log significant changes
            if old_value and old_value["value"] != best_candidate.value:
                logger.info(
                    f"📊 {element_type} changed: {old_value['value']} → {best_candidate.value} "
                    f"(confidence: {best_candidate.avg_confidence:.2f}, votes: {best_candidate.votes})"
                )

            self.extraction_stats["voting_decisions"] += 1

    def get_current_value(self, element_type: str) -> Optional[Dict[str, Any]]:
        """Get the current best guess for an element type."""
        return self.current_values.get(element_type)

    def get_all_current_values(self) -> Dict[str, Any]:
        """Get all current best guesses."""
        return self.current_values.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total = self.extraction_stats["total_extractions"]
        skipped = self.extraction_stats["skipped_extractions"]

        return {
            "total_extractions": total,
            "skipped_extractions": skipped,
            "extraction_efficiency": skipped / (total + skipped) if (total + skipped) > 0 else 0,
            "voting_decisions": self.extraction_stats["voting_decisions"],
            "current_values_count": len(self.current_values),
        }

    def reset_element(self, element_type: str) -> None:
        """Reset voting data for a specific element (useful for new plays, etc.)."""
        if element_type in self.voting_data:
            del self.voting_data[element_type]
        if element_type in self.current_values:
            del self.current_values[element_type]

        logger.debug(f"🔄 Reset temporal voting for {element_type}")

    def force_extraction(self, element_type: str) -> None:
        """Force extraction on next frame (useful for triggered events)."""
        self.last_extractions[element_type] = 0  # Reset timestamp to force extraction

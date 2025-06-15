"""
Situational Prediction Engine for SpygateAI
===========================================
Uses football game logic to predict down/distance based on game flow,
providing a hybrid approach with OCR for maximum accuracy.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PlayOutcome(Enum):
    """Possible play outcomes that affect down/distance."""

    INCOMPLETE_PASS = "incomplete"
    COMPLETE_PASS = "complete"
    RUSHING_ATTEMPT = "rush"
    SACK = "sack"
    PENALTY = "penalty"
    TURNOVER = "turnover"
    TOUCHDOWN = "touchdown"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"
    FIRST_DOWN = "first_down"
    UNKNOWN = "unknown"


@dataclass
class GameSituation:
    """Current game situation for prediction."""

    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[int] = None
    territory: Optional[str] = None  # "own" or "opponent"
    possession_team: Optional[str] = None
    time_remaining: Optional[str] = None
    score_differential: Optional[int] = None
    quarter: Optional[int] = None

    # Confidence in current values
    down_confidence: float = 0.0
    distance_confidence: float = 0.0

    # Historical tracking
    last_known_down: Optional[int] = None
    last_known_distance: Optional[int] = None
    last_known_yard_line: Optional[int] = None


@dataclass
class PredictionResult:
    """Result of situational prediction."""

    predicted_down: Optional[int] = None
    predicted_distance: Optional[int] = None
    confidence: float = 0.0
    reasoning: str = ""
    ocr_agreement: bool = False
    recommended_action: str = ""


class SituationalPredictor:
    """
    Predicts down/distance using football game logic.

    Features:
    - Play sequence tracking
    - Yard line progression analysis
    - Possession change detection
    - Time-based situation analysis
    - OCR validation and correction
    """

    def __init__(self):
        """Initialize the situational predictor."""
        self.game_history: List[GameSituation] = []
        self.play_sequence: List[Dict] = []
        self.possession_changes: List[Dict] = []

        # Prediction confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.3

        # Common down/distance patterns
        self.common_patterns = {
            "first_down": [(1, 10), (1, 15), (1, 20)],  # 1st & 10/15/20
            "second_down": [(2, 3), (2, 5), (2, 7), (2, 10), (2, 15)],
            "third_down": [(3, 1), (3, 3), (3, 5), (3, 8), (3, 10), (3, 15)],
            "fourth_down": [(4, 1), (4, 2), (4, 3), (4, 5), (4, 10)],
        }

    def predict_next_down_distance(
        self,
        current_situation: GameSituation,
        ocr_result: Optional[Tuple[int, int]] = None,
        ocr_confidence: float = 0.0,
    ) -> PredictionResult:
        """
        Predict the next down/distance using situational logic.

        Args:
            current_situation: Current game state
            ocr_result: OCR detected (down, distance) tuple
            ocr_confidence: Confidence in OCR result

        Returns:
            Prediction result with confidence and reasoning
        """
        prediction = PredictionResult()

        # Store current situation in history
        self.game_history.append(current_situation)

        # Get prediction based on game logic
        logic_prediction = self._predict_from_game_logic(current_situation)

        # Compare with OCR if available
        if ocr_result:
            ocr_down, ocr_distance = ocr_result
            prediction.ocr_agreement = (
                logic_prediction.predicted_down == ocr_down
                and logic_prediction.predicted_distance == ocr_distance
            )

            # Hybrid decision making
            if prediction.ocr_agreement and ocr_confidence > 0.5:
                # OCR and logic agree - high confidence
                prediction.predicted_down = ocr_down
                prediction.predicted_distance = ocr_distance
                prediction.confidence = min(
                    0.95, logic_prediction.confidence + ocr_confidence * 0.3
                )
                prediction.reasoning = f"OCR and logic agree: {ocr_down} & {ocr_distance}"
                prediction.recommended_action = "use_prediction"

            elif ocr_confidence > 0.7 and logic_prediction.confidence < 0.5:
                # High OCR confidence, low logic confidence - trust OCR
                prediction.predicted_down = ocr_down
                prediction.predicted_distance = ocr_distance
                prediction.confidence = ocr_confidence * 0.8
                prediction.reasoning = f"High OCR confidence: {ocr_down} & {ocr_distance}"
                prediction.recommended_action = "use_ocr"

            elif logic_prediction.confidence > 0.7 and ocr_confidence < 0.5:
                # High logic confidence, low OCR confidence - trust logic
                prediction.predicted_down = logic_prediction.predicted_down
                prediction.predicted_distance = logic_prediction.predicted_distance
                prediction.confidence = logic_prediction.confidence
                prediction.reasoning = f"Logic prediction: {logic_prediction.reasoning}"
                prediction.recommended_action = "use_logic"

            else:
                # Conflict or both low confidence - use weighted average approach
                if self._is_reasonable_down_distance(ocr_down, ocr_distance):
                    # OCR result is at least reasonable
                    weight_ocr = ocr_confidence / (ocr_confidence + logic_prediction.confidence)
                    weight_logic = logic_prediction.confidence / (
                        ocr_confidence + logic_prediction.confidence
                    )

                    if weight_ocr > 0.6:
                        prediction.predicted_down = ocr_down
                        prediction.predicted_distance = ocr_distance
                        prediction.confidence = ocr_confidence * 0.7
                        prediction.reasoning = f"OCR weighted choice: {ocr_down} & {ocr_distance}"
                    else:
                        prediction.predicted_down = logic_prediction.predicted_down
                        prediction.predicted_distance = logic_prediction.predicted_distance
                        prediction.confidence = logic_prediction.confidence * 0.7
                        prediction.reasoning = (
                            f"Logic weighted choice: {logic_prediction.reasoning}"
                        )

                    prediction.recommended_action = "weighted_decision"
                else:
                    # OCR result is unreasonable - use logic
                    prediction.predicted_down = logic_prediction.predicted_down
                    prediction.predicted_distance = logic_prediction.predicted_distance
                    prediction.confidence = logic_prediction.confidence * 0.8
                    prediction.reasoning = (
                        f"OCR unreasonable, using logic: {logic_prediction.reasoning}"
                    )
                    prediction.recommended_action = "reject_ocr"
        else:
            # No OCR result - use pure logic
            prediction.predicted_down = logic_prediction.predicted_down
            prediction.predicted_distance = logic_prediction.predicted_distance
            prediction.confidence = logic_prediction.confidence
            prediction.reasoning = f"Pure logic: {logic_prediction.reasoning}"
            prediction.recommended_action = "use_logic"

        logger.debug(
            f"Prediction: {prediction.predicted_down} & {prediction.predicted_distance} "
            f"(conf: {prediction.confidence:.2f}) - {prediction.reasoning}"
        )

        return prediction

    def _predict_from_game_logic(self, situation: GameSituation) -> PredictionResult:
        """Predict down/distance using pure game logic with deep historical context."""
        prediction = PredictionResult()

        # Check if we have enough history for prediction
        if len(self.game_history) < 2:
            return self._predict_from_common_patterns(situation)

        # Analyze deeper historical context (last 5-10 plays)
        drive_context = self._analyze_drive_context()

        # Get previous situation
        prev_situation = self.game_history[-2]

        # Detect possession change
        if (
            situation.possession_team
            and prev_situation.possession_team
            and situation.possession_team != prev_situation.possession_team
        ):
            # Possession changed - likely 1st & 10
            prediction.predicted_down = 1
            prediction.predicted_distance = 10
            prediction.confidence = 0.85
            prediction.reasoning = "Possession change detected"
            return prediction

        # Check for drive patterns (multiple first downs in sequence)
        if drive_context["consecutive_first_downs"] >= 2:
            # Team is moving the ball well - likely another first down
            prediction.predicted_down = 1
            prediction.predicted_distance = 10
            prediction.confidence = 0.8
            prediction.reasoning = f"Drive momentum: {drive_context['consecutive_first_downs']} consecutive first downs"
            return prediction

        # Check for stalled drive pattern (multiple 3rd downs)
        if drive_context["recent_third_downs"] >= 2:
            # Drive is struggling - more conservative prediction
            if prev_situation.down and prev_situation.down >= 3:
                prediction.predicted_down = 1
                prediction.predicted_distance = 10
                prediction.confidence = 0.7
                prediction.reasoning = "Stalled drive pattern - likely punt/turnover reset"
                return prediction

        # Detect significant yard line change
        if (
            situation.yard_line
            and prev_situation.yard_line
            and abs(situation.yard_line - prev_situation.yard_line) > 15
        ):
            # Big yard line change - likely new series
            prediction.predicted_down = 1
            prediction.predicted_distance = 10
            prediction.confidence = 0.75
            prediction.reasoning = (
                f"Large yard line change: {prev_situation.yard_line} → {situation.yard_line}"
            )
            return prediction

        # Check for red zone efficiency patterns
        if drive_context["in_red_zone"] and drive_context["red_zone_plays"] >= 3:
            # Extended red zone drive - goal line situations more likely
            if situation.yard_line and situation.yard_line <= 5:
                prediction.predicted_down = prev_situation.down + 1 if prev_situation.down else 1
                prediction.predicted_distance = situation.yard_line
                prediction.confidence = 0.8
                prediction.reasoning = "Extended red zone drive - goal line situation"
                return prediction

        # Predict based on down progression
        if prev_situation.down and prev_situation.distance:
            return self._predict_down_progression(prev_situation, situation)

        # Fallback to common patterns
        return self._predict_from_common_patterns(situation)

    def _predict_down_progression(
        self, prev: GameSituation, current: GameSituation
    ) -> PredictionResult:
        """Predict based on down progression logic."""
        prediction = PredictionResult()

        # Calculate yard line change
        yard_change = 0
        if current.yard_line and prev.yard_line:
            # Account for territory changes
            if current.territory == prev.territory:
                yard_change = current.yard_line - prev.yard_line
            else:
                # Territory changed - calculate actual field position change
                if prev.territory == "own":
                    prev_field_pos = prev.yard_line
                else:
                    prev_field_pos = 100 - prev.yard_line

                if current.territory == "own":
                    current_field_pos = current.yard_line
                else:
                    current_field_pos = 100 - current.yard_line

                yard_change = current_field_pos - prev_field_pos

        # Determine if first down was achieved
        if yard_change >= prev.distance:
            # First down achieved
            prediction.predicted_down = 1
            prediction.predicted_distance = 10
            prediction.confidence = 0.9
            prediction.reasoning = (
                f"First down: gained {yard_change} yards (needed {prev.distance})"
            )
            return prediction

        # Normal down progression
        if prev.down < 4:
            prediction.predicted_down = prev.down + 1
            prediction.predicted_distance = max(1, prev.distance - yard_change)
            prediction.confidence = 0.8
            prediction.reasoning = f"Down progression: {prev.down} & {prev.distance} → {prediction.predicted_down} & {prediction.predicted_distance}"
        else:
            # 4th down - likely turnover or punt
            prediction.predicted_down = 1
            prediction.predicted_distance = 10
            prediction.confidence = 0.7
            prediction.reasoning = "After 4th down - likely new possession"

        return prediction

    def _predict_from_common_patterns(self, situation: GameSituation) -> PredictionResult:
        """Predict using common down/distance patterns."""
        prediction = PredictionResult()

        # REMOVED: Default to most common situation logic
        # Instead, return None values when insufficient context
        prediction.predicted_down = None
        prediction.predicted_distance = None
        prediction.confidence = 0.0
        prediction.reasoning = "Insufficient context for prediction"

        # Adjust based on field position if available
        if situation.yard_line and situation.territory:
            if situation.territory == "opponent" and situation.yard_line <= 10:
                # Red zone - could be goal line situation
                prediction.predicted_down = 1
                prediction.predicted_distance = situation.yard_line
                prediction.confidence = 0.6
                prediction.reasoning = f"Red zone: 1st & Goal from {situation.yard_line}"
            elif situation.territory == "opponent" and situation.yard_line <= 20:
                # Near red zone - but don't assume 1st & 10
                prediction.predicted_down = None
                prediction.predicted_distance = None
                prediction.confidence = 0.2
                prediction.reasoning = "Near red zone: Insufficient context for specific prediction"

        # Time/quarter context alone is not enough for prediction
        if situation.quarter == 4 and situation.time_remaining:
            # 4th quarter - more varied situations, don't assume anything
            prediction.confidence = 0.1
            prediction.reasoning = "4th quarter: High variance, insufficient context"

        return prediction

    def _is_reasonable_down_distance(self, down: int, distance: int) -> bool:
        """Check if down/distance combination is reasonable."""
        if not (1 <= down <= 4):
            return False

        if not (1 <= distance <= 99):
            return False

        # Check against common patterns
        for pattern_list in self.common_patterns.values():
            if (down, distance) in pattern_list:
                return True

        # Allow reasonable variations - EXPANDED for real game situations
        if down == 1 and 1 <= distance <= 30:  # 1st down can be 1-30 yards
            return True
        if down == 2 and 1 <= distance <= 25:  # 2nd down can be 1-25 yards
            return True
        if (
            down == 3 and 1 <= distance <= 30
        ):  # 3rd down can be 1-30 yards (includes long 3rd downs)
            return True
        if down == 4 and 1 <= distance <= 20:  # 4th down can be 1-20 yards
            return True

        return False

    def validate_ocr_with_logic(
        self,
        ocr_down: int,
        ocr_distance: int,
        ocr_confidence: float,
        current_situation: GameSituation,
    ) -> Dict[str, any]:
        """
        Validate OCR result against game logic.

        Returns:
            Validation result with corrected values if needed
        """
        # Get logic prediction
        logic_prediction = self._predict_from_game_logic(current_situation)

        # Check if OCR is reasonable
        is_reasonable = self._is_reasonable_down_distance(ocr_down, ocr_distance)

        # Check agreement with logic
        logic_agrees = (
            logic_prediction.predicted_down == ocr_down
            and logic_prediction.predicted_distance == ocr_distance
        )

        result = {
            "original_ocr": (ocr_down, ocr_distance),
            "ocr_confidence": ocr_confidence,
            "is_reasonable": is_reasonable,
            "logic_prediction": (
                logic_prediction.predicted_down,
                logic_prediction.predicted_distance,
            ),
            "logic_confidence": logic_prediction.confidence,
            "logic_agrees": logic_agrees,
            "recommended_down": None,
            "recommended_distance": None,
            "final_confidence": 0.0,
            "correction_applied": False,
            "reasoning": "",
        }

        # Decision logic - FIXED: More conservative, trust reasonable OCR results
        if logic_agrees and is_reasonable and ocr_confidence > 0.3:
            # Perfect agreement
            result["recommended_down"] = ocr_down
            result["recommended_distance"] = ocr_distance
            result["final_confidence"] = min(
                0.95, ocr_confidence + logic_prediction.confidence * 0.2
            )
            result["reasoning"] = "OCR and logic agree"

        elif is_reasonable and ocr_confidence > 0.5:  # LOWERED from 0.7 to 0.5
            # Trust reasonable OCR with moderate confidence
            result["recommended_down"] = ocr_down
            result["recommended_distance"] = ocr_distance
            result["final_confidence"] = ocr_confidence * 0.9
            result["reasoning"] = "Reasonable OCR trusted"

        elif logic_prediction.confidence > 0.8 and (
            not is_reasonable or ocr_confidence < 0.3
        ):  # RAISED from 0.7 to 0.8
            # Only trust logic over OCR when logic is very confident AND OCR is poor/unreasonable
            result["recommended_down"] = logic_prediction.predicted_down
            result["recommended_distance"] = logic_prediction.predicted_distance
            result["final_confidence"] = logic_prediction.confidence
            result["correction_applied"] = True
            result["reasoning"] = f"Logic correction: {logic_prediction.reasoning}"

        else:
            # FIXED: More conservative weighted decision - favor OCR when reasonable
            if is_reasonable and ocr_confidence > 0.3:  # NEW: Trust reasonable OCR
                result["recommended_down"] = ocr_down
                result["recommended_distance"] = ocr_distance
                result["final_confidence"] = ocr_confidence * 0.8
                result["reasoning"] = "OCR trusted (reasonable)"
            elif ocr_confidence > logic_prediction.confidence:
                result["recommended_down"] = ocr_down
                result["recommended_distance"] = ocr_distance
                result["final_confidence"] = ocr_confidence * 0.8
                result["reasoning"] = "OCR weighted choice"
            else:
                result["recommended_down"] = logic_prediction.predicted_down
                result["recommended_distance"] = logic_prediction.predicted_distance
                result["final_confidence"] = logic_prediction.confidence * 0.8
                result["correction_applied"] = True
                result["reasoning"] = "Logic weighted choice"

        return result

    def update_game_state(self, situation: GameSituation) -> None:
        """Update the predictor with new game state information."""
        # TEMPORAL VALIDATION: Check if new play validates previous detection
        if len(self.game_history) >= 1:
            validation_result = self._validate_previous_play_with_current(situation)
            if validation_result:
                logger.info(f"🔄 TEMPORAL: {validation_result}")

        self.game_history.append(situation)

        # Keep history manageable
        if len(self.game_history) > 100:
            self.game_history = self.game_history[-50:]

    def get_prediction_confidence(
        self, down: int, distance: int, situation: GameSituation
    ) -> float:
        """Get confidence score for a specific down/distance in current situation."""
        # Check if it matches logic prediction
        logic_prediction = self._predict_from_game_logic(situation)

        if (
            logic_prediction.predicted_down == down
            and logic_prediction.predicted_distance == distance
        ):
            return logic_prediction.confidence

        # Check if it's a reasonable alternative
        if self._is_reasonable_down_distance(down, distance):
            return 0.4  # Reasonable but not predicted

        return 0.1  # Unreasonable

    def _handle_special_situations(
        self, situation: GameSituation, ocr_text: str = None, region_color: dict = None
    ) -> Optional[PredictionResult]:
        """Handle special game situations that affect down/distance logic."""
        prediction = PredictionResult()

        # Check for penalty situations using FLAG detection
        if self._detect_penalty_situation(ocr_text, region_color):
            # During penalty, down/distance usually stays the same or repeats
            if len(self.game_history) > 0:
                last_situation = self.game_history[-1]
                prediction.predicted_down = last_situation.down  # Don't default to 1
                prediction.predicted_distance = last_situation.distance  # Don't default to 10
                prediction.confidence = (
                    0.9 if (last_situation.down and last_situation.distance) else 0.3
                )
                prediction.reasoning = "Penalty detected (FLAG/yellow) - down repeats"
                return prediction
            else:
                # No history available - can't predict during penalty
                prediction.predicted_down = None
                prediction.predicted_distance = None
                prediction.confidence = 0.1
                prediction.reasoning = "Penalty detected but no historical context"
                return prediction

        # Check for red zone goal line situations
        if situation.yard_line and situation.territory == "opponent" and situation.yard_line <= 5:
            prediction.predicted_down = 1
            prediction.predicted_distance = situation.yard_line  # Goal line
            prediction.confidence = 0.8
            prediction.reasoning = f"Goal line situation: 1st & Goal from {situation.yard_line}"
            return prediction

        # Check for two-minute drill scenarios
        if situation.quarter == 4 and situation.time_remaining:
            try:
                minutes, seconds = map(int, situation.time_remaining.split(":"))
                total_seconds = minutes * 60 + seconds

                if total_seconds <= 120:  # Two-minute warning
                    # More aggressive play calling expected - don't assume anything
                    prediction.predicted_down = None
                    prediction.predicted_distance = None
                    prediction.confidence = 0.2  # Lower confidence due to unpredictability
                    prediction.reasoning = "Two-minute drill: High variance, insufficient context"
                    return prediction
            except:
                pass

        # Check for overtime (5th quarter)
        if situation.quarter and situation.quarter >= 5:
            # Overtime rules - don't assume, need more context
            prediction.predicted_down = None
            prediction.predicted_distance = None
            prediction.confidence = 0.3
            prediction.reasoning = "Overtime: Insufficient context for prediction"
            return prediction

        return None

    def _detect_penalty_situation(self, ocr_text: str, region_color_analysis: dict = None) -> bool:
        """
        Detect if current situation is a penalty based on FLAG text and yellow coloring.

        Args:
            ocr_text: Raw OCR text from down_distance_area
            region_color_analysis: Color analysis of the region (yellow = penalty)

        Returns:
            True if penalty detected
        """
        # Check for FLAG text in OCR
        if ocr_text and "FLAG" in ocr_text.upper():
            return True

        # Check for yellow coloring (penalty indicator)
        if region_color_analysis:
            # Look for dominant yellow color in the region
            yellow_percentage = region_color_analysis.get("yellow_percentage", 0)
            if yellow_percentage > 0.3:  # 30% yellow indicates penalty
                return True

        # Check for other penalty indicators
        penalty_keywords = ["PENALTY", "FLAG", "HOLDING", "OFFSIDES", "FALSE START"]
        if ocr_text:
            for keyword in penalty_keywords:
                if keyword in ocr_text.upper():
                    return True

        return False

    def _validate_previous_play_with_current(
        self, current_situation: GameSituation
    ) -> Optional[str]:
        """
        Use current play to validate if previous play detection was correct.

        Args:
            current_situation: Current down/distance situation

        Returns:
            Validation message if correction needed, None otherwise
        """
        if len(self.game_history) < 1:
            return None

        prev_situation = self.game_history[-1]

        # Skip if either situation is incomplete
        if (
            not prev_situation.down
            or not prev_situation.distance
            or not current_situation.down
            or not current_situation.distance
        ):
            return None

        # Analyze the progression from previous to current
        expected_progressions = self._get_expected_progressions(prev_situation)

        current_actual = (current_situation.down, current_situation.distance)

        # Check if current situation matches any expected progression
        for progression in expected_progressions:
            if current_actual == progression["result"]:
                # Progression matches - previous detection was likely correct
                return None

        # Current situation doesn't match expected progressions
        # This suggests previous detection might have been wrong

        # Try to reverse-engineer what the previous situation should have been
        possible_previous = self._reverse_engineer_previous_situation(current_situation)

        if possible_previous:
            prev_actual = (prev_situation.down, prev_situation.distance)
            if prev_actual != possible_previous:
                return (
                    f"Previous {prev_actual[0]}&{prev_actual[1]} likely wrong, "
                    f"should have been {possible_previous[0]}&{possible_previous[1]} "
                    f"based on current {current_actual[0]}&{current_actual[1]}"
                )

        return None

    def _get_expected_progressions(self, situation: GameSituation) -> List[Dict]:
        """
        Get all possible next down/distance combinations from current situation.

        Returns:
            List of possible progressions with their likelihood
        """
        progressions = []
        down, distance = situation.down, situation.distance

        # Incomplete pass - down advances, distance stays same
        progressions.append(
            {
                "result": (min(4, down + 1), distance),
                "outcome": "incomplete_pass",
                "likelihood": 0.3,
            }
        )

        # Complete pass/rush for various yardages
        for yards_gained in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]:
            new_distance = max(0, distance - yards_gained)

            if new_distance <= 0:
                # First down achieved
                progressions.append(
                    {
                        "result": (1, 10),
                        "outcome": f"first_down_{yards_gained}y",
                        "likelihood": 0.4 if yards_gained >= distance else 0.2,
                    }
                )
            else:
                # Didn't get first down
                progressions.append(
                    {
                        "result": (min(4, down + 1), new_distance),
                        "outcome": f"gain_{yards_gained}y",
                        "likelihood": 0.3,
                    }
                )

        # Sack (loss of yards)
        for yards_lost in [1, 2, 3, 5, 7, 10]:
            new_distance = distance + yards_lost
            progressions.append(
                {
                    "result": (min(4, down + 1), new_distance),
                    "outcome": f"sack_{yards_lost}y",
                    "likelihood": 0.1,
                }
            )

        # Turnover (next possession starts fresh)
        progressions.append({"result": (1, 10), "outcome": "turnover", "likelihood": 0.1})

        # Punt (next possession starts fresh)
        if down == 4:
            progressions.append({"result": (1, 10), "outcome": "punt", "likelihood": 0.6})

        return progressions

    def _reverse_engineer_previous_situation(
        self, current_situation: GameSituation
    ) -> Optional[Tuple[int, int]]:
        """
        Given current situation, determine what previous situation would make sense.

        Args:
            current_situation: Current down/distance

        Returns:
            Tuple of (previous_down, previous_distance) that would lead to current
        """
        current_down = current_situation.down
        current_distance = current_situation.distance

        # If current is 1st & 10, previous could have been any successful first down
        if current_down == 1 and current_distance == 10:
            # Could have been any down that achieved first down
            # Most likely: 3rd down conversion
            return (3, 8)  # Common 3rd down situation

        # If current down > 1, previous was likely (current_down - 1)
        if current_down > 1:
            prev_down = current_down - 1

            # If distance is same, likely incomplete pass
            if prev_down >= 1:
                return (prev_down, current_distance)

        # For other situations, try common scenarios
        if current_down == 2:
            # 2nd down often follows 1st & 10 with some gain
            yards_gained = 10 - current_distance
            if 0 <= yards_gained <= 10:
                return (1, 10)

        if current_down == 3:
            # 3rd down often follows 2nd down
            return (2, current_distance)

        if current_down == 4:
            # 4th down follows 3rd down
            return (3, current_distance)

        return None

    def _analyze_drive_context(self) -> dict:
        """
        Analyze the last 5-10 plays to understand drive patterns and momentum.

        Returns:
            Dictionary with drive context analysis
        """
        context = {
            "consecutive_first_downs": 0,
            "recent_third_downs": 0,
            "in_red_zone": False,
            "red_zone_plays": 0,
            "drive_length": 0,
            "possession_consistency": True,
            "field_position_trend": "neutral",  # "advancing", "stalling", "neutral"
        }

        if len(self.game_history) < 2:
            return context

        # Analyze last 10 plays (or all available)
        analysis_window = min(10, len(self.game_history))
        recent_plays = self.game_history[-analysis_window:]

        # Track consecutive first downs from most recent
        for i in range(len(recent_plays) - 1, -1, -1):
            play = recent_plays[i]
            if play.down == 1:
                context["consecutive_first_downs"] += 1
            else:
                break

        # Count recent third downs (last 5 plays)
        recent_window = min(5, len(recent_plays))
        for play in recent_plays[-recent_window:]:
            if play.down == 3:
                context["recent_third_downs"] += 1

        # Check red zone context
        current_situation = self.game_history[-1] if self.game_history else None
        if (
            current_situation
            and current_situation.yard_line
            and current_situation.territory == "opponent"
            and current_situation.yard_line <= 20
        ):
            context["in_red_zone"] = True

            # Count plays in red zone
            for play in recent_plays:
                if play.yard_line and play.territory == "opponent" and play.yard_line <= 20:
                    context["red_zone_plays"] += 1

        # Check possession consistency
        if len(recent_plays) >= 2:
            first_possession = recent_plays[0].possession_team
            for play in recent_plays[1:]:
                if play.possession_team != first_possession:
                    context["possession_consistency"] = False
                    break

        # Analyze field position trend
        if len(recent_plays) >= 3:
            yard_lines = []
            for play in recent_plays[-3:]:
                if play.yard_line and play.territory:
                    # Convert to field position (0-100 scale)
                    if play.territory == "own":
                        field_pos = play.yard_line
                    else:  # opponent territory
                        field_pos = 100 - play.yard_line
                    yard_lines.append(field_pos)

            if len(yard_lines) >= 2:
                trend = yard_lines[-1] - yard_lines[0]
                if trend > 10:
                    context["field_position_trend"] = "advancing"
                elif trend < -10:
                    context["field_position_trend"] = "stalling"

        # Calculate drive length (plays with same possession)
        if context["possession_consistency"]:
            context["drive_length"] = len(recent_plays)

        return context

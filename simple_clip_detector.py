import copy
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class OCRValidationError(Exception):
    """Raised when OCR data fails validation checks."""

    def __init__(self, raw_data: dict, reason: str):
        self.raw_data = raw_data
        self.reason = reason
        super().__init__(f"OCR validation failed: {reason}")


@dataclass
class PlayState:
    frame: int
    down: Optional[int]
    distance: Optional[int]
    timestamp: float
    raw_game_state: dict[str, Any]
    validation_passed: bool = False


@dataclass
class ClipInfo:
    start_frame: int
    trigger_frame: int
    end_frame: Optional[int]
    play_down: int
    play_distance: int
    preserved_state: dict[str, Any]
    created_at: float
    status: str = "pending"
    validation_confidence: float = 0.0


class SimpleClipDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.play_history: deque = deque(maxlen=300)
        self.active_clips: list[ClipInfo] = []
        self.last_validated_down: Optional[int] = None

        # Timing configuration
        self.pre_snap_buffer_seconds = 3.5
        self.post_play_buffer_seconds = 3.0  # Smart buffer for play completion
        self.max_play_duration_seconds = 12.0
        self.ocr_failure_buffer_seconds = 5.0  # Extra buffer when OCR fails

        # Validation thresholds
        self.min_confidence_threshold = 0.7
        self.min_data_quality_threshold = 0.6

    def process_frame(self, frame_number: int, game_state: dict[str, Any]) -> Optional[ClipInfo]:
        """Process a frame with strict data validation and smart clip management."""
        current_time = time.time()

        try:
            # Step 1: Validate incoming data (ACCURACY OVER COMPLETENESS)
            validated_state = self._validate_game_state(frame_number, game_state, current_time)

            # Step 2: Check for down changes using validated data only
            new_clip = self._detect_new_play_with_validation(validated_state)

            # Step 3: Finalize previous clips with smart buffering
            if new_clip:
                self._finalize_previous_clips_smart(frame_number)

            return new_clip

        except OCRValidationError as e:
            # Handle OCR failures gracefully - don't create new clips but don't cut existing ones
            print(f"⚠️  Frame {frame_number}: {e.reason}")
            self._handle_ocr_failure(frame_number, game_state, current_time)
            return None

    def _validate_game_state(
        self, frame_number: int, game_state: dict[str, Any], timestamp: float
    ) -> PlayState:
        """Validate game state data with strict quality checks."""
        down = game_state.get("down")
        distance = game_state.get("distance")
        confidence = game_state.get("confidence", 0.0)

        # Create play state object
        play_state = PlayState(
            frame=frame_number,
            down=down,
            distance=distance,
            timestamp=timestamp,
            raw_game_state=copy.deepcopy(game_state),
            validation_passed=False,
        )

        # Validation checks
        validation_issues = []

        # Check 1: Basic data presence
        if down is None or distance is None:
            validation_issues.append("Missing down or distance data")

        # Check 2: Confidence threshold
        if confidence < self.min_confidence_threshold:
            validation_issues.append(
                f"Confidence {confidence} below threshold {self.min_confidence_threshold}"
            )

        # Check 3: Football logic validation
        if down is not None:
            if not (1 <= down <= 4):
                validation_issues.append(f"Invalid down value: {down}")

        if distance is not None:
            if not (0 <= distance <= 99):
                validation_issues.append(f"Invalid distance value: {distance}")

        # Check 4: Data format validation (if raw text available)
        raw_text = game_state.get("raw_ocr_text", "")
        if raw_text and not self._validate_ocr_format(raw_text):
            validation_issues.append(f"Invalid OCR format: '{raw_text}'")

        # If validation fails, raise error
        if validation_issues:
            raise OCRValidationError(game_state, "; ".join(validation_issues))

        # Mark as validated
        play_state.validation_passed = True
        self.play_history.append(play_state)

        return play_state

    def _validate_ocr_format(self, raw_text: str) -> bool:
        """Validate OCR text format matches expected football patterns."""
        if not raw_text:
            return False

        # Common valid patterns
        patterns = [
            r"^\d+(ST|ND|RD|TH)\s*&\s*\d+$",  # "1ST & 10"
            r"^\d+\s*&\s*\d+$",  # "1 & 10"
            r"^GOAL$",  # "GOAL"
            r"^\d+(ST|ND|RD|TH)\s*&\s*GOAL$",  # "1ST & GOAL"
        ]

        return any(re.match(pattern, raw_text.upper().strip()) for pattern in patterns)

    def _detect_new_play_with_validation(self, validated_state: PlayState) -> Optional[ClipInfo]:
        """Detect new plays using only validated data."""
        current_down = validated_state.down

        if current_down is None:
            return None

        # First validated play detection
        if self.last_validated_down is None:
            print(f"🏈 FIRST VALIDATED PLAY: Down {current_down}")
            self.last_validated_down = current_down
            return self._create_clip_info_validated(validated_state)

        # Down change detection with validated data
        if self.last_validated_down != current_down:
            print(
                f"🏈 VALIDATED PLAY CHANGE: {self.last_validated_down} → {current_down} at frame {validated_state.frame}"
            )
            self.last_validated_down = current_down
            return self._create_clip_info_validated(validated_state)

        return None

    def _create_clip_info_validated(self, play_state: PlayState) -> ClipInfo:
        """Create clip info with validation metadata."""
        frame_number = play_state.frame
        pre_snap_frames = int(self.fps * self.pre_snap_buffer_seconds)
        start_frame = max(0, frame_number - pre_snap_frames)

        clip_info = ClipInfo(
            start_frame=start_frame,
            trigger_frame=frame_number,
            end_frame=None,  # Will be set during finalization
            play_down=play_state.down,
            play_distance=play_state.distance,
            preserved_state=copy.deepcopy(play_state.raw_game_state),
            created_at=play_state.timestamp,
            status="pending",
            validation_confidence=play_state.raw_game_state.get("confidence", 0.0),
        )

        self.active_clips.append(clip_info)
        print(
            f"🎬 VALIDATED CLIP CREATED: {clip_info.play_down} & {clip_info.play_distance} (confidence: {clip_info.validation_confidence:.2f})"
        )
        return clip_info

    def _finalize_previous_clips_smart(self, new_play_frame: int):
        """Finalize clips with smart buffering to ensure complete plays."""
        for clip in self.active_clips:
            if clip.status == "pending":
                # Calculate smart end frame with post-play buffer
                post_play_frames = int(self.fps * self.post_play_buffer_seconds)
                smart_end_frame = new_play_frame + post_play_frames

                # Ensure minimum play duration (avoid micro-clips)
                min_duration_frames = int(self.fps * 2.0)  # 2 second minimum
                min_end_frame = clip.trigger_frame + min_duration_frames

                # Use the later of the two to ensure complete plays
                clip.end_frame = max(smart_end_frame, min_end_frame)
                clip.status = "finalized"

                duration = (clip.end_frame - clip.start_frame) / self.fps
                print(f"🏁 SMART FINALIZATION: {clip.play_down} & {clip.play_distance}")
                print(
                    f"   - Duration: {duration:.1f}s (start: {clip.start_frame}, end: {clip.end_frame})"
                )

    def _handle_ocr_failure(self, frame_number: int, game_state: dict[str, Any], timestamp: float):
        """Handle OCR failures without cutting existing clips short."""
        # Create unvalidated play state for history tracking
        play_state = PlayState(
            frame=frame_number,
            down=None,
            distance=None,
            timestamp=timestamp,
            raw_game_state=copy.deepcopy(game_state),
            validation_passed=False,
        )
        self.play_history.append(play_state)

        # Extend existing clips with failure buffer (don't cut them short)
        for clip in self.active_clips:
            if clip.status == "pending":
                # Add extra buffer time when OCR fails to ensure we don't cut mid-play
                failure_buffer_frames = int(self.fps * self.ocr_failure_buffer_seconds)
                provisional_end = frame_number + failure_buffer_frames

                # Only extend if this would make the clip longer
                if clip.end_frame is None or provisional_end > clip.end_frame:
                    clip.end_frame = provisional_end
                    print(
                        f"🔄 Extended clip due to OCR failure: {clip.play_down} & {clip.play_distance}"
                    )

    def finalize_remaining_clips(self, final_frame: int):
        """Finalize any remaining clips when video processing ends."""
        for clip in self.active_clips:
            if clip.status == "pending":
                # Use smart buffering for final clips too
                post_play_frames = int(self.fps * self.post_play_buffer_seconds)
                clip.end_frame = final_frame + post_play_frames
                clip.status = "finalized"

                duration = (clip.end_frame - clip.start_frame) / self.fps
                print(
                    f"🏁 FINAL CLIP FINALIZED: {clip.play_down} & {clip.play_distance} (duration: {duration:.1f}s)"
                )

    def get_finalized_clips(self) -> list[ClipInfo]:
        """Get all finalized clips."""
        return [clip for clip in self.active_clips if clip.status == "finalized"]

    def get_validation_stats(self) -> dict:
        """Get validation statistics for monitoring."""
        total_frames = len(self.play_history)
        validated_frames = sum(1 for state in self.play_history if state.validation_passed)

        return {
            "total_frames_processed": total_frames,
            "validated_frames": validated_frames,
            "validation_rate": validated_frames / total_frames if total_frames > 0 else 0.0,
            "active_clips": len([c for c in self.active_clips if c.status == "pending"]),
            "finalized_clips": len([c for c in self.active_clips if c.status == "finalized"]),
        }

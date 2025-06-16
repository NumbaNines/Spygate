import copy
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class PlayState:
    frame: int
    down: Optional[int]
    distance: Optional[int]
    timestamp: float
    raw_game_state: Dict[str, Any]

@dataclass
class ClipInfo:
    start_frame: int
    trigger_frame: int
    end_frame: Optional[int]
    play_down: int
    play_distance: int
    preserved_state: Dict[str, Any]
    created_at: float
    status: str = "pending"

class SimpleClipDetector:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.play_history: deque = deque(maxlen=300)
        self.active_clips: List[ClipInfo] = []
        self.last_down: Optional[int] = None
        self.pre_snap_buffer_seconds = 3.5
        self.max_play_duration_seconds = 12.0

    def process_frame(self, frame_number: int, game_state: Dict[str, Any]) -> Optional[ClipInfo]:
        current_time = time.time()
        down = game_state.get('down')
        distance = game_state.get('distance')

        play_state = PlayState(
            frame=frame_number,
            down=down,
            distance=distance,
            timestamp=current_time,
            raw_game_state=copy.deepcopy(game_state)
        )
        self.play_history.append(play_state)

        new_clip = self._detect_new_play(play_state)
        if new_clip:
            self._finalize_previous_clips(frame_number)

        return new_clip

    def _detect_new_play(self, current_state: PlayState) -> Optional[ClipInfo]:
        current_down = current_state.down

        if current_down is None:
            return None

        # First play detection
        if self.last_down is None:
            print(f"🏈 FIRST PLAY DETECTED: Down {current_down}")
            self.last_down = current_down
            return self._create_clip_info(current_state)

        # Down change detection
        if self.last_down != current_down:
            print(f"🏈 NEW PLAY DETECTED: {self.last_down} → {current_down} at frame {current_state.frame}")
            self.last_down = current_down
            return self._create_clip_info(current_state)

        return None

    def _create_clip_info(self, play_state: PlayState) -> ClipInfo:
        frame_number = play_state.frame
        pre_snap_frames = int(self.fps * self.pre_snap_buffer_seconds)
        start_frame = max(0, frame_number - pre_snap_frames)

        max_duration_frames = int(self.fps * self.max_play_duration_seconds)
        provisional_end_frame = frame_number + max_duration_frames

        clip_info = ClipInfo(
            start_frame=start_frame,
            trigger_frame=frame_number,
            end_frame=provisional_end_frame,
            play_down=play_state.down,
            play_distance=play_state.distance,
            preserved_state=copy.deepcopy(play_state.raw_game_state),
            created_at=play_state.timestamp,
            status="pending"
        )

        self.active_clips.append(clip_info)
        print(f"🎬 CLIP CREATED: {clip_info.play_down} & {clip_info.play_distance}")
        return clip_info

    def _finalize_previous_clips(self, new_play_frame: int):
        for clip in self.active_clips:
            if clip.status == "pending":
                clip.end_frame = new_play_frame - 1
                clip.status = "finalized"
                print(f"🏁 CLIP FINALIZED: {clip.play_down} & {clip.play_distance}")

    def get_finalized_clips(self) -> List[ClipInfo]:
        return [clip for clip in self.active_clips if clip.status == "finalized"] 
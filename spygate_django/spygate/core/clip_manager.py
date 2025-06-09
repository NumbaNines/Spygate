"""
SpygateAI Clip Management System - Full PRD Implementation
"""

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GameVersion(Enum):
    """Supported EA Football games for cross-game intelligence."""

    MADDEN_25 = "madden_25"
    CFB_25 = "cfb_25"
    MADDEN_26 = "madden_26"


class SituationType(Enum):
    """Core football situations for automatic categorization."""

    THIRD_DOWN_LONG = "3rd_long"
    THIRD_DOWN_SHORT = "3rd_short"
    RED_ZONE_OFFENSE = "red_zone_offense"
    RED_ZONE_DEFENSE = "red_zone_defense"
    GOAL_LINE = "goal_line"
    TWO_MINUTE_DRILL = "two_minute_drill"
    FOURTH_DOWN = "fourth_down"
    TURNOVER = "turnover"
    BIG_PLAY = "big_play"


@dataclass
class GameSituation:
    """Represents detected game situation from HUD analysis."""

    down: int
    distance: int
    field_position: str
    time_remaining: str
    score_differential: int
    quarter: int
    game_version: GameVersion
    confidence: float


@dataclass
class ClipBoundary:
    """Represents intelligent clip boundaries."""

    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    pre_snap_buffer: float = 3.0
    post_play_buffer: float = 2.0
    confidence: float = 0.0
    detection_method: str = "auto"


@dataclass
class AnalyzedClip:
    """Comprehensive clip analysis results."""

    clip_id: str
    source_file: str
    boundaries: ClipBoundary
    situation: GameSituation
    detected_formations: list[str]
    suggested_categories: list[SituationType]
    cross_game_mapping: dict[GameVersion, str]
    auto_tags: list[str]
    user_tags: list[str]
    opponent_context: Optional[str] = None
    success_result: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class IntelligentClipManager:
    """Core clip management implementing the full PRD vision."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.clips_db_path = self.project_root / "data" / "clips.db"
        self.gameplans_path = self.project_root / "gameplans"
        self.solutions_path = self.project_root / "solutions"

        self._setup_directory_structure()
        self._initialize_database()

    def _setup_directory_structure(self):
        """Create the PRD-defined gameplan directory structure."""
        directories = [
            self.solutions_path / "Opponents",
            self.solutions_path / "Situations" / "3rd_Long",
            self.solutions_path / "Situations" / "Red_Zone",
            self.solutions_path / "Situations" / "Two_Minute_Drill",
            self.solutions_path / "Cross_Game" / "Universal_Concepts",
            self.project_root / "clips" / "raw",
            self.project_root / "clips" / "analyzed",
            self.project_root / "clips" / "exported",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize SQLite database for clip tracking."""
        self.clips_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.clips_db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    game_version TEXT NOT NULL,
                    situation_data TEXT,
                    boundaries_data TEXT,
                    categories TEXT,
                    opponent_context TEXT,
                    success_result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS gameplans (
                    gameplan_id TEXT PRIMARY KEY,
                    opponent_name TEXT,
                    game_version TEXT,
                    clips TEXT,
                    strategies TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_clips_opponent ON clips(opponent_context);
                CREATE INDEX IF NOT EXISTS idx_clips_game_version ON clips(game_version);
            """
            )

    def detect_intelligent_clip_boundaries(
        self,
        user_selected_frame: int,
        hud_analysis_timeline: list[dict[str, Any]],
        fps: float = 30.0,
    ) -> ClipBoundary:
        """PRD Feature: Auto-extend clip boundaries from user selection."""

        play_start_frame = self._find_pre_snap_boundary(user_selected_frame, hud_analysis_timeline)
        play_end_frame = self._find_post_play_boundary(user_selected_frame, hud_analysis_timeline)

        buffer_frames_pre = int(3.0 * fps)
        buffer_frames_post = int(2.0 * fps)

        final_start = max(0, play_start_frame - buffer_frames_pre)
        final_end = play_end_frame + buffer_frames_post

        return ClipBoundary(
            start_frame=final_start,
            end_frame=final_end,
            start_timestamp=final_start / fps,
            end_timestamp=final_end / fps,
            confidence=0.85,
            detection_method="intelligent_auto_extend",
        )

    def _find_pre_snap_boundary(self, frame: int, timeline: list[dict]) -> int:
        """Find the last pre-snap moment before the selected frame."""
        for i in range(len(timeline) - 1, -1, -1):
            frame_data = timeline[i]
            if frame_data.get("frame_number", 0) <= frame:
                game_state = frame_data.get("game_state")
                if game_state == "pre_snap":
                    return self._find_state_transition_start(i, timeline, "pre_snap")
        return max(0, frame - 150)

    def _find_post_play_boundary(self, frame: int, timeline: list[dict]) -> int:
        """Find the post-play moment after the selected frame."""
        for i, frame_data in enumerate(timeline):
            if frame_data.get("frame_number", 0) >= frame:
                game_state = frame_data.get("game_state")
                if game_state == "post_play":
                    return frame_data.get("frame_number", frame) + 30
        return frame + 240

    def _find_state_transition_start(
        self, index: int, timeline: list[dict], target_state: str
    ) -> int:
        """Find where a game state transition begins."""
        for i in range(index, -1, -1):
            if timeline[i].get("game_state") != target_state:
                return timeline[min(i + 1, len(timeline) - 1)].get("frame_number", 0)
        return timeline[0].get("frame_number", 0)

    def analyze_clip_situation(self, hud_data: dict[str, Any]) -> GameSituation:
        """Analyze HUD data to determine game situation."""
        down = self._extract_down(hud_data.get("down_distance", ""))
        distance = self._extract_distance(hud_data.get("down_distance", ""))
        field_pos = self._construct_field_position(
            hud_data.get("yards_to_goal"), hud_data.get("territory_indicator")
        )

        game_version = GameVersion(hud_data.get("game_version", "madden_25"))

        return GameSituation(
            down=down,
            distance=distance,
            field_position=field_pos,
            time_remaining=hud_data.get("game_clock", "unknown"),
            score_differential=self._calculate_score_diff(hud_data),
            quarter=self._extract_quarter(hud_data.get("game_clock", "")),
            game_version=game_version,
            confidence=0.9,
        )

    def generate_intelligent_categorization(
        self, situation: GameSituation
    ) -> tuple[list[SituationType], dict[GameVersion, str]]:
        """PRD Feature: Auto-suggest categories based on situation analysis."""
        categories = []
        cross_game_mapping = {}

        if situation.down == 3 and situation.distance >= 7:
            categories.append(SituationType.THIRD_DOWN_LONG)
            cross_game_mapping[GameVersion.MADDEN_25] = "3rd_Long_Passing_Concepts"
            cross_game_mapping[GameVersion.CFB_25] = "3rd_Long_Spread_Concepts"

        elif situation.down == 3 and situation.distance <= 3:
            categories.append(SituationType.THIRD_DOWN_SHORT)
            cross_game_mapping[GameVersion.MADDEN_25] = "3rd_Short_Power_Concepts"
            cross_game_mapping[GameVersion.CFB_25] = "3rd_Short_Option_Concepts"

        elif (
            "OPP" in situation.field_position
            and self._extract_yard_number(situation.field_position) <= 20
        ):
            categories.append(SituationType.RED_ZONE_OFFENSE)
            cross_game_mapping[GameVersion.MADDEN_25] = "RedZone_Efficiency_Plays"
            cross_game_mapping[GameVersion.CFB_25] = "RedZone_RPO_Concepts"

        elif situation.down == 4:
            categories.append(SituationType.FOURTH_DOWN)

        if "2:" in situation.time_remaining or "1:" in situation.time_remaining:
            categories.append(SituationType.TWO_MINUTE_DRILL)

        return categories, cross_game_mapping

    def create_analyzed_clip(
        self,
        source_file: str,
        user_selected_frame: int,
        hud_timeline: list[dict[str, Any]],
        opponent_name: Optional[str] = None,
    ) -> AnalyzedClip:
        """PRD Workflow: Complete clip analysis and categorization."""
        clip_id = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(source_file) % 10000}"

        boundaries = self.detect_intelligent_clip_boundaries(user_selected_frame, hud_timeline)
        frame_hud_data = self._get_hud_data_at_frame(user_selected_frame, hud_timeline)
        situation = self.analyze_clip_situation(frame_hud_data)
        suggested_categories, cross_game_mapping = self.generate_intelligent_categorization(
            situation
        )

        analyzed_clip = AnalyzedClip(
            clip_id=clip_id,
            source_file=source_file,
            boundaries=boundaries,
            situation=situation,
            detected_formations=[],
            suggested_categories=suggested_categories,
            cross_game_mapping=cross_game_mapping,
            auto_tags=self._generate_auto_tags(situation),
            user_tags=[],
            opponent_context=opponent_name,
        )

        self._save_clip_to_database(analyzed_clip)
        return analyzed_clip

    def export_to_gameplan(
        self, clip: AnalyzedClip, category: SituationType, custom_tags: list[str] = None
    ) -> str:
        """PRD Feature: Export clip to appropriate gameplan folder structure."""
        if clip.opponent_context:
            gameplan_path = self.solutions_path / "Opponents" / clip.opponent_context
            category_path = gameplan_path / category.value
        else:
            gameplan_path = self.solutions_path / "Situations" / category.value
            category_path = gameplan_path / clip.situation.game_version.value

        category_path.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{clip.clip_id}_{category.value}_{clip.situation.down}_{clip.situation.distance}.mp4"
        )
        export_path = category_path / filename

        metadata = {
            "clip_id": clip.clip_id,
            "situation": asdict(clip.situation),
            "boundaries": asdict(clip.boundaries),
            "categories": [cat.value for cat in clip.suggested_categories],
            "cross_game_mapping": {k.value: v for k, v in clip.cross_game_mapping.items()},
            "auto_tags": clip.auto_tags,
            "user_tags": custom_tags or clip.user_tags,
            "exported_at": datetime.now().isoformat(),
        }

        metadata_path = category_path / f"{clip.clip_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(export_path)

    # Helper methods
    def _extract_down(self, down_distance: str) -> int:
        """Extract down number from down_distance string."""
        if "1st" in down_distance:
            return 1
        elif "2nd" in down_distance:
            return 2
        elif "3rd" in down_distance:
            return 3
        elif "4th" in down_distance:
            return 4
        return 1

    def _extract_distance(self, down_distance: str) -> int:
        """Extract distance from down_distance string."""
        import re

        match = re.search(r"& (\d+)", down_distance)
        if match:
            return int(match.group(1))
        elif "Goal" in down_distance:
            return 0
        return 10

    def _construct_field_position(self, yards_to_goal: str, territory_indicator: str) -> str:
        """Construct field position from HUD elements."""
        try:
            yards = int(str(yards_to_goal))
            if territory_indicator == "▲":
                return f"OPP {yards}"
            elif territory_indicator == "▼":
                return f"OWN {yards}"
        except:
            pass
        return "unknown"

    def _calculate_score_diff(self, hud_data: dict) -> int:
        """Calculate score differential from HUD data."""
        try:
            home_score = int(hud_data.get("home_score", 0))
            away_score = int(hud_data.get("away_score", 0))
            return home_score - away_score
        except:
            return 0

    def _extract_quarter(self, game_clock: str) -> int:
        """Extract quarter from game clock display."""
        if "1st" in game_clock or "1Q" in game_clock:
            return 1
        elif "2nd" in game_clock or "2Q" in game_clock:
            return 2
        elif "3rd" in game_clock or "3Q" in game_clock:
            return 3
        elif "4th" in game_clock or "4Q" in game_clock:
            return 4
        return 1

    def _extract_yard_number(self, field_position: str) -> int:
        """Extract yard number from field position string."""
        import re

        match = re.search(r"(\d+)", field_position)
        return int(match.group(1)) if match else 50

    def _generate_auto_tags(self, situation: GameSituation) -> list[str]:
        """Generate automatic tags based on situation."""
        tags = []
        if situation.down == 3:
            tags.append("third_down")
        if situation.distance <= 3:
            tags.append("short_yardage")
        if (
            "OPP" in situation.field_position
            and self._extract_yard_number(situation.field_position) <= 20
        ):
            tags.append("red_zone")
        if situation.quarter >= 4:
            tags.append("fourth_quarter")
        tags.append(f"game_{situation.game_version.value}")
        return tags

    def _get_hud_data_at_frame(self, frame: int, timeline: list[dict]) -> dict[str, Any]:
        """Get HUD data for specific frame from timeline."""
        for frame_data in timeline:
            if frame_data.get("frame_number", 0) == frame:
                return frame_data.get("hud_data", {})
        return timeline[-1].get("hud_data", {}) if timeline else {}

    def _save_clip_to_database(self, clip: AnalyzedClip):
        """Save analyzed clip to database."""
        with sqlite3.connect(self.clips_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO clips
                (clip_id, source_file, game_version, situation_data, boundaries_data,
                 categories, opponent_context, success_result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    clip.clip_id,
                    clip.source_file,
                    clip.situation.game_version.value,
                    json.dumps(asdict(clip.situation)),
                    json.dumps(asdict(clip.boundaries)),
                    json.dumps([cat.value for cat in clip.suggested_categories]),
                    clip.opponent_context,
                    clip.success_result,
                    clip.created_at.isoformat(),
                ),
            )

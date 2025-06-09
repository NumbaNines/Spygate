"""SpygateAI Tournament Preparation System for MCS and tournament play."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TournamentType(Enum):
    MCS_QUALIFIER = "mcs_qualifier"
    MCS_CHAMPIONSHIP = "mcs_championship"
    PLAYERS_LOUNGE = "players_lounge"
    WEEKLY_TOURNAMENT = "weekly_tournament"


class OpponentAnalysisStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class OpponentProfile:
    username: str
    game_version: str
    tournament_type: str
    analysis_status: str
    formation_usage: dict[str, float]
    situational_tendencies: dict[str, dict[str, Any]]
    success_patterns: dict[str, float]
    clips_analyzed: int
    last_analysis_date: datetime
    confidence_score: float
    tournament_name: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CounterStrategy:
    counter_id: str
    target_formation: str
    target_situation: str
    recommended_counter: str
    success_rate: float
    usage_difficulty: str
    notes: list[str]
    clip_references: list[str]
    cross_game_compatible: bool


@dataclass
class TournamentGameplan:
    gameplan_id: str
    opponent_username: str
    tournament_type: str
    game_version: str
    opponent_profile: OpponentProfile
    counter_strategies: list[CounterStrategy]
    key_situations: dict[str, Any]
    practice_priorities: list[str]
    preparation_status: str
    estimated_prep_time: int
    completion_percentage: float
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()


class TournamentPrepEngine:
    """Core engine for tournament preparation."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.tournament_db_path = self.project_root / "data" / "tournament_prep.db"
        self.gameplans_path = self.project_root / "tournament_gameplans"

        self._setup_directory_structure()
        self._initialize_database()

    def _setup_directory_structure(self):
        """Create tournament preparation directory structure."""
        directories = [
            self.gameplans_path / "mcs_qualifiers",
            self.gameplans_path / "mcs_championship",
            self.gameplans_path / "players_lounge",
            self.gameplans_path / "weekly_tournaments",
            self.gameplans_path / "archived",
            self.project_root / "practice_footage",
            self.project_root / "opponent_footage",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize tournament preparation database."""
        self.tournament_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.tournament_db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS opponent_profiles (
                    username TEXT,
                    game_version TEXT,
                    tournament_type TEXT,
                    profile_data TEXT,
                    analysis_status TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (username, game_version, tournament_type)
                );

                CREATE TABLE IF NOT EXISTS tournament_gameplans (
                    gameplan_id TEXT PRIMARY KEY,
                    opponent_username TEXT NOT NULL,
                    tournament_type TEXT,
                    game_version TEXT,
                    gameplan_data TEXT,
                    preparation_status TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS counter_strategies (
                    counter_id TEXT PRIMARY KEY,
                    opponent_username TEXT,
                    target_formation TEXT,
                    target_situation TEXT,
                    counter_data TEXT,
                    effectiveness_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

    def analyze_opponent_from_footage(
        self,
        opponent_username: str,
        footage_files: list[str],
        tournament_type: str,
        game_version: str = "madden_25",
    ) -> OpponentProfile:
        """Analyze 3-5 gameplay files from opponent to generate scouting report."""

        print(f"🔍 Analyzing opponent: {opponent_username}")
        print(f"📁 Processing {len(footage_files)} footage files...")

        formation_usage = {}
        situational_tendencies = {}
        success_patterns = {}
        clips_analyzed = 0

        # Process each footage file
        for footage_file in footage_files:
            print(f"⚙️ Processing: {Path(footage_file).name}")

            # Mock analysis for demonstration
            clips_analyzed += 5  # Simulate 5 clips per file

            # Mock formation data
            mock_formations = ["Shotgun Trips TE", "I-Form Pro", "Singleback Ace"]
            for formation in mock_formations:
                formation_usage[formation] = formation_usage.get(formation, 0) + 1

        # Convert to percentages
        total_formations = sum(formation_usage.values())
        if total_formations > 0:
            formation_usage = {k: (v / total_formations) * 100 for k, v in formation_usage.items()}

        confidence_score = min(1.0, clips_analyzed / 20)

        if clips_analyzed >= 10:
            analysis_status = "completed"
        elif clips_analyzed >= 3:
            analysis_status = "in_progress"
        else:
            analysis_status = "insufficient_data"

        opponent_profile = OpponentProfile(
            username=opponent_username,
            game_version=game_version,
            tournament_type=tournament_type,
            analysis_status=analysis_status,
            formation_usage=formation_usage,
            situational_tendencies=situational_tendencies,
            success_patterns=success_patterns,
            clips_analyzed=clips_analyzed,
            last_analysis_date=datetime.now(),
            confidence_score=confidence_score,
        )

        self._save_opponent_profile(opponent_profile)

        print(f"✅ Analysis complete: {clips_analyzed} clips analyzed")
        print(f"🎯 Confidence score: {confidence_score:.2f}")

        return opponent_profile

    def generate_tournament_gameplan(
        self, opponent_profile: OpponentProfile, tournament_name: Optional[str] = None
    ) -> TournamentGameplan:
        """Generate complete tournament gameplan with counter-strategies."""

        gameplan_id = (
            f"gameplan_{opponent_profile.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        print(f"🎮 Generating tournament gameplan for {opponent_profile.username}")

        # Generate counter-strategies
        counter_strategies = []
        top_formations = sorted(
            opponent_profile.formation_usage.items(), key=lambda x: x[1], reverse=True
        )[:3]

        for formation, usage_pct in top_formations:
            counter = CounterStrategy(
                counter_id=f"counter_{formation}_{int(usage_pct)}",
                target_formation=formation,
                target_situation="general",
                recommended_counter=f"Recommended counter for {formation}",
                success_rate=0.75,
                usage_difficulty="medium",
                notes=[f"Used {usage_pct:.1f}% of the time"],
                clip_references=[],
                cross_game_compatible=True,
            )
            counter_strategies.append(counter)

        practice_priorities = [
            f"Counter {top_formations[0][0]}" if top_formations else "General defense",
            "3rd down situations",
            "Red zone defense",
        ]

        estimated_prep_time = 60 + len(counter_strategies) * 15

        gameplan = TournamentGameplan(
            gameplan_id=gameplan_id,
            opponent_username=opponent_profile.username,
            tournament_type=opponent_profile.tournament_type,
            game_version=opponent_profile.game_version,
            opponent_profile=opponent_profile,
            counter_strategies=counter_strategies,
            key_situations={
                "high_priority": practice_priorities[:2],
                "medium_priority": practice_priorities[2:],
                "low_priority": [],
            },
            practice_priorities=practice_priorities,
            preparation_status="generated",
            estimated_prep_time=estimated_prep_time,
            completion_percentage=0.0,
        )

        self._save_tournament_gameplan(gameplan)
        self._export_gameplan_files(gameplan)

        print(f"📋 Gameplan generated: {len(counter_strategies)} counter-strategies")
        print(f"⏱️ Estimated prep time: {estimated_prep_time} minutes")

        return gameplan

    def get_pre_match_summary(self, gameplan_id: str) -> dict[str, Any]:
        """Generate pre-match summary for tournament day."""

        with sqlite3.connect(self.tournament_db_path) as conn:
            cursor = conn.execute(
                "SELECT gameplan_data FROM tournament_gameplans WHERE gameplan_id = ?",
                (gameplan_id,),
            )
            result = cursor.fetchone()

            if not result:
                return {"error": "Gameplan not found"}

            gameplan_data = json.loads(result[0])

            summary = {
                "opponent": gameplan_data["opponent_username"],
                "tournament": gameplan_data["tournament_type"],
                "game_version": gameplan_data["game_version"],
                "generated_at": datetime.now().isoformat(),
                "critical_counters": gameplan_data.get("counter_strategies", [])[:3],
                "key_reminders": [
                    "Focus on top formation counters",
                    "Practice 3rd down defense",
                    "Stay calm under pressure",
                ],
                "confidence_level": f"{gameplan_data['opponent_profile']['confidence_score']:.0%}",
                "clips_analyzed": gameplan_data["opponent_profile"]["clips_analyzed"],
            }

            return summary

    def _save_opponent_profile(self, profile: OpponentProfile):
        """Save opponent profile to database."""
        with sqlite3.connect(self.tournament_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO opponent_profiles
                (username, game_version, tournament_type, profile_data, analysis_status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    profile.username,
                    profile.game_version,
                    profile.tournament_type,
                    json.dumps(asdict(profile), default=str),
                    profile.analysis_status,
                    profile.last_analysis_date.isoformat(),
                ),
            )

    def _save_tournament_gameplan(self, gameplan: TournamentGameplan):
        """Save tournament gameplan to database."""
        with sqlite3.connect(self.tournament_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tournament_gameplans
                (gameplan_id, opponent_username, tournament_type, game_version,
                 gameplan_data, preparation_status, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    gameplan.gameplan_id,
                    gameplan.opponent_username,
                    gameplan.tournament_type,
                    gameplan.game_version,
                    json.dumps(asdict(gameplan), default=str),
                    gameplan.preparation_status,
                    gameplan.generated_at.isoformat(),
                ),
            )

    def _export_gameplan_files(self, gameplan: TournamentGameplan):
        """Export gameplan to organized files."""

        tournament_dir = self.gameplans_path / gameplan.tournament_type
        gameplan_dir = tournament_dir / f"{gameplan.opponent_username}_{gameplan.gameplan_id}"
        gameplan_dir.mkdir(parents=True, exist_ok=True)

        # Export summary
        summary_file = gameplan_dir / "gameplan_summary.json"
        with open(summary_file, "w") as f:
            json.dump(asdict(gameplan), f, indent=2, default=str)

        # Export practice plan
        practice_file = gameplan_dir / "practice_plan.txt"
        with open(practice_file, "w") as f:
            f.write(f"Tournament Preparation Plan\n")
            f.write(f"Opponent: {gameplan.opponent_username}\n")
            f.write(f"Tournament: {gameplan.tournament_type}\n")
            f.write(f"Estimated Time: {gameplan.estimated_prep_time} minutes\n\n")

            f.write("High Priority Practice:\n")
            for priority in gameplan.key_situations["high_priority"]:
                f.write(f"- {priority}\n")

            f.write("\nCounter Strategies:\n")
            for counter in gameplan.counter_strategies:
                f.write(f"- {counter.target_formation}: {counter.recommended_counter}\n")

        print(f"📁 Gameplan exported to: {gameplan_dir}")

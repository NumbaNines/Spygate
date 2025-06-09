"""Cross-Game Strategy Migration System for SpygateAI."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class GameVersion(Enum):
    MADDEN_25 = "madden_25"
    CFB_25 = "cfb_25"
    MADDEN_26 = "madden_26"


class StrategyMigrationStatus(Enum):
    DIRECT_COMPATIBLE = "direct_compatible"
    MINOR_ADJUSTMENTS = "minor_adjustments"
    MAJOR_ADAPTATION = "major_adaptation"
    INCOMPATIBLE = "incompatible"


@dataclass
class UniversalConcept:
    concept_id: str
    name: str
    description: str
    core_principle: str
    game_implementations: dict[str, str]
    effectiveness_data: dict[str, float]
    migration_difficulty: str


@dataclass
class StrategyMigration:
    migration_id: str
    source_game: str
    target_game: str
    original_strategy: dict
    migrated_strategy: dict
    confidence_score: float
    migration_notes: list[str]
    status: str


class CrossGameIntelligenceEngine:
    """Core engine for cross-game strategy migration."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.intelligence_db_path = self.project_root / "data" / "cross_game_intelligence.db"
        self.concepts_path = self.project_root / "concepts"

        self._setup_directory_structure()
        self._initialize_database()
        self._load_universal_concepts()

    def _setup_directory_structure(self):
        """Create directory structure for cross-game intelligence."""
        directories = [
            self.concepts_path / "universal",
            self.concepts_path / "madden_25",
            self.concepts_path / "cfb_25",
            self.concepts_path / "migrations",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self):
        """Initialize database for cross-game intelligence tracking."""
        self.intelligence_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.intelligence_db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS universal_concepts (
                    concept_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    core_principle TEXT,
                    game_implementations TEXT,
                    effectiveness_data TEXT,
                    migration_difficulty TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS strategy_migrations (
                    migration_id TEXT PRIMARY KEY,
                    source_game TEXT NOT NULL,
                    target_game TEXT NOT NULL,
                    original_strategy TEXT,
                    migrated_strategy TEXT,
                    confidence_score REAL,
                    migration_notes TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )

    def _load_universal_concepts(self):
        """Load pre-defined universal football concepts."""
        concepts = [
            {
                "concept_id": "trips_concept_family",
                "name": "Trips Formation Family",
                "description": "3-receiver bunched formations",
                "core_principle": "3_receivers_same_side_spacing",
                "game_implementations": {
                    "madden_25": "Shotgun Trips TE",
                    "cfb_25": "Spread Trips Right",
                },
                "effectiveness_data": {"avg_success_rate": 0.65},
                "migration_difficulty": "direct_compatible",
            },
            {
                "concept_id": "cover_2_defense",
                "name": "Cover 2 Defense Family",
                "description": "2 safeties deep coverage concepts",
                "core_principle": "2_safeties_deep_coverage",
                "game_implementations": {
                    "madden_25": "Cover 2 Man/Zone",
                    "cfb_25": "Cover 2 Match",
                },
                "effectiveness_data": {"avg_success_rate": 0.58},
                "migration_difficulty": "minor_adjustments",
            },
        ]

        for concept in concepts:
            self._save_universal_concept(concept)

    def migrate_strategy_to_new_game(
        self, source_strategy: dict, source_game: str, target_game: str
    ) -> dict:
        """Migrate strategies between EA football games."""
        migration_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        universal_concepts = self._identify_universal_concepts(source_strategy)
        migrated_strategy = {}
        migration_notes = []
        confidence_scores = []

        for concept_id in universal_concepts:
            concept = self._get_universal_concept(concept_id)
            if concept and target_game in concept.get("game_implementations", {}):
                target_implementation = concept["game_implementations"][target_game]
                migrated_strategy[concept_id] = {
                    "implementation": target_implementation,
                    "confidence": 0.85,
                }
                migration_notes.append(f"✅ {concept['name']}: Direct migration available")
                confidence_scores.append(0.85)
            else:
                migrated_strategy[concept_id] = {
                    "implementation": "REQUIRES_MANUAL_ADAPTATION",
                    "confidence": 0.3,
                }
                migration_notes.append(f"⚠️ Manual adaptation required")
                confidence_scores.append(0.3)

        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )

        if overall_confidence >= 0.8:
            status = "direct_compatible"
        elif overall_confidence >= 0.6:
            status = "minor_adjustments"
        else:
            status = "major_adaptation"

        migration_data = {
            "migration_id": migration_id,
            "source_game": source_game,
            "target_game": target_game,
            "original_strategy": source_strategy,
            "migrated_strategy": migrated_strategy,
            "confidence_score": overall_confidence,
            "migration_notes": migration_notes,
            "status": status,
        }

        self._save_strategy_migration(migration_data)
        return migration_data

    def get_day_one_gameplan(self, target_game: str) -> dict:
        """Generate day-1 gameplan for new game releases."""
        day_one_plan = {
            "target_game": target_game,
            "generated_at": datetime.now().isoformat(),
            "migrated_strategies": {},
            "actionable_concepts": [],
            "manual_adaptation_needed": [],
        }

        with sqlite3.connect(self.intelligence_db_path) as conn:
            cursor = conn.execute("SELECT * FROM universal_concepts")
            concepts_data = cursor.fetchall()

        for concept_row in concepts_data:
            concept_id = concept_row[0]
            name = concept_row[1]
            game_implementations = json.loads(concept_row[4])

            if target_game in game_implementations:
                day_one_plan["migrated_strategies"][concept_id] = {
                    "name": name,
                    "implementation": game_implementations[target_game],
                    "ready_for_use": True,
                }
                day_one_plan["actionable_concepts"].append(concept_id)
            else:
                day_one_plan["manual_adaptation_needed"].append(concept_id)

        return day_one_plan

    def _identify_universal_concepts(self, strategy: dict) -> list[str]:
        """Identify which universal concepts are present in a strategy."""
        concepts = []
        strategy_text = json.dumps(strategy).lower()

        if "trips" in strategy_text:
            concepts.append("trips_concept_family")
        if "cover 2" in strategy_text:
            concepts.append("cover_2_defense")

        return concepts

    def _get_universal_concept(self, concept_id: str) -> Optional[dict]:
        """Retrieve a universal concept by ID."""
        with sqlite3.connect(self.intelligence_db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM universal_concepts WHERE concept_id = ?", (concept_id,)
            )
            result = cursor.fetchone()

            if result:
                return {
                    "concept_id": result[0],
                    "name": result[1],
                    "description": result[2],
                    "core_principle": result[3],
                    "game_implementations": json.loads(result[4]),
                    "effectiveness_data": json.loads(result[5]),
                    "migration_difficulty": result[6],
                }
        return None

    def _save_universal_concept(self, concept: dict):
        """Save universal concept to database."""
        with sqlite3.connect(self.intelligence_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO universal_concepts
                (concept_id, name, description, core_principle, game_implementations,
                 effectiveness_data, migration_difficulty, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    concept["concept_id"],
                    concept["name"],
                    concept["description"],
                    concept["core_principle"],
                    json.dumps(concept["game_implementations"]),
                    json.dumps(concept["effectiveness_data"]),
                    concept["migration_difficulty"],
                    datetime.now().isoformat(),
                ),
            )

    def _save_strategy_migration(self, migration: dict):
        """Save strategy migration to database."""
        with sqlite3.connect(self.intelligence_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_migrations
                (migration_id, source_game, target_game, original_strategy, migrated_strategy,
                 confidence_score, migration_notes, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    migration["migration_id"],
                    migration["source_game"],
                    migration["target_game"],
                    json.dumps(migration["original_strategy"]),
                    json.dumps(migration["migrated_strategy"]),
                    migration["confidence_score"],
                    json.dumps(migration["migration_notes"]),
                    migration["status"],
                    datetime.now().isoformat(),
                ),
            )

    def analyze_cross_game_effectiveness(self, situation_type: str) -> dict:
        """Analyze effectiveness of strategies across different games."""
        effectiveness_data = {
            "situation_type": situation_type,
            "cross_game_analysis": {},
            "effectiveness_scores": {},
            "recommended_adaptations": [],
        }

        # Get all universal concepts related to this situation
        with sqlite3.connect(self.intelligence_db_path) as conn:
            cursor = conn.execute("SELECT * FROM universal_concepts")
            concepts = cursor.fetchall()

        for concept_row in concepts:
            concept_id = concept_row[0]
            name = concept_row[1]
            game_implementations = json.loads(concept_row[4]) if concept_row[4] else {}
            effectiveness_raw = json.loads(concept_row[5]) if concept_row[5] else {}

            # Analyze effectiveness across games
            for game, implementation in game_implementations.items():
                if game not in effectiveness_data["cross_game_analysis"]:
                    effectiveness_data["cross_game_analysis"][game] = []

                effectiveness_data["cross_game_analysis"][game].append(
                    {
                        "concept": name,
                        "implementation": implementation,
                        "effectiveness": effectiveness_raw.get("avg_success_rate", 0.5),
                    }
                )

        # Calculate overall effectiveness scores
        for game in effectiveness_data["cross_game_analysis"]:
            scores = [c["effectiveness"] for c in effectiveness_data["cross_game_analysis"][game]]
            effectiveness_data["effectiveness_scores"][game] = {
                "avg_effectiveness": sum(scores) / len(scores) if scores else 0.0,
                "concept_count": len(scores),
            }

        # Generate recommendations
        best_game = max(
            effectiveness_data["effectiveness_scores"].items(),
            key=lambda x: x[1]["avg_effectiveness"],
            default=(None, {"avg_effectiveness": 0}),
        )[0]

        if best_game:
            effectiveness_data["recommended_adaptations"].append(
                f"Consider adapting {best_game} implementations for {situation_type}"
            )

        return effectiveness_data

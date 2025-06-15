#!/usr/bin/env python3
"""
SpygateAI Desktop Application
=============================

Modern desktop application with SpygateAI functionality and sleek UI design.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QObject, QPoint, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

from formation_editor import FormationEditor

# Import other dependencies
from profile_picture_manager import ProfilePictureManager, is_emoji_profile
from spygate.core.hardware import HardwareDetector
from spygate.ml.cache_manager import GameAnalyzerCache, get_game_analyzer_cache
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.enhanced_ocr import EnhancedOCR
from user_database import User, UserDatabase


class AnalysisWorker(QThread):
    """Worker thread for video analysis using enhanced 5-class detection."""

    progress_updated = pyqtSignal(int, str)
    analysis_finished = pyqtSignal(str, list)
    error_occurred = pyqtSignal(str)
    clip_detected = pyqtSignal(object)

    def __init__(self, video_path, situation_preferences=None, analysis_mode="full"):
        super().__init__()
        self.video_path = video_path
        self.situation_preferences = situation_preferences or {
            "1st_down": True,
            "2nd_down": False,
            "3rd_down": True,
            "3rd_long": True,
            "4th_down": True,
            "goal_line": True,
        }

        # INDIVIDUAL CLIP SELECTION SYSTEM: Users can select exactly what they want
        self.clip_tags = {
            # Down & Distance Tags
            "1st_down": {"name": "1st Down", "icon": "🥇", "category": "Downs", "priority": "high"},
            "2nd_down": {
                "name": "2nd Down",
                "icon": "🥈",
                "category": "Downs",
                "priority": "medium",
            },
            "3rd_down": {"name": "3rd Down", "icon": "🥉", "category": "Downs", "priority": "high"},
            "3rd_long": {
                "name": "3rd & Long (7+ yards)",
                "icon": "📏",
                "category": "Downs",
                "priority": "critical",
            },
            "4th_down": {
                "name": "4th Down",
                "icon": "🔥",
                "category": "Downs",
                "priority": "critical",
            },
            # Field Position Tags
            "red_zone": {
                "name": "Red Zone (25 yard line)",
                "icon": "🎯",
                "category": "Field Position",
                "priority": "high",
            },
            "goal_line": {
                "name": "Goal Line (10 yard line)",
                "icon": "🏁",
                "category": "Field Position",
                "priority": "critical",
            },
            "midfield": {
                "name": "Midfield (50 yard line)",
                "icon": "⚖️",
                "category": "Field Position",
                "priority": "medium",
            },
            "deep_territory": {
                "name": "Deep Territory (Own 20)",
                "icon": "🏠",
                "category": "Field Position",
                "priority": "medium",
            },
            # Scoring Tags
            "touchdown": {
                "name": "Touchdown",
                "icon": "🏈",
                "category": "Scoring",
                "priority": "critical",
            },
            "field_goal": {
                "name": "Field Goal",
                "icon": "🥅",
                "category": "Scoring",
                "priority": "high",
            },
            "pat": {
                "name": "Point After Touchdown",
                "icon": "➕",
                "category": "Scoring",
                "priority": "medium",
            },
            "safety": {
                "name": "Safety",
                "icon": "🛡️",
                "category": "Scoring",
                "priority": "critical",
            },
            # Game Situation Tags
            "two_minute_drill": {
                "name": "Two Minute Drill",
                "icon": "⏰",
                "category": "Game Situations",
                "priority": "critical",
            },
            "overtime": {
                "name": "Overtime",
                "icon": "🕐",
                "category": "Game Situations",
                "priority": "critical",
            },
            "penalty": {
                "name": "Penalty",
                "icon": "🚩",
                "category": "Game Situations",
                "priority": "medium",
            },
            "turnover": {
                "name": "Turnover",
                "icon": "🔄",
                "category": "Game Situations",
                "priority": "critical",
            },
            "sack": {
                "name": "Sack",
                "icon": "💥",
                "category": "Game Situations",
                "priority": "high",
            },
            # Strategic Tags
            "blitz": {"name": "Blitz", "icon": "⚡", "category": "Strategy", "priority": "medium"},
            "play_action": {
                "name": "Play Action",
                "icon": "🎭",
                "category": "Strategy",
                "priority": "medium",
            },
            "screen_pass": {
                "name": "Screen Pass",
                "icon": "🕸️",
                "category": "Strategy",
                "priority": "medium",
            },
            "trick_play": {
                "name": "Trick Play",
                "icon": "🎪",
                "category": "Strategy",
                "priority": "high",
            },
            # Performance Tags
            "big_play": {
                "name": "Big Play (20+ yards)",
                "icon": "💨",
                "category": "Performance",
                "priority": "high",
            },
            "explosive_play": {
                "name": "Explosive Play (40+ yards)",
                "icon": "💥",
                "category": "Performance",
                "priority": "critical",
            },
            "three_and_out": {
                "name": "Three and Out",
                "icon": "🚫",
                "category": "Performance",
                "priority": "medium",
            },
            "sustained_drive": {
                "name": "Sustained Drive (8+ plays)",
                "icon": "🚂",
                "category": "Performance",
                "priority": "medium",
            },
        }

        # User's selected clip preferences (can be customized via UI)
        self.selected_clips = situation_preferences or {
            "1st_down": True,
            "3rd_down": True,
            "3rd_long": True,
            "4th_down": True,
            "red_zone": True,
            "goal_line": True,
            "touchdown": True,
            "turnover": True,
            "two_minute_drill": True,
        }

        # Dynamic optimization based on selected clips
        self.analysis_mode = analysis_mode
        self._optimize_processing_speed()

        self.should_stop = False
        self.analyzer = None
        self.hardware = HardwareDetector()
        self.last_progress_update = 0
        self.progress_update_interval = 1000
        self.memory_cleanup_interval = 5000
        self.last_memory_cleanup = 0

        # STATE TRACKING: Track previous game state to detect changes
        self.previous_game_state = None
        self.last_clip_frame = {}  # Track last frame where each clip type was created
        self.min_clip_interval = 300  # Minimum frames between same clip type (10 seconds at 30fps)

        # ADVANCED PLAY DETECTION SYSTEM: Handle face cam occlusion and streaming scenarios
        self.play_detection_state = {
            "last_down_change_frame": None,
            "last_preplay_detected": None,
            "last_playcall_detected": None,
            "waiting_for_play_confirmation": False,
            "down_change_timeout": 300,  # 10 seconds at 30fps - max time to wait for play confirmation
            "play_confirmation_window": 150,  # 5 seconds at 30fps - window to detect pre-play/play-call
            "preplay_occlusion_active": False,
            "consecutive_preplay_failures": 0,
            "max_preplay_failures": 5,  # If pre-play indicator fails 5 times in a row, assume face cam occlusion
            "expected_preplay_frames": [],  # Track when we expect pre-play but don't see it
        }

        # COMPREHENSIVE PLAY BOUNDARY DETECTION SYSTEM
        self.play_boundary_state = {
            # Previous play state for comparison
            "last_down": None,
            "last_distance": None,
            "last_yard_line": None,
            "last_possession_team": None,
            "last_territory": None,
            "last_game_clock": None,
            "last_quarter": None,
            # Play progression tracking
            "play_start_frame": None,
            "play_end_frame": None,
            "yards_gained": 0,
            "first_down_achieved": False,
            "possession_changed": False,
            # Clip boundary management
            "active_clip_start": None,
            "active_clip_situation": None,
            "clips_created_this_drive": [],
            # Full play tracking for entire play clips
            "current_play_start_frame": None,
            "current_play_start_info": None,
            # Advanced detection
            "consecutive_same_state": 0,
            "max_same_state_frames": 90,  # 3 seconds at 30fps before assuming play ended
            "min_play_duration": 60,  # 2 seconds minimum play duration
        }
        self.pending_clip_info = None  # Track clips that span play boundaries

        # DUPLICATE CLIP PREVENTION SYSTEM
        self.duplicate_prevention = {
            "created_clips": [],  # List of all created clips with timeframes
            "last_clip_end_frame": 0,  # Track when the last clip ended
            "min_clip_gap_frames": 60,  # Minimum 2 seconds between clips (at 30fps)
            "overlap_threshold": 0.5,  # 50% overlap threshold for duplicate detection
            "max_clips_per_minute": 6,  # Maximum 6 clips per minute to prevent spam
            "recent_clips_window": 1800,  # 60 seconds window for rate limiting (at 30fps)
        }

    def _optimize_processing_speed(self):
        """Dynamically optimize processing speed based on selected clips."""
        selected_count = sum(1 for enabled in self.selected_clips.values() if enabled)
        total_clips = len(self.clip_tags)
        selection_ratio = selected_count / total_clips

        # Calculate optimal frame skip based on selection
        if selection_ratio <= 0.2:  # Very selective (≤20% of clips)
            self.optimized_frame_skip = 960  # 16 seconds - Ultra fast
            self.optimization_level = "🚀 Ultra Fast"
        elif selection_ratio <= 0.4:  # Selective (≤40% of clips)
            self.optimized_frame_skip = 720  # 12 seconds - Very fast
            self.optimization_level = "⚡ Very Fast"
        elif selection_ratio <= 0.6:  # Moderate (≤60% of clips)
            self.optimized_frame_skip = 480  # 8 seconds - Fast
            self.optimization_level = "🏃 Fast"
        elif selection_ratio <= 0.8:  # Comprehensive (≤80% of clips)
            self.optimized_frame_skip = 360  # 6 seconds - Balanced
            self.optimization_level = "⚖️ Balanced"
        else:  # Full analysis (>80% of clips)
            self.optimized_frame_skip = 240  # 4 seconds - Comprehensive
            self.optimization_level = "📊 Comprehensive"

        # Identify high-priority clips for extra processing
        self.priority_clips = {
            tag: info
            for tag, info in self.clip_tags.items()
            if self.selected_clips.get(tag, False) and info["priority"] == "critical"
        }

        print(
            f"🎛️ Optimization: {self.optimization_level} ({selected_count}/{total_clips} clips selected)"
        )
        print(
            f"📈 Frame skip: {self.optimized_frame_skip} frames ({self.optimized_frame_skip/60:.1f}s intervals)"
        )
        if self.priority_clips:
            print(f"🔥 Priority clips: {', '.join(self.priority_clips.keys())}")

    def _should_process_situation(self, situation_type: str) -> bool:
        """Check if we should process OCR for this situation type."""
        return self.selected_clips.get(situation_type, False)

    def map_situation_type_to_preference(self, situation_type: str) -> list[str]:
        """Map analyzer situation types to preference keys."""
        mapping = {
            # Third down mappings
            "third_and_long": ["3rd_long", "3rd_down"],
            "third_and_long_red_zone": ["3rd_long", "red_zone"],
            "third_and_short": ["3rd_down"],
            "third_and_short_goal_line": ["3rd_down", "goal_line"],
            "third_and_short_red_zone": ["3rd_down", "red_zone"],
            "third_down": ["3rd_down"],
            "third_down_red_zone": ["3rd_down", "red_zone"],
            # Fourth down mappings
            "fourth_down": ["4th_down"],
            "fourth_down_goal_line": ["4th_down", "goal_line"],
            "fourth_down_red_zone": ["4th_down", "red_zone"],
            # Field position mappings
            "goal_line_offense": ["goal_line"],
            "goal_line_defense": ["goal_line"],
            "red_zone_offense": ["red_zone"],
            "red_zone_defense": ["red_zone"],
            "backed_up_offense": ["deep_territory"],  # When backed up in own territory
            "pressure_defense": [],  # No direct mapping
            "scoring_position": ["red_zone"],
            # Time-based mappings
            "two_minute_drill": ["two_minute_drill"],
            "fourth_quarter": [],  # No direct mapping
            # Default
            "normal_play": [],
        }

        return mapping.get(situation_type, [])

    def _should_create_clip(self, game_state, situation_context) -> bool:
        """Determine if we should create a clip based on selected clip preferences and state changes."""
        if not game_state:
            return False

        # Extract situation details
        down = game_state.down
        distance = game_state.distance
        current_frame = getattr(situation_context, "frame_number", 0)

        # DEBUG: Log game state every 300 frames (10 seconds)
        if current_frame % 300 == 0:
            print(f"🔍 CLIP CHECK at frame {current_frame}: Down={down}, Distance={distance}")
            print(
                f"   📋 Situation Preferences: {list(k for k, v in self.situation_preferences.items() if v)}"
            )
            print(f"   🎮 Game State Details: {game_state}")
            if hasattr(game_state, "__dict__"):
                print(f"   📊 Game State Attributes: {game_state.__dict__}")

        # AGGRESSIVE DEBUG: Create test clips every 600 frames (20 seconds) regardless of game state
        # DISABLED: This was causing incorrect clip creation
        # if current_frame % 600 == 0:
        #     print(f"🧪 CREATING TEST CLIP at frame {current_frame} for debugging")
        #     return True

        # Check if this is a down change (the key fix!)
        down_changed = False
        if self.previous_game_state:
            prev_down = self.previous_game_state.down
            if prev_down != down and down is not None:
                down_changed = True
                print(f"🔄 DOWN CHANGE DETECTED: {prev_down} → {down} at frame {current_frame}")

        # CRITICAL FIX: Store the CURRENT game state for analysis BEFORE updating previous_game_state
        analysis_game_state = game_state
        analysis_down = down
        analysis_distance = distance

        # Map game state to clip tags
        clip_matches = []

        # === DOWNS CATEGORY ===
        # Create clips for ALL occurrences of selected downs (every single occurrence)
        # But add basic deduplication to prevent spam (minimum 3 seconds between same down clips)
        if analysis_down is not None:
            min_gap_frames = 180  # 3 seconds at 60fps

            if analysis_down == 1 and self.situation_preferences.get("1st_down", False):
                if current_frame - self.last_clip_frame.get("1st_down", 0) >= min_gap_frames:
                    clip_matches.append("1st_down")
            elif analysis_down == 2 and self.situation_preferences.get("2nd_down", False):
                if current_frame - self.last_clip_frame.get("2nd_down", 0) >= min_gap_frames:
                    clip_matches.append("2nd_down")
            elif analysis_down == 3:
                if self.situation_preferences.get("3rd_down", False):
                    if current_frame - self.last_clip_frame.get("3rd_down", 0) >= min_gap_frames:
                        clip_matches.append("3rd_down")
                if (
                    analysis_distance
                    and analysis_distance >= 7
                    and self.situation_preferences.get("3rd_long", False)
                ):
                    if current_frame - self.last_clip_frame.get("3rd_long", 0) >= min_gap_frames:
                        clip_matches.append("3rd_long")
            elif analysis_down == 4 and self.situation_preferences.get("4th_down", False):
                if current_frame - self.last_clip_frame.get("4th_down", 0) >= min_gap_frames:
                    clip_matches.append("4th_down")

        # === FIELD POSITION CATEGORY (FIXED WITH TERRITORY CONTEXT) ===
        if hasattr(analysis_game_state, "yard_line") and analysis_game_state.yard_line:
            # Get territory context if available
            territory = getattr(analysis_game_state, "territory", None)

            # Goal line - must be in opponent territory
            if (
                territory == "opponent"
                and analysis_game_state.yard_line <= 10
                and self.situation_preferences.get("goal_line", False)
            ):
                clip_matches.append("goal_line")
            # Red zone - must be in opponent territory
            elif (
                territory == "opponent"
                and analysis_game_state.yard_line <= 25
                and self.situation_preferences.get("red_zone", False)
            ):
                clip_matches.append("red_zone")
            # Midfield - exactly at 50
            elif analysis_game_state.yard_line == 50 and self.situation_preferences.get(
                "midfield", False
            ):
                clip_matches.append("midfield")
            # Deep territory - backed up in own territory
            elif (
                territory == "own"
                and analysis_game_state.yard_line <= 20
                and self.situation_preferences.get("deep_territory", False)
            ):
                clip_matches.append("deep_territory")

        # === SCORING CATEGORY ===
        # Check for touchdown indicators
        if (
            hasattr(analysis_game_state, "is_touchdown")
            and analysis_game_state.is_touchdown
            and self.situation_preferences.get("touchdown", False)
        ):
            clip_matches.append("touchdown")

        # Check for field goal indicators
        if (
            hasattr(analysis_game_state, "is_field_goal")
            and analysis_game_state.is_field_goal
            and self.situation_preferences.get("field_goal", False)
        ):
            clip_matches.append("field_goal")

        # Check for PAT indicators
        if (
            hasattr(analysis_game_state, "is_pat")
            and analysis_game_state.is_pat
            and self.situation_preferences.get("pat", False)
        ):
            clip_matches.append("pat")

        # Check for safety indicators
        if (
            hasattr(analysis_game_state, "is_safety")
            and analysis_game_state.is_safety
            and self.situation_preferences.get("safety", False)
        ):
            clip_matches.append("safety")

        # === GAME SITUATIONS CATEGORY (FIXED WITH MAPPING) ===
        if hasattr(situation_context, "situation_type"):
            situation_type = situation_context.situation_type

            # Map the analyzer situation type to preference keys
            mapped_preferences = self.map_situation_type_to_preference(situation_type)
            for pref_key in mapped_preferences:
                if self.situation_preferences.get(pref_key, False):
                    clip_matches.append(pref_key)
                    print(f"🎯 MAPPED: {situation_type} → {pref_key}")

            # Direct mappings that don't need translation
            if situation_type == "two_minute_drill" and self.situation_preferences.get(
                "two_minute_drill", False
            ):
                clip_matches.append("two_minute_drill")

            # Overtime detection
            if situation_type == "overtime" and self.situation_preferences.get("overtime", False):
                clip_matches.append("overtime")

            # Penalty detection
            if situation_type == "penalty" and self.situation_preferences.get("penalty", False):
                clip_matches.append("penalty")

            # Turnover detection
            if situation_type == "turnover" and self.situation_preferences.get("turnover", False):
                clip_matches.append("turnover")

            # Sack detection
            if situation_type == "sack" and self.situation_preferences.get("sack", False):
                clip_matches.append("sack")

        # === SPECIAL SITUATIONS (NEW) ===
        # Handle special situations detected by the analyzer
        if hasattr(situation_context, "special_situations"):
            for special_sit in situation_context.special_situations:
                if self.situation_preferences.get(special_sit, False):
                    clip_matches.append(special_sit)
                    print(f"🎯 SPECIAL: {special_sit} detected")

        # === STRATEGY CATEGORY ===
        # Check for strategic indicators in game state
        if hasattr(analysis_game_state, "play_type"):
            play_type = analysis_game_state.play_type

            if play_type == "blitz" and self.situation_preferences.get("blitz", False):
                clip_matches.append("blitz")
            elif play_type == "play_action" and self.situation_preferences.get(
                "play_action", False
            ):
                clip_matches.append("play_action")
            elif play_type == "screen_pass" and self.situation_preferences.get(
                "screen_pass", False
            ):
                clip_matches.append("screen_pass")
            elif play_type == "trick_play" and self.situation_preferences.get("trick_play", False):
                clip_matches.append("trick_play")

        # === PERFORMANCE CATEGORY ===
        # Check for yards gained indicators
        if hasattr(analysis_game_state, "yards_gained"):
            yards_gained = analysis_game_state.yards_gained

            if yards_gained >= 40 and self.situation_preferences.get("explosive_play", False):
                clip_matches.append("explosive_play")
            elif yards_gained >= 20 and self.situation_preferences.get("big_play", False):
                clip_matches.append("big_play")

        # Check for three and out
        if (
            hasattr(situation_context, "is_three_and_out")
            and situation_context.is_three_and_out
            and self.situation_preferences.get("three_and_out", False)
        ):
            clip_matches.append("three_and_out")

        # Check for sustained drive (8+ plays)
        if (
            hasattr(situation_context, "play_count")
            and situation_context.play_count >= 8
            and self.situation_preferences.get("sustained_drive", False)
        ):
            clip_matches.append("sustained_drive")

        # === STRICT PREFERENCE MATCHING ===
        # ONLY create clips that exactly match user's selected preferences

        # Update last clip frame for matched clips
        for clip_type in clip_matches:
            self.last_clip_frame[clip_type] = current_frame

        # CRITICAL FIX: Update previous game state AFTER all analysis is complete
        self.previous_game_state = game_state

        # DEBUG: Log clip matches with EXACT values used for analysis
        if len(clip_matches) > 0:
            print(f"✅ CLIP MATCHES at frame {current_frame}: {clip_matches}")
            print(
                f"   🎯 EXACT VALUES USED FOR CLIP: Down={analysis_down}, Distance={analysis_distance}"
            )
            print(
                f"   📋 User wants: {list(k for k, v in self.situation_preferences.items() if v)}"
            )
            return True
        elif current_frame % 300 == 0:
            print(f"❌ NO CLIP MATCHES at frame {current_frame}")
            print(f"   🎯 Detected: Down={analysis_down}, Distance={analysis_distance}")
            print(
                f"   📋 User wants: {list(k for k, v in self.situation_preferences.items() if v)}"
            )

        return False

    def _is_duplicate_clip(self, start_frame: int, end_frame: int, current_frame: int) -> bool:
        """
        Comprehensive duplicate clip prevention using multiple strategies:
        1. Overlap detection - Check if new clip overlaps significantly with existing clips
        2. Minimum gap enforcement - Ensure minimum time between clips
        3. Rate limiting - Prevent too many clips in a short time window
        4. Boundary validation - Ensure clips have reasonable duration
        """
        # Strategy 1: Check for overlapping clips
        for existing_clip in self.duplicate_prevention["created_clips"]:
            existing_start = existing_clip["start_frame"]
            existing_end = existing_clip["end_frame"]

            # Calculate overlap
            overlap_start = max(start_frame, existing_start)
            overlap_end = min(end_frame, existing_end)

            if overlap_start < overlap_end:  # There is overlap
                overlap_duration = overlap_end - overlap_start
                new_clip_duration = end_frame - start_frame
                existing_clip_duration = existing_end - existing_start

                # Calculate overlap percentage for both clips
                new_overlap_pct = overlap_duration / max(new_clip_duration, 1)
                existing_overlap_pct = overlap_duration / max(existing_clip_duration, 1)

                # If either clip has significant overlap, consider it a duplicate
                if (
                    new_overlap_pct > self.duplicate_prevention["overlap_threshold"]
                    or existing_overlap_pct > self.duplicate_prevention["overlap_threshold"]
                ):
                    print(
                        f"🚫 DUPLICATE DETECTED: {new_overlap_pct:.1%} overlap with existing clip"
                    )
                    print(
                        f"   New: {start_frame}-{end_frame}, Existing: {existing_start}-{existing_end}"
                    )
                    return True

        # Strategy 2: Minimum gap enforcement
        min_gap = self.duplicate_prevention["min_clip_gap_frames"]
        if (start_frame - self.duplicate_prevention["last_clip_end_frame"]) < min_gap:
            gap_seconds = (start_frame - self.duplicate_prevention["last_clip_end_frame"]) / 30.0
            print(
                f"🚫 MINIMUM GAP VIOLATION: Only {gap_seconds:.1f}s since last clip (need {min_gap/30.0:.1f}s)"
            )
            return True

        # Strategy 3: Rate limiting - Check clips in recent window
        recent_window = self.duplicate_prevention["recent_clips_window"]
        recent_clips = [
            clip
            for clip in self.duplicate_prevention["created_clips"]
            if (current_frame - clip["end_frame"]) <= recent_window
        ]

        max_clips = self.duplicate_prevention["max_clips_per_minute"]
        if len(recent_clips) >= max_clips:
            print(
                f"🚫 RATE LIMIT EXCEEDED: {len(recent_clips)} clips in last {recent_window/30.0:.0f}s (max {max_clips})"
            )
            return True

        # Strategy 4: Boundary validation
        clip_duration = end_frame - start_frame
        if clip_duration < 30:  # Less than 1 second
            print(f"🚫 CLIP TOO SHORT: {clip_duration/30.0:.1f}s duration (minimum 1s)")
            return True

        if clip_duration > 1800:  # More than 60 seconds
            print(f"🚫 CLIP TOO LONG: {clip_duration/30.0:.1f}s duration (maximum 60s)")
            return True

        # All checks passed - not a duplicate
        return False

    def _register_created_clip(self, start_frame: int, end_frame: int, situation: str):
        """Register a newly created clip in the duplicate prevention system."""
        clip_info = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "situation": situation,
            "created_at_frame": end_frame,  # When this clip was created
        }

        self.duplicate_prevention["created_clips"].append(clip_info)
        self.duplicate_prevention["last_clip_end_frame"] = end_frame

        # Clean up old clips to prevent memory bloat
        current_frame = end_frame
        cleanup_window = (
            self.duplicate_prevention["recent_clips_window"] * 3
        )  # Keep 3x the recent window
        self.duplicate_prevention["created_clips"] = [
            clip
            for clip in self.duplicate_prevention["created_clips"]
            if (current_frame - clip["end_frame"]) <= cleanup_window
        ]

        print(
            f"✅ CLIP REGISTERED: {situation} ({start_frame}-{end_frame}) - Total tracked: {len(self.duplicate_prevention['created_clips'])}"
        )

    def _should_create_clip_by_interval(self, clip_type: str, current_frame: int) -> bool:
        """Check if enough time has passed since the last clip of this type."""
        if clip_type not in self.last_clip_frame:
            return True  # First time detecting this clip type

        frames_since_last = current_frame - self.last_clip_frame[clip_type]
        return frames_since_last >= self.min_clip_interval

    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:
        """
        FIXED: Individual play boundary detection that properly ends clips at play completion.

        This method now:
        1. Detects when a play starts (down change, first down, etc.)
        2. Tracks the play in progress
        3. Detects when the play ends (next down change, clock stoppage, etc.)
        4. Creates clips that contain ONLY ONE PLAY
        """
        boundary_info = {
            "play_started": False,
            "play_ended": False,
            "play_in_progress": False,
            "clip_should_start": False,
            "clip_should_end": False,
            "recommended_clip_start": None,
            "recommended_clip_end": None,
            "play_type": "unknown",
            "play_situation": "normal",
            "confidence": 0.0,
        }

        if not game_state:
            return boundary_info

        # Extract current game state
        current_down = game_state.down
        current_distance = game_state.distance
        current_yard_line = getattr(game_state, "yard_line", None)
        current_possession = getattr(game_state, "possession_team", None)
        current_game_clock = getattr(game_state, "time", None)
        current_quarter = getattr(game_state, "quarter", None)

        # Get previous state
        prev_state = self.play_boundary_state

        # === CHECK IF WE'RE TRACKING AN ACTIVE PLAY ===
        if prev_state.get("active_play_frame") is not None:
            # We have an active play - check if it has ended
            frames_since_play_start = current_frame - prev_state["active_play_frame"]

            # CRITICAL: Detect play end conditions
            play_ended = False
            end_reason = None

            # 1. DOWN CHANGE - Most reliable indicator that previous play ended
            if (
                prev_state["last_down"] is not None
                and current_down is not None
                and current_down != prev_state["last_down"]
            ):
                play_ended = True
                end_reason = "down_changed"
                print(f"🏁 PLAY ENDED: Down changed {prev_state['last_down']} → {current_down}")

            # 2. FIRST DOWN ACHIEVED - Distance reset to 10
            elif (
                current_down == 1
                and current_distance == 10
                and prev_state.get("last_distance") != 10
            ):
                play_ended = True
                end_reason = "first_down_achieved"
                print(f"🏁 PLAY ENDED: First down achieved")

            # 3. POSSESSION CHANGE - Turnover occurred
            elif (
                prev_state.get("last_possession_team")
                and current_possession
                and current_possession != prev_state["last_possession_team"]
            ):
                play_ended = True
                end_reason = "possession_changed"
                print(f"🏁 PLAY ENDED: Possession changed to {current_possession}")

            # 4. SIGNIFICANT YARD LINE CHANGE - Play resulted in big gain/loss
            elif prev_state.get("last_yard_line") is not None and current_yard_line is not None:
                yard_change = abs(current_yard_line - prev_state["last_yard_line"])
                if yard_change >= 15:  # 15+ yard change indicates play completed
                    play_ended = True
                    end_reason = f"big_play_{yard_change}_yards"
                    print(f"🏁 PLAY ENDED: Big play for {yard_change} yards")

            # 5. CLOCK STOPPAGE - Incomplete pass, out of bounds, etc.
            elif (
                current_game_clock
                and prev_state.get("last_game_clock")
                and self._detect_clock_stoppage(current_game_clock, prev_state["last_game_clock"])
            ):
                # Wait a bit to confirm it's not just a measurement
                if frames_since_play_start > 60:  # 2 seconds at 30fps
                    play_ended = True
                    end_reason = "clock_stopped"
                    print(f"🏁 PLAY ENDED: Clock stopped at {current_game_clock}")

            # 6. MAXIMUM PLAY DURATION - Safety limit (most plays < 10 seconds)
            elif frames_since_play_start > 300:  # 10 seconds at 30fps
                play_ended = True
                end_reason = "max_duration"
                print(f"🏁 PLAY ENDED: Maximum duration reached")

            # If play ended, mark it for clip completion
            if play_ended:
                boundary_info["play_ended"] = True
                boundary_info["clip_should_end"] = True
                # End clip shortly after the play ends (1 second buffer)
                boundary_info["recommended_clip_end"] = current_frame + 30

                # Clear active play tracking
                self.play_boundary_state["active_play_frame"] = None
                self.play_boundary_state["active_play_data"] = None

                print(
                    f"🎬 CLIP END: Play ended ({end_reason}) - Clip will end at frame {boundary_info['recommended_clip_end']}"
                )

        # === DETECT NEW PLAY START ===
        # Only start new plays if we're not currently tracking one
        if prev_state.get("active_play_frame") is None:
            play_started = False
            start_reason = None

            # Check various play start conditions
            if prev_state["last_down"] is None and current_down is not None:
                # First play detected
                play_started = True
                start_reason = "initial_play"
            elif (
                prev_state["last_down"] is not None
                and current_down is not None
                and current_down != prev_state["last_down"]
            ):
                # Down changed - new play starting
                play_started = True
                start_reason = f"down_change_{prev_state['last_down']}_to_{current_down}"
            elif current_down == 1 and prev_state["last_down"] != 1:
                # First down achieved
                play_started = True
                start_reason = "first_down"

            if play_started:
                boundary_info["play_started"] = True
                boundary_info["clip_should_start"] = True
                boundary_info["play_type"] = self._classify_play_by_down(
                    current_down, current_distance
                )

                # Start clip with 2-second pre-roll
                boundary_info["recommended_clip_start"] = max(0, current_frame - 60)

                # Track this as the active play
                self.play_boundary_state["active_play_frame"] = current_frame
                self.play_boundary_state["active_play_data"] = {
                    "down": current_down,
                    "distance": current_distance,
                    "yard_line": current_yard_line,
                    "start_reason": start_reason,
                }

                print(f"🎬 PLAY STARTED: {start_reason} - {current_down} & {current_distance}")
                print(f"   📍 Clip will start at frame {boundary_info['recommended_clip_start']}")

        # === UPDATE STATE TRACKING ===
        # Only update with non-None values to prevent state loss
        if current_down is not None:
            self.play_boundary_state["last_down"] = current_down
        if current_distance is not None:
            self.play_boundary_state["last_distance"] = current_distance
        if current_yard_line is not None:
            self.play_boundary_state["last_yard_line"] = current_yard_line
        if current_possession is not None:
            self.play_boundary_state["last_possession_team"] = current_possession
        if current_game_clock is not None:
            self.play_boundary_state["last_game_clock"] = current_game_clock

        # Calculate confidence
        confidence_factors = []
        if current_down is not None:
            confidence_factors.append(0.3)
        if current_distance is not None:
            confidence_factors.append(0.2)
        if current_yard_line is not None:
            confidence_factors.append(0.2)
        if current_possession is not None:
            confidence_factors.append(0.15)
        if current_game_clock is not None:
            confidence_factors.append(0.15)

        boundary_info["confidence"] = sum(confidence_factors)

        return boundary_info

    def _classify_play_by_down(self, down: int, distance: int) -> str:
        """Classify play type based on down and distance."""
        if down == 1:
            return "first_down"
        elif down == 2:
            if distance <= 3:
                return "second_and_short"
            elif distance >= 8:
                return "second_and_long"
            else:
                return "second_down"
        elif down == 3:
            if distance <= 3:
                return "third_and_short"
            elif distance >= 8:
                return "third_and_long"
            else:
                return "third_down"
        elif down == 4:
            return "fourth_down"
        else:
            return "unknown_down"

    def _classify_play_situation(
        self, down: int, distance: int, yard_line: int, game_clock: str, quarter: int
    ) -> str:
        """Classify the overall situation context for better clip naming."""
        situations = []

        # Down situation
        if down == 3:
            situations.append("third_down")
        elif down == 4:
            situations.append("fourth_down")

        # Field position
        if yard_line is not None:
            if yard_line <= 20:
                situations.append("red_zone")
            elif yard_line <= 5:
                situations.append("goal_line")

        # Time situation
        if game_clock and quarter:
            if self._is_two_minute_drill(game_clock, quarter):
                situations.append("two_minute_drill")
            elif self._is_end_of_quarter(game_clock):
                situations.append("end_of_quarter")

        # Distance situation
        if distance is not None:
            if distance <= 2:
                situations.append("short_yardage")
            elif distance >= 10:
                situations.append("long_yardage")

        return "_".join(situations) if situations else "normal_play"

    def _detect_clock_stoppage(self, current_clock: str, previous_clock: str) -> bool:
        """Detect if the game clock has stopped, indicating play ended."""
        try:
            # Parse clock times (format: "MM:SS")
            current_parts = current_clock.split(":")
            previous_parts = previous_clock.split(":")

            current_total = int(current_parts[0]) * 60 + int(current_parts[1])
            previous_total = int(previous_parts[0]) * 60 + int(previous_parts[1])

            # Clock should be running down, if it's the same for multiple frames, it's stopped
            return current_total == previous_total

        except (ValueError, IndexError):
            return False

    def _is_two_minute_drill(self, game_clock: str, quarter: int) -> bool:
        """Check if we're in a two-minute drill situation."""
        try:
            if quarter not in [2, 4]:  # Only 2nd and 4th quarters
                return False

            parts = game_clock.split(":")
            total_seconds = int(parts[0]) * 60 + int(parts[1])
            return total_seconds <= 120  # 2 minutes or less

        except (ValueError, IndexError):
            return False

    def _is_end_of_quarter(self, game_clock: str) -> bool:
        """Check if we're near the end of a quarter."""
        try:
            parts = game_clock.split(":")
            total_seconds = int(parts[0]) * 60 + int(parts[1])
            return total_seconds <= 30  # 30 seconds or less

        except (ValueError, IndexError):
            return False

    def _validate_play_detection(self, game_state, situation_context, current_frame: int) -> bool:
        """
        Advanced play detection with fallback systems for face cam occlusion and streaming scenarios.

        This method implements a multi-layered approach:
        1. Primary: Down change detection + pre-play/play-call confirmation
        2. Secondary: Time-based fallback for face cam occlusion
        3. Tertiary: Pattern recognition for consistent streaming setups
        """

        # Extract current state
        current_down = game_state.down if game_state else None
        has_preplay = (
            hasattr(situation_context, "preplay_detected") and situation_context.preplay_detected
        )
        has_playcall = (
            hasattr(situation_context, "playcall_detected") and situation_context.playcall_detected
        )
        hud_visible = game_state is not None and current_down is not None

        # Track pre-play indicator visibility for face cam detection
        # The key insight: HUD (down/distance) might be visible, but pre-play indicator could be blocked
        if self.play_detection_state["waiting_for_play_confirmation"]:
            # We're expecting to see pre-play indicator after a down change
            if has_preplay or has_playcall:
                # Great! Pre-play indicator is working
                self.play_detection_state["consecutive_preplay_failures"] = 0
                self.play_detection_state["preplay_occlusion_active"] = False
            else:
                # We expected pre-play but didn't see it - could be face cam occlusion
                self.play_detection_state["consecutive_preplay_failures"] += 1
                if (
                    self.play_detection_state["consecutive_preplay_failures"]
                    >= self.play_detection_state["max_preplay_failures"]
                ):
                    self.play_detection_state["preplay_occlusion_active"] = True
                    print(
                        f"🎥 PRE-PLAY OCCLUSION DETECTED: Face cam likely blocking pre-play indicator at frame {current_frame}"
                    )
        else:
            # Reset failure counter when not actively waiting for pre-play
            if has_preplay or has_playcall:
                self.play_detection_state["consecutive_preplay_failures"] = 0
                self.play_detection_state["preplay_occlusion_active"] = False

        # === PRIMARY DETECTION: Down Change + Play Confirmation ===
        if current_down and self.previous_game_state:
            prev_down = self.previous_game_state.down

            # Detect down change
            if prev_down != current_down:
                self.play_detection_state["last_down_change_frame"] = current_frame
                self.play_detection_state["waiting_for_play_confirmation"] = True
                print(
                    f"🔄 DOWN CHANGE: {prev_down} → {current_down} at frame {current_frame}, waiting for play confirmation..."
                )
                return True  # Immediate clip creation on down change

        # === SIMPLIFIED DETECTION: For now, allow clips based on interval to ensure system works ===
        # This ensures we get clips while the advanced system is being refined
        if current_down is not None:
            return self._should_create_clip_by_interval(f"down_{current_down}", current_frame)

        # Update pre-play/play-call detection timestamps
        if has_preplay:
            self.play_detection_state["last_preplay_detected"] = current_frame
            print(f"🏈 PRE-PLAY INDICATOR detected at frame {current_frame}")

        if has_playcall:
            self.play_detection_state["last_playcall_detected"] = current_frame
            print(f"📋 PLAY CALL SCREEN detected at frame {current_frame}")

        # === SECONDARY DETECTION: Time-Based Fallback ===
        if self.play_detection_state["waiting_for_play_confirmation"]:
            last_down_change = self.play_detection_state["last_down_change_frame"]
            frames_since_down_change = current_frame - last_down_change

            # Check if we got play confirmation within the window
            confirmation_window = self.play_detection_state["play_confirmation_window"]
            if frames_since_down_change <= confirmation_window:
                if has_preplay or has_playcall:
                    print(
                        f"✅ PLAY CONFIRMED: Pre-play/play-call detected {frames_since_down_change} frames after down change"
                    )
                    self.play_detection_state["waiting_for_play_confirmation"] = False
                    return True

            # Timeout: No confirmation received, use fallback
            timeout = self.play_detection_state["down_change_timeout"]
            if frames_since_down_change >= timeout:
                print(
                    f"⏰ TIMEOUT FALLBACK: No play confirmation after {frames_since_down_change} frames, assuming valid play"
                )
                self.play_detection_state["waiting_for_play_confirmation"] = False
                return True

        # === TERTIARY DETECTION: Pre-Play Occlusion Fallback System ===
        if self.play_detection_state["preplay_occlusion_active"]:
            # Use time-based intervals when pre-play indicator is consistently blocked by face cam
            print(f"🎥 Using pre-play occlusion fallback at frame {current_frame}")
            return self._should_create_clip_by_interval("preplay_occlusion_fallback", current_frame)

        # === QUATERNARY DETECTION: Pattern Recognition ===
        # Detect consistent streaming patterns (e.g., streamer always hides HUD during certain times)
        if self._detect_streaming_pattern(current_frame):
            print(f"📺 STREAMING PATTERN: Detected consistent HUD occlusion pattern")
            return self._should_create_clip_by_interval("streaming_pattern", current_frame)

        return False

    def _detect_streaming_pattern(self, current_frame: int) -> bool:
        """
        Detect if the streamer has consistent patterns of pre-play indicator occlusion.
        This could be face cam positioning, overlay schedules, etc.
        """
        # Simple pattern detection: if we've had multiple pre-play failures in a pattern
        failures = self.play_detection_state["consecutive_preplay_failures"]

        # Pattern 1: Consistent face cam placement blocking pre-play area (moderate failures)
        if 3 <= failures < 5:
            return current_frame % 300 == 0  # Every 10 seconds

        # Pattern 2: Heavy face cam occlusion of pre-play area
        if failures >= 2:
            return current_frame % 150 == 0  # Every 5 seconds

        return False

    def load_model(self):
        """Initialize enhanced game analyzer with 8-class detection, advanced caching, and hybrid OCR system."""
        try:
            print("🤖 Initializing enhanced game analyzer with hybrid OCR + situational logic...")

            # Initialize global cache system
            cache = get_game_analyzer_cache()
            cache_stats = cache.cache.get_stats()
            cache_health = cache.cache.health_check()
            print(
                f"📊 Cache Status: {cache_health['overall_status']} - Hit Rate: {cache_stats['hit_rate']:.1%}"
            )

            # Initialize analyzer with hardware detection, caching, and hybrid OCR
            self.analyzer = EnhancedGameAnalyzer(
                hardware=self.hardware,
                model_path=Path(
                    "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
                ),
                debug_output_dir=Path(
                    "debug_triangle_detection"
                ),  # Enable 97.6% accuracy triangle detection
            )

            # Enable debug mode for detailed analysis logging
            self.analyzer.enable_debug_mode(True)
            print("🔍 Debug mode enabled - collecting detailed analysis data")

            # Enable hybrid OCR features
            print("🧠 Enabling hybrid OCR + situational logic features...")
            print("   ✅ PAT Detection: Recognizes Point After Touchdown situations")
            print("   ✅ Penalty Detection: FLAG text + yellow color analysis (30% threshold)")
            print("   ✅ Temporal Validation: Uses next play to validate OCR results")
            print("   ✅ Yard Line Extraction: Robust OCR with corrections (A35, H22, 50)")
            print("   ✅ Deep Historical Context: Drive momentum and stalled patterns")
            print("   ✅ Color Analysis: HSV color detection for penalty regions")
            print("   ✅ Drive Intelligence: Possession consistency and field position trends")

            print("✅ Enhanced game analyzer with hybrid OCR system initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to initialize analyzer: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            return False

    def cleanup_memory(self, cap=None):
        """Clean up memory by releasing resources."""
        if cap is not None:
            cap.release()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        import gc

        gc.collect()

    def stop(self):
        """Stop the analysis process."""
        self.should_stop = True

    def run(self):
        """Run video analysis with enhanced 5-class detection."""
        if not self.load_model():
            return

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("Failed to open video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0
            detected_clips = []

            print(f"📊 Processing video: {total_frames} frames at {fps} FPS")

            # SMART CLIP-FOCUSED OPTIMIZATION: Use dynamically calculated settings
            frame_skip = self.optimized_frame_skip
            selected_clip_names = [
                self.clip_tags[tag]["name"]
                for tag, enabled in self.selected_clips.items()
                if enabled
            ]

            print(f"🎯 Selected clips: {', '.join(selected_clip_names)}")
            print(f"⏭️ Frame Skip: {frame_skip} frames ({frame_skip/fps:.1f}s intervals)")
            print(f"🚀 Speed optimization: {self.optimization_level}")

            while frame_number < total_frames and not self.should_stop:
                # Skip to next analysis frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress periodically
                if frame_number % self.progress_update_interval == 0:
                    progress = int((frame_number / total_frames) * 100)
                    self.progress_updated.emit(
                        progress, f"Analyzing frame {frame_number}/{total_frames}"
                    )

                try:
                    # Analyze frame with enhanced game analyzer and hybrid OCR
                    current_time = frame_number / fps

                    # DEBUG: Print every 300 frames to confirm we're reaching this point
                    if frame_number % 300 == 0:
                        print(f"🔄 Processing frame {frame_number}, calling analyze_frame...")

                    # SMART OCR OPTIMIZATION: Always run full analysis but filter clips later
                    game_state = self.analyzer.analyze_frame(
                        frame, current_time=current_time, frame_number=frame_number
                    )

                    # DEBUG: Print game state info every 300 frames
                    if frame_number % 300 == 0:
                        print(
                            f"🔍 Frame {frame_number}: GameState={game_state is not None}, Down={getattr(game_state, 'down', 'N/A')}, Distance={getattr(game_state, 'distance', 'N/A')}, Confidence={getattr(game_state, 'confidence', 'N/A')}"
                        )

                        # CRITICAL DEBUG: Save the actual frame and detected regions for manual inspection
                        if game_state and hasattr(game_state, "visualization_layers"):
                            debug_dir = "debug_ocr_regions"
                            import os

                            os.makedirs(debug_dir, exist_ok=True)

                            # Save the original frame
                            cv2.imwrite(f"{debug_dir}/frame_{frame_number}_original.jpg", frame)
                            print(
                                f"💾 Saved original frame to {debug_dir}/frame_{frame_number}_original.jpg"
                            )

                            # Save HUD detection visualization if available
                            if "hud_detection" in game_state.visualization_layers:
                                cv2.imwrite(
                                    f"{debug_dir}/frame_{frame_number}_hud_detection.jpg",
                                    game_state.visualization_layers["hud_detection"],
                                )
                                print(
                                    f"💾 Saved HUD detection to {debug_dir}/frame_{frame_number}_hud_detection.jpg"
                                )

                            print(
                                f"🎯 ACTUAL HUD SHOWS: Look at the saved images to see what the system should be reading vs what it detected"
                            )

                    if game_state:
                        # COMPREHENSIVE PLAY BOUNDARY ANALYSIS
                        boundary_info = self._analyze_play_boundaries(game_state, frame_number)

                        # Enhanced situation detection with hybrid OCR features
                        situation_context = self._analyze_enhanced_situation(
                            game_state, frame_number
                        )

                        # DEBUG: Print situation context every 300 frames
                        if frame_number % 300 == 0:
                            print(
                                f"🧠 Situation Context: Type={getattr(situation_context, 'situation_type', 'N/A')}, Pressure={getattr(situation_context, 'pressure_level', 'N/A')}"
                            )
                            print(
                                f"🎯 Boundary Info: New Play={boundary_info['play_started']}, Type={boundary_info['play_type']}"
                            )

                        # SMART CLIP FILTERING: Check if this situation matches selected clips
                        if self._should_create_clip(game_state, situation_context):
                            print(f"✅ CLIP SHOULD BE CREATED at frame {frame_number}")

                            # INTELLIGENT CLIP CREATION: Use game state changes to determine boundaries
                            # First try intelligent clip boundary detection
                            if hasattr(self.analyzer, "validate_clip_timing"):
                                timing_info = self.analyzer.validate_clip_timing(
                                    game_state, frame_number
                                )
                                if (
                                    timing_info["is_valid"]
                                    and timing_info["start_frame"]
                                    and timing_info["end_frame"]
                                ):
                                    clip_start_frame = timing_info["start_frame"]
                                    clip_end_frame = timing_info["end_frame"]
                                    print(
                                        f"🎯 USING INTELLIGENT BOUNDARIES: {clip_start_frame} → {clip_end_frame} ({(clip_end_frame-clip_start_frame)/fps:.1f}s)"
                                    )
                                else:
                                    # Fallback: Use game state history to find natural boundaries
                                    clip_start_frame, clip_end_frame = (
                                        self._find_natural_clip_boundaries(
                                            frame_number, fps, game_state
                                        )
                                    )

                            # FIXED: Limit clip to current play only
                            if (
                                hasattr(self.analyzer, "game_history")
                                and len(self.analyzer.game_history) > 2
                            ):
                                # Check if we can detect the next play starting
                                recent_history = self.analyzer.game_history[-5:]
                                for i in range(len(recent_history) - 1):
                                    if recent_history[i].down != recent_history[i + 1].down:
                                        # Down changed - play likely ended
                                        frames_ahead = (len(recent_history) - i - 1) * 30
                                        clip_end_frame = min(
                                            clip_end_frame, frame_number + frames_ahead + 30
                                        )
                                        print(
                                            f"🎯 Detected play end {frames_ahead/30:.1f}s ahead - limiting clip"
                                        )
                                        break
                                    print(
                                        f"🔄 USING NATURAL BOUNDARIES: {clip_start_frame} → {clip_end_frame} ({(clip_end_frame-clip_start_frame)/fps:.1f}s)"
                                    )
                            else:
                                # Final fallback: Use game state history
                                clip_start_frame, clip_end_frame = (
                                    self._find_natural_clip_boundaries(
                                        frame_number, fps, game_state
                                    )
                                )
                                print(
                                    f"🔄 USING FALLBACK BOUNDARIES: {clip_start_frame} → {clip_end_frame} ({(clip_end_frame-clip_start_frame)/fps:.1f}s)"
                                )

                            # DUPLICATE PREVENTION: Check if this clip would be a duplicate
                            if not self._is_duplicate_clip(
                                clip_start_frame, clip_end_frame, frame_number
                            ):
                                print(
                                    f"🎬 CREATING IMMEDIATE CLIP: Frame {clip_start_frame} → {clip_end_frame}"
                                )

                                # 🚨 CRITICAL DEBUG: Log game state BEFORE clip creation
                                print(f"🚨 PRE-CLIP GAME STATE DEBUG:")
                                print(f"   Down: {getattr(game_state, 'down', 'MISSING')}")
                                print(f"   Distance: {getattr(game_state, 'distance', 'MISSING')}")
                                print(
                                    f"   Yard Line: {getattr(game_state, 'yard_line', 'MISSING')}"
                                )
                                print(f"   Quarter: {getattr(game_state, 'quarter', 'MISSING')}")
                                print(
                                    f"   Game Clock: {getattr(game_state, 'game_clock', 'MISSING')}"
                                )
                                print(
                                    f"   Possession: {getattr(game_state, 'possession_team', 'MISSING')}"
                                )
                                print(
                                    f"   Territory: {getattr(game_state, 'territory', 'MISSING')}"
                                )

                                # Create the clip immediately
                                clip = self._create_enhanced_clip_with_boundaries(
                                    clip_start_frame,
                                    clip_end_frame,
                                    fps,
                                    game_state,
                                    situation_context,
                                    boundary_info,
                                )
                                detected_clips.append(clip)

                                # Register this clip to prevent future duplicates
                                situation_type = getattr(
                                    situation_context,
                                    "situation_type",
                                    boundary_info.get("play_type", "unknown"),
                                )
                                self._register_created_clip(
                                    clip_start_frame, clip_end_frame, situation_type
                                )

                                # Emit clip detected signal
                                print(f"📡 EMITTING CLIP SIGNAL for {situation_type}")
                                self.clip_detected.emit(clip)

                                # Log enhanced detection with boundary info
                                self._log_enhanced_detection_with_boundaries(
                                    game_state, situation_context, frame_number, boundary_info
                                )
                            else:
                                print(f"🚫 SKIPPED DUPLICATE CLIP at frame {frame_number}")

                            # OPTIONAL: FULL PLAY TRACKING (for future enhancement)
                            # Keep the advanced play tracking for future use, but don't rely on it for clip creation
                            if boundary_info["play_started"]:
                                print(
                                    f"🎬 STARTING FULL PLAY TRACKING: {boundary_info['play_type']} at frame {frame_number}"
                                )
                                self.play_boundary_state["current_play_start_frame"] = (
                                    boundary_info["recommended_clip_start"]
                                )
                                self.play_boundary_state["current_play_start_info"] = {
                                    "frame_number": frame_number,
                                    "game_state": game_state,
                                    "situation_context": situation_context,
                                    "boundary_info": boundary_info,
                                    "fps": fps,
                                }

                            elif (
                                boundary_info["play_ended"]
                                and self.play_boundary_state.get("current_play_start_frame")
                                is not None
                            ):
                                print(f"🏁 PLAY ENDED: Resetting tracking state")
                                self.play_boundary_state["current_play_start_frame"] = None
                                self.play_boundary_state["current_play_start_info"] = None

                        # Update previous game state for down change detection
                        self.previous_game_state = game_state
                    else:
                        # DEBUG: Print when game_state is None/False
                        if frame_number % 300 == 0:
                            print(f"❌ Frame {frame_number}: GameState is None or False!")

                except Exception as e:
                    # DEBUG: Catch any exceptions in the analysis
                    if frame_number % 300 == 0:
                        print(f"💥 EXCEPTION at frame {frame_number}: {str(e)}")
                        import traceback

                        traceback.print_exc()

                # Periodic memory cleanup
                if frame_number % self.memory_cleanup_interval == 0:
                    self.cleanup_memory()

                # PERFORMANCE FIX: Skip frames instead of processing every single one
                frame_number += frame_skip

            # Final cleanup
            self.cleanup_memory(cap)

            # Emit completion signal
            self.analysis_finished.emit("Analysis complete", detected_clips)

        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_occurred.emit(error_msg)
            self.cleanup_memory(cap)

    def _analyze_enhanced_situation(self, game_state, frame_number):
        """Analyze enhanced situation context using hybrid OCR features."""
        try:
            # Use the analyzer's advanced situational intelligence
            situation_context = self.analyzer.analyze_advanced_situation(game_state)

            # Add frame-specific context
            situation_context.frame_number = frame_number
            situation_context.timestamp = frame_number / 30.0  # Assume 30 FPS

            return situation_context

        except Exception as e:
            print(f"⚠️ Enhanced situation analysis error: {e}")
            # Return basic context as fallback
            from spygate.ml.enhanced_game_analyzer import SituationContext

            return SituationContext()

    def _check_enhanced_situation_match(self, game_state, situation_context):
        """Check if current game state matches user preferences with enhanced context."""
        if not game_state:
            return False

        # Extract situation from game state
        down = game_state.down
        distance = game_state.distance

        # DEBUG: Log what we're checking (but don't auto-create clips)
        frame_num = getattr(situation_context, "frame_number", 0)
        if frame_num % 600 == 0:  # Every 20 seconds
            print(f"🔍 SITUATION CHECK at frame {frame_num}: Down={down}, Distance={distance}")
            print(
                f"   📋 Active Preferences: {[k for k, v in self.situation_preferences.items() if v]}"
            )

        # Enhanced PAT Detection
        if hasattr(game_state, "is_pat") and game_state.is_pat:
            print(f"🏈 PAT Detected at frame {frame_num}")
            return True  # Always capture PAT situations

        # Enhanced Penalty Detection
        if situation_context.situation_type == "penalty":
            print(f"🚩 Penalty Detected: {situation_context.pressure_level} pressure")
            return True  # Always capture penalty situations

        # Enhanced Red Zone Detection
        if situation_context.situation_type in ["red_zone", "goal_line"]:
            print(f"🎯 Red Zone/Goal Line: {situation_context.situation_type}")
            return True  # Always capture red zone situations

        # Enhanced Critical Situations
        if situation_context.pressure_level in ["high", "critical"]:
            print(f"⚡ Critical Situation: {situation_context.pressure_level} pressure")
            return True  # Always capture high-pressure situations

        # Standard situation checks with enhanced context
        if down == 1 and self.situation_preferences.get("1st_down", True):
            return True
        if down == 2 and self.situation_preferences.get("2nd_down", False):
            return True
        if down == 3:
            if distance >= 7 and self.situation_preferences.get("3rd_long", True):
                return True
            if self.situation_preferences.get("3rd_down", True):
                return True
        if down == 4 and self.situation_preferences.get("4th_down", True):
            return True
        if distance == 0 and self.situation_preferences.get("goal_line", True):
            return True

        return False

    def _check_situation_match(self, game_state):
        """Legacy method - redirects to enhanced version."""
        from spygate.ml.enhanced_game_analyzer import SituationContext

        return self._check_enhanced_situation_match(game_state, SituationContext())

    def _create_enhanced_clip(self, frame_number, fps, game_state, situation_context):
        """Create an enhanced clip object with intelligent game-state-based boundaries."""

        # 🎯 INTELLIGENT CLIP BOUNDARIES: Use game state changes instead of hardcoded durations
        if hasattr(self.analyzer, "validate_clip_timing"):
            # Use the enhanced game analyzer's intelligent clip boundary detection
            timing_info = self.analyzer.validate_clip_timing(game_state, frame_number)

            if timing_info["is_valid"] and timing_info["start_frame"] and timing_info["end_frame"]:
                # Use game-state-validated boundaries
                start_frame = timing_info["start_frame"]
                end_frame = timing_info["end_frame"]
                base_confidence = timing_info["confidence"]

                print(f"🎯 INTELLIGENT CLIP BOUNDARIES: Using game-state validation")
                print(f"   📊 Validation methods: {timing_info['validation_methods']}")
                print(
                    f"   🎬 Frames: {start_frame} → {end_frame} ({(end_frame-start_frame)/fps:.1f}s duration)"
                )
                print(f"   🎯 Confidence: {base_confidence:.2f}")

            else:
                # Fallback to enhanced situation-based boundaries (better than hardcoded)
                print(
                    f"🔄 FALLBACK: Game-state validation failed, using situation-based boundaries"
                )
                start_frame, end_frame, base_confidence = self._get_situation_based_boundaries(
                    frame_number, fps, situation_context
                )
        else:
            # Fallback if analyzer doesn't have the method
            print(f"⚠️ FALLBACK: Enhanced analyzer not available, using situation-based boundaries")
            start_frame, end_frame, base_confidence = self._get_situation_based_boundaries(
                frame_number, fps, situation_context
            )

        # Boost confidence for special situations
        if situation_context.situation_type in ["pat", "penalty", "red_zone"]:
            base_confidence = min(1.0, base_confidence + 0.05)

        # Format the situation description
        situation_desc = self._format_enhanced_situation(game_state, situation_context)

        # DEBUG: Log the ACTUAL clip creation that's happening
        print(
            f"🎬 ACTUAL CLIP CREATION: Down={game_state.down}, Distance={game_state.distance}, Description='{situation_desc}'"
        )

        return DetectedClip(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / fps,
            end_time=end_frame / fps,
            confidence=base_confidence,
            situation=situation_desc,
        )

    def _find_natural_clip_boundaries(self, frame_number, fps, game_state):
        """FIXED: Find natural clip boundaries that capture exactly one play."""

        # Default: 3 seconds before, 5 seconds after current frame
        default_pre_buffer = int(fps * 3)
        default_post_buffer = int(fps * 5)

        start_frame = max(0, frame_number - default_pre_buffer)
        end_frame = frame_number + default_post_buffer

        # Try to use game history to find actual play boundaries
        if hasattr(self.analyzer, "game_history") and self.analyzer.game_history:
            history = self.analyzer.game_history
            current_idx = len(history) - 1

            # Look backwards for play start (down/distance change)
            play_start_idx = None
            for i in range(current_idx - 1, max(0, current_idx - 30), -1):
                if i >= 0 and i < len(history) - 1:
                    curr_state = history[i]
                    next_state = history[i + 1]

                    # Check for down change or significant game state change
                    if curr_state.down != next_state.down or (
                        curr_state.distance
                        and next_state.distance
                        and abs(curr_state.distance - next_state.distance) > 5
                    ):
                        play_start_idx = i + 1
                        break

            # Look forward for play end (next down change)
            play_end_idx = None
            for i in range(current_idx + 1, min(len(history), current_idx + 30)):
                if i > 0 and i < len(history):
                    prev_state = history[i - 1]
                    curr_state = history[i]

                    # Check for down change indicating play ended
                    if prev_state.down != curr_state.down:
                        play_end_idx = i
                        break

            # Calculate frame numbers based on indices
            if play_start_idx is not None:
                # Assuming ~1 state per second of gameplay
                frames_back = (current_idx - play_start_idx) * fps
                start_frame = max(
                    0, frame_number - int(frames_back) - int(fps * 2)
                )  # 2s pre-buffer
                print(
                    f"📍 Found play start in history: {play_start_idx} ({frames_back/fps:.1f}s ago)"
                )

            if play_end_idx is not None:
                frames_forward = (play_end_idx - current_idx) * fps
                end_frame = frame_number + int(frames_forward) + int(fps * 1)  # 1s post-buffer
                print(
                    f"📍 Found play end in history: {play_end_idx} ({frames_forward/fps:.1f}s ahead)"
                )
            elif play_start_idx is not None:
                # No end found, use reasonable play duration (8 seconds)
                end_frame = start_frame + int(fps * 8)
                print(f"📍 No play end found, using 8-second duration")

        # Ensure reasonable clip duration (3-15 seconds)
        duration = end_frame - start_frame
        min_duration = int(fps * 3)  # 3 seconds minimum
        max_duration = int(fps * 15)  # 15 seconds maximum

        if duration < min_duration:
            end_frame = start_frame + min_duration
        elif duration > max_duration:
            # Trim from the end to keep the most relevant part
            end_frame = start_frame + max_duration

        print(
            f"🎯 Natural boundaries: {start_frame} → {end_frame} ({(end_frame-start_frame)/fps:.1f}s)"
        )
        return start_frame, end_frame

    def _get_situation_based_boundaries(self, frame_number, fps, situation_context):
        """Get clip boundaries based on actual game state changes (NO MORE HARDCODED DURATIONS)."""
        base_confidence = 0.75  # Lower confidence for fallback method

        # 🎯 INTELLIGENT BOUNDARY DETECTION: Use game state history instead of hardcoded times
        if hasattr(self.analyzer, "game_history") and len(self.analyzer.game_history) > 1:
            # Find the last significant game state change
            current_game_state = (
                self.analyzer.game_history[-1] if self.analyzer.game_history else None
            )
            previous_states = (
                self.analyzer.game_history[-10:]
                if len(self.analyzer.game_history) >= 10
                else self.analyzer.game_history[:-1]
            )

            # Look for play start/end markers in recent history
            play_start_frame = None
            play_end_frame = None

            for i, state in enumerate(reversed(previous_states)):
                frames_back = i * int(fps * 0.5)  # Assume states are ~0.5s apart
                state_frame = max(0, frame_number - frames_back)

                # Detect play boundaries from down/distance changes
                if i > 0:
                    prev_state = previous_states[-(i)]
                    if state.down != prev_state.down or state.distance != prev_state.distance:
                        if play_start_frame is None:
                            play_start_frame = state_frame
                        play_end_frame = state_frame

            # Use detected boundaries if found
            if play_start_frame and play_end_frame:
                # Add reasonable buffers around detected play
                buffer_frames = int(fps * 2)  # 2 second buffer
                start_frame = max(0, play_start_frame - buffer_frames)
                end_frame = min(frame_number + buffer_frames, play_end_frame + buffer_frames)
                base_confidence = 0.9  # Higher confidence for detected boundaries
                print(
                    f"🎯 DETECTED PLAY BOUNDARIES: Start={play_start_frame}, End={play_end_frame}"
                )
                print(
                    f"   📍 With buffers: {start_frame} → {end_frame} ({(end_frame-start_frame)/fps:.1f}s)"
                )
                return start_frame, end_frame, base_confidence

        # 🔄 FALLBACK: Use minimal reasonable boundaries (NO HARDCODED LONG DURATIONS)
        print(f"⚠️ NO GAME STATE BOUNDARIES DETECTED - Using minimal fallback")

        # Minimal clip: just enough to capture the moment
        buffer_frames = int(fps * 2.0)  # 1.5 second buffer each side
        start_frame = max(0, frame_number - buffer_frames)
        end_frame = frame_number + int(fps * 3)  # Max 3 seconds after detection

        print(
            f"   📍 Minimal clip: {start_frame} → {end_frame} ({(end_frame-start_frame)/fps:.1f}s)"
        )
        return start_frame, end_frame, base_confidence

    def _create_clip(self, frame_number, fps, game_state):
        """Legacy method - redirects to enhanced version."""
        from spygate.ml.enhanced_game_analyzer import SituationContext

        return self._create_enhanced_clip(frame_number, fps, game_state, SituationContext())

    def _format_enhanced_situation(self, game_state, situation_context):
        """Format game state with enhanced hybrid OCR context for all clip types."""

        # === SCORING SITUATIONS ===
        if hasattr(game_state, "is_touchdown") and game_state.is_touchdown:
            return "🏈 Touchdown"
        if hasattr(game_state, "is_field_goal") and game_state.is_field_goal:
            return "🥅 Field Goal"
        if hasattr(game_state, "is_pat") and game_state.is_pat:
            return "➕ PAT (Point After Touchdown)"
        if hasattr(game_state, "is_safety") and game_state.is_safety:
            return "🛡️ Safety"

        # === GAME SITUATIONS ===
        if hasattr(situation_context, "situation_type"):
            situation_type = situation_context.situation_type

            if situation_type == "penalty":
                return f"🚩 Penalty Situation ({situation_context.pressure_level} pressure)"
            elif situation_type == "turnover":
                return "🔄 Turnover"
            elif situation_type == "sack":
                return "💥 Sack"
            elif situation_type == "two_minute_drill":
                return "⏰ Two Minute Drill"
            elif situation_type == "overtime":
                return "🕐 Overtime"

        # === STRATEGY SITUATIONS ===
        if hasattr(game_state, "play_type"):
            play_type = game_state.play_type

            if play_type == "blitz":
                return "⚡ Blitz"
            elif play_type == "play_action":
                return "🎭 Play Action"
            elif play_type == "screen_pass":
                return "🕸️ Screen Pass"
            elif play_type == "trick_play":
                return "🎪 Trick Play"

        # === PERFORMANCE SITUATIONS ===
        if hasattr(game_state, "yards_gained"):
            yards_gained = game_state.yards_gained
            if yards_gained >= 40:
                return f"💥 Explosive Play ({yards_gained} yards)"
            elif yards_gained >= 20:
                return f"💨 Big Play ({yards_gained} yards)"

        if hasattr(situation_context, "is_three_and_out") and situation_context.is_three_and_out:
            return "🚫 Three and Out"
        if hasattr(situation_context, "play_count") and situation_context.play_count >= 8:
            return f"🚂 Sustained Drive ({situation_context.play_count} plays)"

        # === STANDARD DOWN/DISTANCE FORMATTING ===
        # Check if we have valid down/distance data
        if game_state.down is None or game_state.distance is None:
            # DEBUG: Log when we have missing data
            print(
                f"🔍 CLIP FORMATTING DEBUG: Missing data - Down={game_state.down}, Distance={game_state.distance}"
            )
            return "Unknown Down & Distance"

        down_map = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        down_text = down_map.get(game_state.down, "Unknown")

        # DEBUG: Log the actual values being used for formatting
        print(
            f"🔍 CLIP FORMATTING DEBUG: Down={game_state.down}, Distance={game_state.distance}, Formatted='{down_text} & {game_state.distance}'"
        )

        if game_state.distance == 0:
            base_text = f"{down_text} & Goal"
        else:
            base_text = f"{down_text} & {game_state.distance}"

        # === FIELD POSITION CONTEXT ===
        if hasattr(game_state, "yard_line") and game_state.yard_line:
            if game_state.yard_line <= 10:
                base_text += " 🏁 (Goal Line)"
            elif game_state.yard_line <= 25:
                base_text += " 🎯 (Red Zone)"
            elif game_state.yard_line == 50:
                base_text += " ⚖️ (Midfield)"
            elif game_state.yard_line >= 80:
                base_text += " 🏠 (Deep Territory)"

        # === ENHANCED CONTEXT ===
        if hasattr(situation_context, "situation_type"):
            if situation_context.situation_type == "red_zone":
                base_text += " (Red Zone)"
            elif situation_context.situation_type == "goal_line":
                base_text += " (Goal Line)"
            elif situation_context.situation_type == "two_minute_drill":
                base_text += " (2-Min Drill)"

        # === DOWN-SPECIFIC ENHANCEMENTS ===
        if game_state.down == 3 and hasattr(game_state, "distance") and game_state.distance >= 7:
            base_text += " 📏 (Long)"

        # === PRESSURE INDICATORS ===
        if hasattr(situation_context, "pressure_level"):
            if situation_context.pressure_level == "critical":
                base_text += " ⚡"
            elif situation_context.pressure_level == "high":
                base_text += " 🔥"

        return base_text

    def _log_enhanced_detection(self, game_state, situation_context, frame_number):
        """Log enhanced detection with hybrid OCR details for all clip types."""
        timestamp = frame_number / 30.0  # Assume 30 FPS

        print(f"🎯 Enhanced Detection at {timestamp:.1f}s (Frame {frame_number}):")
        print(f"   Situation: {self._format_enhanced_situation(game_state, situation_context)}")

        # Log pressure and leverage if available
        if hasattr(situation_context, "pressure_level"):
            print(f"   Pressure: {situation_context.pressure_level}")
        if hasattr(situation_context, "leverage_index"):
            print(f"   Leverage: {situation_context.leverage_index:.2f}")

        # === SCORING DETECTIONS ===
        if hasattr(game_state, "is_touchdown") and game_state.is_touchdown:
            print(f"   🏈 Touchdown Detection: CONFIRMED")
        if hasattr(game_state, "is_field_goal") and game_state.is_field_goal:
            print(f"   🥅 Field Goal Detection: CONFIRMED")
        if hasattr(game_state, "is_pat") and game_state.is_pat:
            print(f"   ➕ PAT Detection: CONFIRMED")
        if hasattr(game_state, "is_safety") and game_state.is_safety:
            print(f"   🛡️ Safety Detection: CONFIRMED")

        # === GAME SITUATION DETECTIONS ===
        if hasattr(situation_context, "situation_type"):
            situation_type = situation_context.situation_type

            if situation_type == "penalty":
                print(f"   🚩 Penalty Detection: FLAG + Color Analysis")
            elif situation_type == "turnover":
                print(f"   🔄 Turnover Detection: Possession Change")
            elif situation_type == "sack":
                print(f"   💥 Sack Detection: QB Tackled Behind LOS")
            elif situation_type == "two_minute_drill":
                print(f"   ⏰ Two Minute Drill: Critical Game Time")
            elif situation_type == "overtime":
                print(f"   🕐 Overtime: Extended Play Period")

        # === STRATEGY DETECTIONS ===
        if hasattr(game_state, "play_type"):
            play_type = game_state.play_type

            if play_type == "blitz":
                print(f"   ⚡ Blitz Detection: Extra Pass Rush")
            elif play_type == "play_action":
                print(f"   🎭 Play Action: Fake Handoff")
            elif play_type == "screen_pass":
                print(f"   🕸️ Screen Pass: Delayed Route")
            elif play_type == "trick_play":
                print(f"   🎪 Trick Play: Unconventional Formation")

        # === PERFORMANCE DETECTIONS ===
        if hasattr(game_state, "yards_gained"):
            yards_gained = game_state.yards_gained
            if yards_gained >= 40:
                print(f"   💥 Explosive Play: {yards_gained} yards gained")
            elif yards_gained >= 20:
                print(f"   💨 Big Play: {yards_gained} yards gained")

        if hasattr(situation_context, "is_three_and_out") and situation_context.is_three_and_out:
            print(f"   🚫 Three and Out: Failed Drive")
        if hasattr(situation_context, "play_count") and situation_context.play_count >= 8:
            print(f"   🚂 Sustained Drive: {situation_context.play_count} plays")

    def _create_enhanced_clip_with_boundaries(
        self, start_frame, end_frame, fps, game_state, situation_context, boundary_info
    ):
        """Create an enhanced clip object with intelligent boundary detection."""
        # Ensure minimum and maximum clip durations
        min_duration_frames = int(fps * 3)  # 3 seconds minimum
        max_duration_frames = int(fps * 20)  # 20 seconds maximum

        # Adjust boundaries if needed
        duration = end_frame - start_frame
        if duration < min_duration_frames:
            end_frame = start_frame + min_duration_frames
        elif duration > max_duration_frames:
            end_frame = start_frame + max_duration_frames

        # Ensure we don't go negative
        start_frame = max(0, start_frame)

        # COMPREHENSIVE DEBUG: Log ALL game state values before formatting
        print(f"🔍 COMPREHENSIVE CLIP DEBUG:")
        print(f"   📊 GameState Values:")
        print(f"      Down: {getattr(game_state, 'down', 'MISSING')}")
        print(f"      Distance: {getattr(game_state, 'distance', 'MISSING')}")
        print(f"      Yard Line: {getattr(game_state, 'yard_line', 'MISSING')}")
        print(f"      Quarter: {getattr(game_state, 'quarter', 'MISSING')}")
        print(f"      Game Clock: {getattr(game_state, 'game_clock', 'MISSING')}")
        print(f"      Possession: {getattr(game_state, 'possession_team', 'MISSING')}")
        print(f"      Territory: {getattr(game_state, 'territory', 'MISSING')}")
        print(f"   🎯 Boundary Info:")
        print(f"      Play Type: {boundary_info.get('play_type', 'MISSING')}")
        print(f"      Play Situation: {boundary_info.get('play_situation', 'MISSING')}")
        print(f"      Confidence: {boundary_info.get('confidence', 'MISSING')}")
        print(f"   🧠 Situation Context:")
        print(f"      Type: {getattr(situation_context, 'situation_type', 'MISSING')}")
        print(f"      Pressure: {getattr(situation_context, 'pressure_level', 'MISSING')}")

        # Create enhanced situation description with boundary info
        situation_desc = self._format_enhanced_situation_with_boundaries(
            game_state, situation_context, boundary_info
        )

        # FINAL DEBUG: Log the final description that will be used
        print(f"   🎬 FINAL CLIP DESCRIPTION: '{situation_desc}'")
        print(f"   📍 Clip Frames: {start_frame} → {end_frame}")
        print(f"   ⏱️ Clip Time: {start_frame / fps:.1f}s → {end_frame / fps:.1f}s")

        return DetectedClip(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_frame / fps,
            end_time=end_frame / fps,
            confidence=game_state.confidence if hasattr(game_state, "confidence") else 0.9,
            situation=situation_desc,
        )

    def _format_enhanced_situation_with_boundaries(
        self, game_state, situation_context, boundary_info
    ):
        """Format game state with boundary detection context."""
        # Start with the standard enhanced situation
        base_situation = self._format_enhanced_situation(game_state, situation_context)

        # Add boundary detection context
        play_type = boundary_info.get("play_type", "unknown")

        if play_type == "down_change":
            base_situation += " 🔄 (Down Change)"
        elif play_type == "first_down_achieved":
            base_situation += " 🎯 (First Down)"
        elif play_type == "possession_change":
            base_situation += " 🔄 (Turnover)"
        elif play_type == "big_play":
            base_situation += " 💥 (Big Play)"
        elif play_type == "territory_change":
            base_situation += " 🏈 (Territory Change)"

        # Add special indicators
        if boundary_info.get("first_down_achieved"):
            base_situation += " ✅"
        if boundary_info.get("possession_changed"):
            base_situation += " 🔄"
        if boundary_info.get("significant_yard_gain"):
            base_situation += " 💨"

        return base_situation

    def _log_enhanced_detection_with_boundaries(
        self, game_state, situation_context, frame_number, boundary_info
    ):
        """Log enhanced detection with boundary analysis details."""
        timestamp = frame_number / 30.0  # Assume 30 FPS

        print(f"🎯 BOUNDARY-AWARE Detection at {timestamp:.1f}s (Frame {frame_number}):")
        print(
            f"   Situation: {self._format_enhanced_situation_with_boundaries(game_state, situation_context, boundary_info)}"
        )
        print(f"   Play Type: {boundary_info.get('play_type', 'unknown')}")

        # Log boundary detection details
        if boundary_info.get("play_started"):
            print(f"   🆕 New Play Detected: {boundary_info['play_type']}")
        if boundary_info.get("play_ended"):
            print(f"   🏁 Play Ended")
        if boundary_info.get("play_in_progress"):
            print(f"   🎬 Play in Progress")
        if boundary_info.get("confidence", 0) > 0.8:
            print(f"   ✅ High Confidence: {boundary_info['confidence']:.2f}")

        # Log clip boundaries
        if boundary_info.get("recommended_clip_start"):
            start_time = boundary_info["recommended_clip_start"] / 30.0
            print(f"   📍 Recommended Start: {start_time:.1f}s")
        if boundary_info.get("recommended_clip_end"):
            end_time = boundary_info["recommended_clip_end"] / 30.0
            print(f"   📍 Recommended End: {end_time:.1f}s")

        # Log standard enhanced detection details
        self._log_enhanced_detection(game_state, situation_context, frame_number)

        # === FIELD POSITION ===
        if hasattr(game_state, "yard_line") and game_state.yard_line:
            print(f"   📍 Field Position: {game_state.yard_line} yard line")

        # === DOWN/DISTANCE DETAILS ===
        if hasattr(game_state, "down") and hasattr(game_state, "distance"):
            print(f"   🏈 Down & Distance: {game_state.down} & {game_state.distance}")

        # === SELECTED CLIPS DEBUG ===
        selected_tags = [tag for tag, enabled in self.selected_clips.items() if enabled]
        print(
            f"   🎯 Active Clip Tags: {', '.join(selected_tags[:5])}{'...' if len(selected_tags) > 5 else ''}"
        )

    def _format_situation(self, game_state):
        """Legacy method - redirects to enhanced version."""
        from spygate.ml.enhanced_game_analyzer import SituationContext

        return self._format_enhanced_situation(game_state, SituationContext())

    def detect_team_scores_and_possession(self, frame, hud_box, frame_number=0):
        """Detect team scores and possession from the HUD box."""
        try:
            import re

            import cv2
            import numpy as np
            import pytesseract

            # Extract HUD region
            x1, y1, x2, y2 = map(int, hud_box.xyxy[0].cpu().numpy())
            hud_region = frame[int(y1) : int(y2), int(x1) : int(x2)]

            # Split HUD into left and right sections for team info
            hud_height, hud_width = hud_region.shape[:2]

            # Left team section (away team)
            left_x1 = int(hud_width * 0.05)  # 5% from left
            left_x2 = int(hud_width * 0.30)  # 30% from left

            # Right team section (home team)
            right_x1 = int(hud_width * 0.70)  # 70% from left
            right_x2 = int(hud_width * 0.95)  # 95% from left

            # Vertical range for both sections
            y1_score = int(hud_height * 0.20)  # 20% from top
            y2_score = int(hud_height * 0.80)  # 80% from top

            # Extract team regions
            away_region = hud_region[y1_score:y2_score, left_x1:left_x2]
            home_region = hud_region[y1_score:y2_score, right_x1:right_x2]

            # Process each region
            def process_team_region(region, is_home):
                # Scale up for better OCR
                scale_factor = 8
                scaled = cv2.resize(
                    region,
                    (region.shape[1] * scale_factor, region.shape[0] * scale_factor),
                    interpolation=cv2.INTER_CUBIC,
                )

                # Convert to grayscale
                gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # Denoise
                denoised = cv2.fastNlMeansDenoising(enhanced)

                # Adaptive threshold
                binary = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                )

                # Clean up with morphology
                kernel = np.ones((2, 2), np.uint8)
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # OCR with optimized settings for scores
                score_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
                score_text = pytesseract.image_to_string(cleaned, config=score_config).strip()

                # Extract score (look for 1-2 digit number)
                score_match = re.search(r"\d{1,2}", score_text)
                score = int(score_match.group()) if score_match else None

                return score

            # Get scores for both teams
            away_score = process_team_region(away_region, False)
            home_score = process_team_region(home_region, True)

            # Detect possession triangle
            # Look in the middle section for the triangle
            mid_x1 = int(hud_width * 0.45)  # 45% from left
            mid_x2 = int(hud_width * 0.55)  # 55% from left
            mid_region = hud_region[y1_score:y2_score, mid_x1:mid_x2]

            # Convert to HSV for better triangle detection
            hsv = cv2.cvtColor(mid_region, cv2.COLOR_BGR2HSV)

            # Define yellow color range for the triangle
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Create mask for yellow triangle
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Find contours
            contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            possession = None
            if contours:
                # Get the largest contour (should be the triangle)
                largest_contour = max(contours, key=cv2.contourArea)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Check if triangle points right (home) or left (away)
                # by looking at the center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    # If center is in right half, triangle points right (home possession)
                    possession = "home" if cx > mid_region.shape[1] / 2 else "away"

            return {"home_score": home_score, "away_score": away_score, "possession": possession}

        except Exception as e:
            print(f"⚠️ Score/possession detection error: {e}")
            return {"home_score": None, "away_score": None, "possession": None}


try:
    import cv2
    import numpy as np
    from PIL import Image
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtSvg import QSvgRenderer
    from PyQt6.QtSvgWidgets import QSvgWidget
    from PyQt6.QtWidgets import *

    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class DetectedClip:
    """Represents a detected clip with metadata."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    situation: str
    thumbnail_path: Optional[str] = None
    approved: Optional[bool] = None


class YouTubeStyleClipCard(QWidget):
    """YouTube-style clip card with hover preview and thumbnail"""

    preview_requested = pyqtSignal(int, object)  # index, clip_data
    export_requested = pyqtSignal(int, object)  # index, clip_data

    def __init__(self, index, clip_data, video_path, parent=None):
        super().__init__(parent)
        self.index = index
        self.clip_data = clip_data
        self.video_path = video_path
        self.is_hovered = False
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self.show_hover_preview)

        # Extract clip info
        if hasattr(clip_data, "situation"):
            self.situation = clip_data.situation
            self.start_time = clip_data.start_time
            self.end_time = clip_data.end_time
            print(
                f"🎬 CLIP CARD DEBUG (DetectedClip): Index={index}, Situation='{self.situation}', Start={self.start_time:.1f}s"
            )
        else:
            self.situation = clip_data.get("situation", "Unknown")
            self.start_time = clip_data.get("start_time", 0)
            self.end_time = clip_data.get("end_time", 0)
            print(
                f"🎬 CLIP CARD DEBUG (Dict): Index={index}, Situation='{self.situation}', Start={self.start_time:.1f}s"
            )

        self.duration = self.end_time - self.start_time

        self.setup_ui()
        self.generate_thumbnail()

    def setup_ui(self):
        """Setup the YouTube-style card UI"""
        self.setFixedSize(280, 200)
        self.setStyleSheet(
            """
            YouTubeStyleClipCard {
                background-color: #1a1a1a;
                border-radius: 12px;
                border: 1px solid #333;
            }
            YouTubeStyleClipCard:hover {
                background-color: #2a2a2a;
                border: 1px solid #29d28c;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thumbnail container
        self.thumbnail_container = QLabel()
        self.thumbnail_container.setFixedHeight(140)
        self.thumbnail_container.setStyleSheet(
            """
            QLabel {
                background-color: #2a2a2a;
                border-radius: 8px 8px 0 0;
                border: none;
            }
        """
        )
        self.thumbnail_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_container.setScaledContents(True)
        layout.addWidget(self.thumbnail_container)

        # Info section
        info_widget = QWidget()
        info_widget.setFixedHeight(60)
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(4)

        # Title
        self.title_label = QLabel(f"#{self.index + 1}: {self.situation}")
        self.title_label.setStyleSheet(
            """
            color: #ffffff;
            font-weight: bold;
            font-size: 13px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        self.title_label.setWordWrap(True)
        info_layout.addWidget(self.title_label)

        # Time info
        time_info = QLabel(f"⏱️ {self.start_time:.1f}s • {self.duration:.1f}s duration")
        time_info.setStyleSheet(
            """
            color: #767676;
            font-size: 11px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        info_layout.addWidget(time_info)

        layout.addWidget(info_widget)

        # Set mouse tracking for hover
        self.setMouseTracking(True)

    def generate_thumbnail(self):
        """Generate thumbnail from video at clip start time"""
        try:
            # Check if video path exists
            if not self.video_path or not Path(self.video_path).exists():
                print(f"❌ Video path invalid: {self.video_path}")
                self.create_fallback_thumbnail()
                return

            import cv2

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {self.video_path}")
                self.create_fallback_thumbnail()
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print(f"❌ Invalid FPS: {fps}")
                self.create_fallback_thumbnail()
                cap.release()
                return

            # Seek to middle of clip for better thumbnail
            mid_time = self.start_time + (self.duration / 2)
            frame_number = int(mid_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Convert to QPixmap
                height, width, channel = frame.shape
                if channel == 3:  # RGB
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
                    ).rgbSwapped()

                    pixmap = QPixmap.fromImage(q_image)

                    # Scale to fit thumbnail
                    scaled_pixmap = pixmap.scaled(
                        268,
                        140,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )

                    # Add duration overlay
                    overlay_pixmap = self.add_duration_overlay(scaled_pixmap)
                    self.thumbnail_container.setPixmap(overlay_pixmap)
                else:
                    print(f"❌ Unexpected frame format: {channel} channels")
                    self.create_fallback_thumbnail()
            else:
                print(f"❌ Could not read frame at {frame_number}")
                self.create_fallback_thumbnail()

            cap.release()

        except Exception as e:
            print(f"❌ Thumbnail generation error: {e}")
            self.create_fallback_thumbnail()

    def create_fallback_thumbnail(self):
        """Create a fallback thumbnail when video extraction fails"""
        pixmap = QPixmap(268, 140)
        pixmap.fill(QColor("#2a2a2a"))

        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#767676")))
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🎬\nVideo Clip")

        # Add duration overlay
        overlay_pixmap = self.add_duration_overlay(pixmap)
        self.thumbnail_container.setPixmap(overlay_pixmap)
        painter.end()

    def add_duration_overlay(self, pixmap):
        """Add YouTube-style duration overlay to thumbnail"""
        overlay_pixmap = QPixmap(pixmap)
        painter = QPainter(overlay_pixmap)

        # Duration text
        duration_text = f"{int(self.duration // 60)}:{int(self.duration % 60):02d}"

        # Background for duration
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        text_rect = painter.fontMetrics().boundingRect(duration_text)
        padding = 4
        bg_rect = QRectF(
            pixmap.width() - text_rect.width() - padding * 2 - 8,
            pixmap.height() - text_rect.height() - padding * 2 - 8,
            text_rect.width() + padding * 2,
            text_rect.height() + padding * 2,
        )

        # Draw background
        painter.fillRect(bg_rect, QBrush(QColor(0, 0, 0, 180)))

        # Draw text
        painter.setPen(QPen(QColor("#ffffff")))
        text_pos = QPoint(
            int(bg_rect.x() + padding), int(bg_rect.y() + text_rect.height() + padding)
        )
        painter.drawText(text_pos, duration_text)

        painter.end()
        return overlay_pixmap

    def enterEvent(self, event):
        """Mouse enters the card"""
        self.is_hovered = True
        self.hover_timer.start(500)  # 500ms delay for hover preview
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leaves the card"""
        self.is_hovered = False
        self.hover_timer.stop()
        self.hide_hover_preview()
        super().leaveEvent(event)

    def show_hover_preview(self):
        """Show hover preview (placeholder for now)"""
        if self.is_hovered:
            print(f"🎬 Hover preview for clip #{self.index + 1}")
            # TODO: Implement actual hover preview

    def hide_hover_preview(self):
        """Hide hover preview"""
        pass  # TODO: Implement

    def mousePressEvent(self, event):
        """Handle click to open full preview"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.preview_requested.emit(self.index, self.clip_data)
        super().mousePressEvent(event)


class VideoPlayerDialog(QDialog):
    """Full-featured video player dialog for clip previews"""

    def __init__(self, clip_index, clip_data, video_path, parent=None):
        super().__init__(parent)
        self.clip_index = clip_index
        self.clip_data = clip_data
        self.video_path = video_path

        # Extract clip info
        if hasattr(clip_data, "situation"):
            self.situation = clip_data.situation
            self.start_time = clip_data.start_time
            self.end_time = clip_data.end_time
        else:
            self.situation = clip_data.get("situation", "Unknown")
            self.start_time = clip_data.get("start_time", 0)
            self.end_time = clip_data.get("end_time", 0)

        self.duration = self.end_time - self.start_time
        self.setup_ui()
        self.setup_player()

    def setup_ui(self):
        """Setup the video player UI"""
        self.setWindowTitle(f"Preview Clip #{self.clip_index + 1}: {self.situation}")
        self.setFixedSize(900, 650)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Header
        header = QLabel(f"🎬 {self.situation}")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 20px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 5px;
        """
        )
        layout.addWidget(header)

        # Time info
        time_info = QLabel(
            f"⏱️ {self.start_time:.1f}s - {self.end_time:.1f}s ({self.duration:.1f}s duration)"
        )
        time_info.setStyleSheet(
            """
            color: #767676;
            font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        layout.addWidget(time_info)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setFixedHeight(400)
        self.video_widget.setStyleSheet(
            """
            QVideoWidget {
                background-color: #000000;
                border: 2px solid #565656;
                border-radius: 8px;
            }
        """
        )
        layout.addWidget(self.video_widget)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_btn = QPushButton("▶️")
        self.play_btn.setFixedSize(50, 40)
        self.play_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)

        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #565656;
                height: 6px;
                background: #2a2a2a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #29d28c;
                border: 1px solid #29d28c;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                background: #29d28c;
                border-radius: 3px;
            }
        """
        )
        controls_layout.addWidget(self.progress_slider)

        # Time display
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 0 10px;
        """
        )
        controls_layout.addWidget(self.time_label)

        layout.addLayout(controls_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        # Export button
        export_btn = QPushButton("📥 Export This Clip")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        export_btn.clicked.connect(self.export_clip)
        button_layout.addWidget(export_btn)

        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #767676;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #868686; }
        """
        )
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def setup_player(self):
        """Setup the media player with enhanced error handling"""
        try:
            # Initialize media player components
            self.media_player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)

            # Set up player connections
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)

            # Connect signals with error handling
            self.media_player.positionChanged.connect(self.update_position)
            self.media_player.durationChanged.connect(self.update_duration)
            self.media_player.playbackStateChanged.connect(self.update_play_button)
            self.media_player.errorOccurred.connect(self.handle_media_error)
            self.progress_slider.sliderMoved.connect(self.set_position)

            # Load video with proper path handling
            if self.video_path and Path(self.video_path).exists():
                video_url = QUrl.fromLocalFile(str(Path(self.video_path).resolve()))
                self.media_player.setSource(video_url)
                print(f"🎬 Loading video: {self.video_path}")

                # Set initial position to clip start
                QTimer.singleShot(
                    100, lambda: self.media_player.setPosition(int(self.start_time * 1000))
                )
            else:
                print(f"❌ Video file not found: {self.video_path}")
                self.disable_player()

        except Exception as e:
            print(f"❌ Video player setup error: {e}")
            self.disable_player()

    def disable_player(self):
        """Disable player controls when setup fails"""
        self.play_btn.setEnabled(False)
        self.progress_slider.setEnabled(False)
        self.play_btn.setText("❌")

    def handle_media_error(self, error, error_string):
        """Handle media player errors"""
        print(f"❌ Media player error: {error} - {error_string}")
        self.disable_player()

    def toggle_playback(self):
        """Toggle play/pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            # Ensure we're within clip bounds
            current_pos = self.media_player.position() / 1000.0
            if current_pos < self.start_time or current_pos > self.end_time:
                self.media_player.setPosition(int(self.start_time * 1000))
            self.media_player.play()

    def update_position(self, position):
        """Update position slider and time display"""
        position_sec = position / 1000.0

        # Stop playback if we've exceeded clip end time
        if position_sec > self.end_time:
            self.media_player.pause()
            self.media_player.setPosition(int(self.end_time * 1000))
            return

        # Update slider (relative to clip duration)
        clip_position = position_sec - self.start_time
        clip_progress = (clip_position / self.duration) * 1000 if self.duration > 0 else 0
        self.progress_slider.setValue(int(clip_progress))

        # Update time display
        elapsed = max(0, clip_position)
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        total_str = f"{int(self.duration // 60)}:{int(self.duration % 60):02d}"
        self.time_label.setText(f"{elapsed_str} / {total_str}")

    def update_duration(self, duration):
        """Update duration slider"""
        self.progress_slider.setRange(0, 1000)  # 0-100% in 0.1% increments

    def set_position(self, position):
        """Set playback position from slider"""
        # Convert slider position (0-1000) to actual time within clip
        progress = position / 1000.0
        actual_time = self.start_time + (progress * self.duration)
        self.media_player.setPosition(int(actual_time * 1000))

    def update_play_button(self, state):
        """Update play button text based on playback state"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("⏸️")
        else:
            self.play_btn.setText("▶️")

    def export_clip(self):
        """Export the current clip"""
        # Import export functionality from main window
        if hasattr(self.parent(), "export_clip"):
            self.parent().export_clip(self.clip_index, self.clip_data)
        else:
            print(f"📥 Export functionality not available")

    def closeEvent(self, event):
        """Clean up when dialog closes"""
        try:
            if hasattr(self, "media_player") and self.media_player:
                # Disconnect all signals first
                try:
                    self.media_player.positionChanged.disconnect()
                    self.media_player.durationChanged.disconnect()
                    self.media_player.playbackStateChanged.disconnect()
                    if hasattr(self.media_player, "errorOccurred"):
                        self.media_player.errorOccurred.disconnect()
                except:
                    pass  # Signals may already be disconnected

                # Stop and clear the player
                self.media_player.stop()
                self.media_player.setSource(QUrl())
                self.media_player.setVideoOutput(None)
                self.media_player.setAudioOutput(None)

            # 🎯 CRITICAL FIX: Safer audio output cleanup to prevent Qt freeze
            if hasattr(self, "audio_output") and self.audio_output:
                try:
                    # Set audio output to None first, then delete safely
                    self.audio_output.setDevice(None)
                    self.audio_output = None
                except:
                    # If cleanup fails, just set to None to prevent freeze
                    self.audio_output = None

            print(f"🎬 Video player dialog #{self.clip_index + 1} cleaned up successfully")

        except Exception as e:
            print(f"❌ Error during video player cleanup: {e}")

        # 🎯 CRITICAL: Process events before closing to prevent Qt freeze
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

        super().closeEvent(event)


class HoverableLogoLabel(QLabel):
    """Custom QLabel that changes logo on hover"""

    def __init__(self, default_logo_path, hover_logo_path):
        super().__init__()
        self.default_logo_path = default_logo_path
        self.hover_logo_path = hover_logo_path

        # Load and set default logo
        self.load_logo(self.default_logo_path)
        self.setToolTip("SpygateAI Desktop")

    def load_logo(self, logo_path):
        """Load and set a logo image"""
        try:
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale logo to fit nicely in header (max 180x50)
                scaled_pixmap = pixmap.scaled(
                    180,
                    50,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.setPixmap(scaled_pixmap)
                return True
        except Exception as e:
            print(f"❌ Failed to load logo from {logo_path}: {e}")
        return False

    def enterEvent(self, event):
        """Mouse enters the logo area - show hover logo"""
        self.load_logo(self.hover_logo_path)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leaves the logo area - show default logo"""
        self.load_logo(self.default_logo_path)
        super().leaveEvent(event)


class SpygateDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop")
        self.setGeometry(100, 100, 1400, 900)

        # Make window frameless for custom controls
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Enable window dragging
        self.drag_pos = QPoint()

        # Setup keyboard shortcuts
        self.setup_shortcuts()

        # Set dark background with rounded corners
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: #0b0c0f;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
                border-radius: 12px;
            }}
        """
        )

        self.current_content = "dashboard"  # Track current tab

        # Initialize user database and current user
        self.user_db = UserDatabase()
        self.profile_manager = ProfilePictureManager()
        self.current_user = self.user_db.get_user_by_username("NumbaNines")
        if not self.current_user:
            print("❌ User not found! Creating user...")
            from user_database import setup_demo_user

            self.current_user = setup_demo_user()

        print(
            f"👤 Logged in as: {self.current_user.display_name} ({self.current_user.subscription_type})"
        )

        # Update last login
        self.user_db.update_last_login(self.current_user.user_id)

        # Initialize advanced caching system
        self.cache = get_game_analyzer_cache()
        self.cache_stats_timer = None
        cache_health = self.cache.cache.health_check()
        print(f"🚀 Advanced caching system initialized: {cache_health['overall_status']}")

        # Initialize formation data
        self.players = {}
        self.formation_presets = self.load_formation_presets()

        # Initialize clips storage
        self.detected_clips = []

        # Initialize clip preferences with default selections
        self.clip_preferences = {
            "1st_down": True,
            "2nd_down": False,
            "3rd_down": True,
            "3rd_long": True,
            "4th_down": True,
            "red_zone": True,
            "goal_line": True,
            "midfield": False,
            "deep_territory": False,
            "touchdown": True,
            "field_goal": True,
            "pat": False,
            "safety": True,
            "two_minute_drill": True,
            "overtime": True,
            "penalty": False,
            "turnover": True,
            "sack": True,
            "blitz": False,
            "play_action": False,
            "screen_pass": False,
            "trick_play": True,
            "big_play": True,
            "explosive_play": True,
            "three_and_out": False,
            "sustained_drive": False,
        }

        self.init_ui()

        # Start cache monitoring
        self.start_cache_monitoring()

    def start_cache_monitoring(self):
        """Start monitoring cache performance."""
        from PyQt6.QtCore import QTimer

        self.cache_stats_timer = QTimer()
        self.cache_stats_timer.timeout.connect(self.update_cache_stats)
        self.cache_stats_timer.start(5000)  # Update every 5 seconds
        print("📊 Cache monitoring started")

    def update_cache_stats(self):
        """Update cache statistics for display."""
        try:
            if hasattr(self, "cache"):
                stats = self.cache.cache.get_stats()
                health = self.cache.cache.health_check()
                # Update UI elements if they exist
                if hasattr(self, "cache_status_label"):
                    total_ops = stats.get("hits", 0) + stats.get("misses", 0)
                    self.cache_status_label.setText(
                        f"Status: {health['overall_status'].title()}\n"
                        f"Hit Rate: {stats['hit_rate']:.1%}\n"
                        f"Operations: {total_ops}"
                    )
        except Exception as e:
            # Silently handle cache monitoring errors to prevent crashes
            pass

    def get_cache_performance_data(self):
        """Get cache performance data for dashboard display."""
        if hasattr(self, "cache"):
            stats = self.cache.cache.get_stats()
            health = self.cache.cache.health_check()
            return {
                "status": health["overall_status"],
                "hit_rate": stats["hit_rate"],
                "total_hits": stats.get("hits", 0),
                "total_misses": stats.get("misses", 0),
                "total_operations": stats.get("hits", 0) + stats.get("misses", 0),
                "cache_size": stats.get("fallback_cache_size", 0),
            }
        return {
            "status": "disabled",
            "hit_rate": 0.0,
            "total_hits": 0,
            "total_misses": 0,
            "total_operations": 0,
            "cache_size": 0,
        }

    def load_formation_presets(self):
        """Load formation presets from JSON file or return defaults"""
        presets_file = Path("assets/formations/formation_presets.json")

        # Default formations if file doesn't exist
        default_formations = {
            "Gun Bunch": {
                "description": "3 WR bunch formation",
                "positions": {
                    "QB": (396, 347),
                    "RB": (448, 348),
                    "WR1": (148, 299),
                    "WR2": (552, 300),
                    "WR3": (594, 309),
                    "TE": (508, 309),
                    "LT": (333, 300),
                    "LG": (366, 300),
                    "C": (400, 300),
                    "RG": (433, 300),
                    "RT": (466, 300),
                },
            },
            "I-Formation": {
                "description": "Traditional I-Formation",
                "positions": {
                    "QB": (396, 347),
                    "RB": (396, 380),
                    "FB": (396, 365),
                    "WR1": (148, 299),
                    "WR2": (644, 299),
                    "TE": (508, 309),
                    "LT": (333, 300),
                    "LG": (366, 300),
                    "C": (400, 300),
                    "RG": (433, 300),
                    "RT": (466, 300),
                },
            },
        }

        try:
            if presets_file.exists():
                with open(presets_file) as f:
                    formations = json.load(f)
                print(f"✅ Loaded formations from: {presets_file}")
                return formations
        except Exception as e:
            print(f"❌ Error loading formations: {e}")

        print("📝 Using default formations")
        return default_formations

    def browse_file(self):
        """Open file browser to select video files for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files (*)",
        )

        if file_path:
            print(f"🎬 Processing video: {file_path}")
            # Start the enhanced video analysis with multi-strategy detection
            self.start_video_analysis(file_path)

    def start_video_analysis(self, video_path):
        """Start the video analysis process."""
        self.video_path = video_path

        # Get clip preferences if they exist
        clip_preferences = getattr(
            self,
            "clip_preferences",
            {
                "1st_down": True,
                "3rd_down": True,
                "3rd_long": True,
                "4th_down": True,
                "red_zone": True,
                "goal_line": True,
                "touchdown": True,
                "turnover": True,
                "two_minute_drill": True,
            },
        )

        # Initialize detected_clips list
        self.detected_clips = []

        self.analysis_worker = AnalysisWorker(video_path, situation_preferences=clip_preferences)
        self.analysis_worker.progress_updated.connect(
            self.update_analysis_progress
        )  # Fixed method name
        self.analysis_worker.analysis_finished.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.clip_detected.connect(self.on_clip_detected)
        self.analysis_worker.start()

        # Create and show analysis animation
        self.create_analysis_animation()
        self.center_animation_overlay()
        self.animate_images()

    def create_analysis_animation(self):
        """Create analysis animation overlay"""
        # Create animation overlay
        self.animation_overlay = QLabel(self)
        self.animation_overlay.setFixedSize(120, 120)
        self.animation_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.animation_overlay.setScaledContents(True)
        self.animation_overlay.setStyleSheet(
            """
            QLabel {
                background-color: rgba(26, 26, 26, 200);
                border-radius: 60px;
                border: 3px solid #29d28c;
                font-size: 48px;
            }
        """
        )

        # Create analyzing text
        self.analyzing_text = QLabel("🎯 Analyzing Video...", self)
        self.analyzing_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyzing_text.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                background-color: rgba(26, 26, 26, 200);
                border-radius: 8px;
                padding: 10px 20px;
            }
        """
        )
        self.analyzing_text.adjustSize()

        # Create stop button
        self.stop_button = QPushButton("🛑 Stop Analysis", self)
        self.stop_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ff4757;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #ff3838; }
        """
        )
        self.stop_button.clicked.connect(self.stop_analysis)

        # Load animation images or use emoji fallback
        self.animation_images = []
        self.animation_index = 0

        # Try to load animation images
        animation_paths = [
            "assets/analysis_1.png",
            "assets/analysis_2.png",
            "assets/analysis_3.png",
            "assets/analysis_4.png",
        ]

        for path in animation_paths:
            if os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    self.animation_images.append(
                        pixmap.scaled(
                            80,
                            80,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )

        # Fallback to emoji if no images found
        if not self.animation_images:
            self.animation_overlay.setText("🏈")
        else:
            self.animation_overlay.setPixmap(self.animation_images[0])

        # Show elements
        self.animation_overlay.show()
        self.analyzing_text.show()
        self.stop_button.show()

        # Start animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_images)
        self.animation_timer.start(500)  # 500ms interval

        # Bring to front
        self.animation_overlay.raise_()
        self.analyzing_text.raise_()
        self.stop_button.raise_()

    def center_animation_overlay(self):
        """Center the animation overlay, text, and stop button on the main window"""
        if (
            hasattr(self, "animation_overlay")
            and hasattr(self, "analyzing_text")
            and hasattr(self, "stop_button")
        ):
            # Get the center position of the main window
            main_rect = self.rect()
            animation_size = self.animation_overlay.size()
            text_size = self.analyzing_text.size()
            button_size = self.stop_button.size()

            # Calculate center position for animation
            anim_x = (main_rect.width() - animation_size.width()) // 2
            anim_y = (main_rect.height() - animation_size.height()) // 2

            # Calculate position for text (above animation with 10px gap)
            text_x = (main_rect.width() - text_size.width()) // 2
            text_y = anim_y - text_size.height() - 10

            # Calculate position for stop button (below animation with 20px gap)
            button_x = (main_rect.width() - button_size.width()) // 2
            button_y = anim_y + animation_size.height() + 20

            # Position all elements
            self.animation_overlay.move(anim_x, anim_y)
            self.analyzing_text.move(text_x, text_y)
            self.stop_button.move(button_x, button_y)
            print(f"🎬 Animation centered at ({anim_x}, {anim_y})")
            print(f"🎬 Text positioned at ({text_x}, {text_y})")
            print(f"🎬 Stop button positioned at ({button_x}, {button_y})")

    def animate_images(self):
        """Cycle through the 4 animation images"""
        if hasattr(self, "animation_overlay") and self.animation_overlay.isVisible():
            if self.animation_images:
                # Move to next image
                self.animation_index = (self.animation_index + 1) % len(self.animation_images)
                self.animation_overlay.setPixmap(self.animation_images[self.animation_index])

                # Make sure animation stays visible and on top
                self.animation_overlay.raise_()

                print(f"🎬 Showing image {self.animation_index + 1}")
            # No emoji fallback - if no images found, animation won't show

    def resizeEvent(self, event):
        """Handle window resize to keep animation centered"""
        super().resizeEvent(event)
        if hasattr(self, "animation_overlay") and self.animation_overlay.isVisible():
            self.center_animation_overlay()

    def update_analysis_progress(self, progress, message):
        """Update analysis progress with visual indicator"""
        # Progress is shown through the animated clock overlay
        print(f"📊 Analysis progress: {progress}% - {message}")

        # Update progress text if it exists
        if hasattr(self, "analyzing_text"):
            self.analyzing_text.setText(f"🎯 Analyzing Video... {progress}%\n{message}")
            self.analyzing_text.show()
            self.analyzing_text.raise_()

    def on_clip_detected(self, clip_data):
        """Handle detected clip from analysis worker"""

        # COMPREHENSIVE DEBUG: This is where clips are actually being added to the UI!
        print(f"🎬 FINAL CLIP RECEIVED IN DESKTOP APP:")
        print(f"   📊 Clip Data from Worker:")
        print(f"      Situation: '{clip_data.situation}'")
        print(f"      Start Frame: {clip_data.start_frame}")
        print(f"      End Frame: {clip_data.end_frame}")
        print(f"      Start Time: {clip_data.start_time:.1f}s")
        print(f"      End Time: {clip_data.end_time:.1f}s")
        print(f"      Confidence: {clip_data.confidence}")
        print(f"      Thumbnail: {getattr(clip_data, 'thumbnail_path', 'None')}")
        print(f"      Approved: {getattr(clip_data, 'approved', 'None')}")

        # Convert DetectedClip to dictionary format expected by FACEIT app
        clip_dict = {
            "start_frame": clip_data.start_frame,
            "end_frame": clip_data.end_frame,
            "start_time": clip_data.start_time,
            "end_time": clip_data.end_time,
            "situation": clip_data.situation,
            "confidence": clip_data.confidence,
            "timestamp": clip_data.start_time,
        }

        print(f"   🔄 Converted to Dictionary:")
        print(f"      Dict Situation: '{clip_dict['situation']}'")
        print(f"      Dict Start Time: {clip_dict['start_time']:.1f}s")

        self.detected_clips.append(clip_dict)
        print(f"   ✅ Added to detected_clips list (total: {len(self.detected_clips)})")
        print(f"   🎯 This clip will show as: '{clip_data.situation}' in the UI")

    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        print(f"❌ Analysis error: {error_message}")

        # Stop animation and show error
        if hasattr(self, "animation_timer"):
            self.animation_timer.stop()
        if hasattr(self, "animation_overlay"):
            self.animation_overlay.hide()
        if hasattr(self, "analyzing_text"):
            self.analyzing_text.hide()

        # Show error dialog
        QMessageBox.critical(
            self,
            "Analysis Error",
            f"An error occurred during video analysis:\n\n{error_message}",
            QMessageBox.StandardButton.Ok,
        )

    def stop_analysis(self):
        """Stop the video analysis process"""
        print("🛑 Stopping video analysis...")

        # Stop the analysis worker if it exists
        if hasattr(self, "analysis_worker") and self.analysis_worker:
            if hasattr(self.analysis_worker, "stop"):
                self.analysis_worker.stop()  # Use the stop method from production worker
            self.analysis_worker.terminate()
            self.analysis_worker.wait()  # Wait for thread to finish
            print("🛑 Analysis worker stopped")

        # Stop animation and hide overlays
        if hasattr(self, "animation_timer"):
            self.animation_timer.stop()
        if hasattr(self, "animation_overlay"):
            self.animation_overlay.hide()
        if hasattr(self, "analyzing_text"):
            self.analyzing_text.hide()
            print("🛑 Animation stopped and hidden")

        # Hide stop button during non-analysis times
        if hasattr(self, "stop_button"):
            self.stop_button.hide()

    def _detect_hardware_tier(self) -> str:
        """Detect hardware capabilities for optimal performance."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                if gpu.memoryTotal >= 8000:  # 8GB+ VRAM
                    return "ultra"
                elif gpu.memoryTotal >= 6000:  # 6GB+ VRAM
                    return "high"
                elif gpu.memoryTotal >= 4000:  # 4GB+ VRAM
                    return "medium"
                else:
                    return "low"
            else:
                return "low"
        except:
            # Fallback to CPU detection
            import psutil

            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)

            if cpu_count >= 8 and memory_gb >= 16:
                return "medium"
            elif cpu_count >= 4 and memory_gb >= 8:
                return "low"
            else:
                return "ultra_low"

    def on_analysis_complete(self, message, detected_clips):
        """Handle analysis completion"""
        # Get the video path from the worker
        video_path = (
            self.analysis_worker.video_path if hasattr(self, "analysis_worker") else "Unknown"
        )
        # Stop animation and hide overlays
        if hasattr(self, "animation_timer"):
            self.animation_timer.stop()
        if hasattr(self, "animation_overlay"):
            self.animation_overlay.hide()
        if hasattr(self, "analyzing_text"):
            self.analyzing_text.hide()
        # Hide stop button after analysis is complete
        if hasattr(self, "stop_button"):
            self.stop_button.hide()
            print("🎬 Animation, text, and stop button stopped and hidden")

        # Store clips data for viewing
        self.current_video_path = video_path

        # CRITICAL DEBUG: Log what clips are being received from worker
        print(f"🚨 ANALYSIS COMPLETE - RECEIVED {len(detected_clips)} CLIPS FROM WORKER:")
        for i, clip in enumerate(detected_clips):
            if hasattr(clip, "situation"):
                print(f"   Clip {i+1}: '{clip.situation}' at {clip.start_time:.1f}s")
            else:
                print(
                    f"   Clip {i+1}: '{clip.get('situation', 'UNKNOWN')}' at {clip.get('start_time', 0):.1f}s"
                )

        self.detected_clips = detected_clips

        # Show results
        num_clips = len(self.detected_clips)
        result_msg = (
            f"Analysis complete!\n\nVideo: {Path(video_path).name}\nDetected clips: {num_clips}"
        )

        if num_clips > 0:
            result_msg += f"\n\nClips detected with smart tag-based filtering:\n"
            for i, clip in enumerate(self.detected_clips[:5]):  # Show first 5
                # Handle both DetectedClip objects and dictionaries
                if hasattr(clip, "start_time"):
                    start_time = clip.start_time
                    situation = clip.situation
                else:
                    start_time = clip.get("start_time", 0)
                    situation = clip.get("situation", "Unknown")
                result_msg += f"• {start_time:.1f}s - {situation}\n"
            if num_clips > 5:
                result_msg += f"... and {num_clips - 5} more clips"

            result_msg += "\n\nClick OK to view and export your clips!"

        # Show brief notification and automatically switch to clips viewer
        if num_clips > 0:
            # Automatically switch to analysis tab and show clips
            self.switch_to_tab("analysis")
            self.show_clips_viewer()

            # Show brief success notification
            QMessageBox.information(
                self,
                "Analysis Complete! 🎯",
                result_msg,
                QMessageBox.StandardButton.Ok,
            )
        else:
            # Show no clips found message
            QMessageBox.information(
                self,
                "Analysis Complete",
                result_msg
                + "\n\nTry adjusting your clip selection preferences or check if the video contains gameplay footage.",
                QMessageBox.StandardButton.Ok,
            )

    def show_clips_viewer(self):
        """Show clips viewer with detected clips"""
        print(f"🎬 Showing clips viewer with {len(self.detected_clips)} clips")
        # Refresh the analysis content to show clips
        self.update_main_content()

    def show_play_builder(self):
        """Show the play builder interface"""
        print("🏈 Opening Play Builder...")
        # This would open your formation editor or play builder
        QMessageBox.information(
            self,
            "Play Builder",
            "Play Builder interface will be implemented here.\n\nThis will integrate with your formation editor.",
            QMessageBox.StandardButton.Ok,
        )

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # F11 for fullscreen toggle
        fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        else:
            self.showFullScreen()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout to accommodate header
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Header Bar
        self.create_header_bar(main_layout)

        # Main content layout (3-column like FaceIt)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Left Sidebar
        self.create_left_sidebar(content_layout)

        # Main Content Area
        self.create_main_content(content_layout)

        # Right Sidebar
        self.create_right_sidebar(content_layout)

        # Add content layout to main layout
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

    def create_header_bar(self, parent_layout):
        """Create top header bar with user controls in top-right"""
        header_bar = QFrame()
        header_bar.setFixedHeight(50)
        header_bar.setStyleSheet(
            """
            QFrame {
                background-color: #0b0c0f;
            }
        """
        )

        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 5, 20, 5)
        header_layout.setSpacing(0)

        # Left side - could add breadcrumbs or app title here if needed
        header_layout.addStretch()

        # Right side - Window controls and User controls
        self.create_window_controls(header_layout)
        self.create_user_controls(header_layout)

        parent_layout.addWidget(header_bar)

    def create_window_controls(self, parent_layout):
        """Create custom window control buttons (minimize, maximize, close)"""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(1)

        # Minimize button
        min_btn = QPushButton("−")
        min_btn.setFixedSize(30, 30)
        min_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #565656;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """
        )
        min_btn.setToolTip("Minimize")
        min_btn.clicked.connect(self.showMinimized)
        controls_layout.addWidget(min_btn)

        # Maximize/Restore button
        self.max_btn = QPushButton("□")
        self.max_btn.setFixedSize(30, 30)
        self.max_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #565656;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """
        )
        self.max_btn.setToolTip("Maximize")
        self.max_btn.clicked.connect(self.toggle_maximize)
        controls_layout.addWidget(self.max_btn)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """
        )
        close_btn.setToolTip("Close")
        close_btn.clicked.connect(self.close)
        controls_layout.addWidget(close_btn)

        # Add some spacing before profile picture
        controls_layout.addSpacing(15)

        parent_layout.addLayout(controls_layout)

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.isFullScreen():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        elif self.isMaximized():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        else:
            self.showMaximized()
            self.max_btn.setText("⧉")
            self.max_btn.setToolTip("Restore")

    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if event.buttons() & Qt.MouseButton.LeftButton and not self.drag_pos.isNull():

            # Only allow dragging from the top area (header bar)
            if event.position().y() < 50:  # Header bar height
                if self.isFullScreen():
                    # In fullscreen, restore to normal first then move
                    self.showNormal()
                    self.max_btn.setText("□")
                    self.max_btn.setToolTip("Maximize")
                    # Position window under cursor
                    self.move(event.globalPosition().toPoint() - QPoint(self.width() // 2, 25))
                elif not self.isMaximized():
                    # Normal window dragging
                    self.move(self.pos() + event.globalPosition().toPoint() - self.drag_pos)

                self.drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for window dragging"""
        self.drag_pos = QPoint()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to toggle fullscreen/restore"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Only allow double-click from the top area (header bar)
            if event.position().y() < 50:  # Header bar height
                if self.isFullScreen():
                    self.showNormal()
                    self.max_btn.setText("□")
                    self.max_btn.setToolTip("Maximize")
                elif self.isMaximized():
                    self.showFullScreen()
                else:
                    self.showMaximized()
                    self.max_btn.setText("⧉")
                    self.max_btn.setToolTip("Restore")
        super().mouseDoubleClickEvent(event)

    def create_user_controls(self, parent_layout):
        """Create circular profile picture button"""
        # Get user's profile picture from database
        profile_pic = self.get_display_profile_picture()

        # Create a container with padding to position profile pic lower
        profile_container = QWidget()
        profile_container.setFixedSize(50, 50)
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 5, 0, 5)  # Balanced padding
        profile_layout.setSpacing(0)

        # Circular Profile Picture button
        if is_emoji_profile(self.current_user.profile_picture if self.current_user else "🏈"):
            # Emoji profile picture
            profile_btn = QPushButton(profile_pic)
            profile_btn.setFixedSize(40, 40)
            profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    color: #e3e3e3;
                    border: 2px solid #565656;
                    border-radius: 20px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 20px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )
        else:
            # Custom image profile picture
            profile_btn = QPushButton()
            profile_btn.setFixedSize(40, 40)

            # Load and set custom image
            pixmap = self.load_profile_pixmap(profile_pic)
            if pixmap:
                profile_btn.setIcon(QIcon(pixmap))
                profile_btn.setIconSize(QSize(36, 36))

            profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    border: 2px solid #565656;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )

        profile_btn.setToolTip("Profile & Settings")
        profile_btn.clicked.connect(self.show_settings_dialog)

        # Add profile button to container
        profile_layout.addWidget(profile_btn)
        profile_layout.addStretch()

        # Add container to parent layout
        parent_layout.addWidget(profile_container)

        # Store reference to profile button for later updates
        self.profile_btn = profile_btn

    def get_display_profile_picture(self):
        """Get the profile picture for display (emoji or file path)"""
        if not self.current_user:
            return "🏈"  # Default football emoji

        # If no profile picture set, default to football
        if not self.current_user.profile_picture:
            return "🏈"

        return self.current_user.profile_picture

    def load_profile_pixmap(self, image_path: str) -> Optional[QPixmap]:
        """Load a custom profile picture as a circular QPixmap"""
        try:
            if not Path(image_path).exists():
                return None

            # Load image with PIL for processing
            with Image.open(image_path) as img:
                # Convert to QPixmap
                img_array = np.array(img)
                height, width, channel = img_array.shape
                bytes_per_line = 3 * width

                if channel == 4:  # RGBA
                    q_image = QImage(
                        img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
                    )
                else:  # RGB
                    q_image = QImage(
                        img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
                    )

                pixmap = QPixmap.fromImage(q_image)

                # Create circular mask
                circular_pixmap = QPixmap(40, 40)
                circular_pixmap.fill(Qt.GlobalColor.transparent)

                painter = QPainter(circular_pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setBrush(
                    QBrush(
                        pixmap.scaled(
                            40,
                            40,
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(0, 0, 40, 40)
                painter.end()

                return circular_pixmap

        except Exception as e:
            print(f"❌ Error loading profile picture: {e}")
            return None

    def get_default_profile_picture(self):
        """Get a default profile picture based on user preferences or random selection"""
        # Default profile pictures - various styles
        default_profiles = [
            "🏈",  # Football (main theme)
            "👤",  # Generic user
            "🎯",  # Target (strategy theme)
            "⚡",  # Lightning (speed/power)
            "🔥",  # Fire (intensity)
            "💪",  # Strength
            "🧠",  # Brain (intelligence/strategy)
            "🏆",  # Trophy (winning)
            "⭐",  # Star (excellence)
            "🎮",  # Gaming controller
        ]

        # For now, return the football as default - later this could be user-configurable
        return default_profiles[0]  # "🏈"

    def upload_custom_profile_picture(self):
        """Upload and set a custom profile picture"""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Profile Picture",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;All Files (*)",
        )

        if not file_path:
            return

        print(f"📤 Uploading profile picture: {file_path}")

        # Validate the image
        if not self.profile_manager.validate_image(file_path):
            QMessageBox.warning(
                self,
                "Invalid Image",
                "Please select a valid image file (PNG, JPG, GIF, BMP, or WEBP).",
                QMessageBox.StandardButton.Ok,
            )
            return

        # Process and save the image
        if self.current_user:
            # Clean up old custom profile pictures
            if (
                hasattr(self.current_user, "profile_picture_type")
                and self.current_user.profile_picture_type == "custom"
            ):
                self.profile_manager.cleanup_old_pictures(
                    self.current_user.user_id, self.current_user.profile_picture
                )

            # Process the new image
            processed_path = self.profile_manager.process_image(
                file_path, self.current_user.user_id
            )

            if processed_path:
                # Update database
                success = self.user_db.update_user_profile_picture(
                    self.current_user.user_id, processed_path, "custom"
                )

                if success:
                    # Update current user object
                    self.current_user.profile_picture = processed_path
                    self.current_user.profile_picture_type = "custom"

                    # Update UI
                    self.refresh_profile_button()
                    print(f"✅ Custom profile picture uploaded successfully!")

                    QMessageBox.information(
                        self,
                        "Success!",
                        "Your profile picture has been updated successfully!",
                        QMessageBox.StandardButton.Ok,
                    )
                else:
                    print("❌ Failed to update profile picture in database")
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to save your profile picture. Please try again.",
                        QMessageBox.StandardButton.Ok,
                    )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to process your image. Please try a different image.",
                    QMessageBox.StandardButton.Ok,
                )

    def update_profile_picture(self, new_pic, pic_type="emoji"):
        """Update the profile picture button and save to database"""
        # Save to database if user is logged in
        if self.current_user:
            # Clean up old custom profile pictures if switching to emoji
            if (
                pic_type == "emoji"
                and hasattr(self.current_user, "profile_picture_type")
                and self.current_user.profile_picture_type == "custom"
            ):
                self.profile_manager.cleanup_old_pictures(self.current_user.user_id)

            success = self.user_db.update_user_profile_picture(
                self.current_user.user_id, new_pic, pic_type
            )

            if success:
                # Update current user object
                self.current_user.profile_picture = new_pic
                self.current_user.profile_picture_type = pic_type

                # Update UI
                self.refresh_profile_button()
                print(f"✅ Profile picture updated to {new_pic}")
            else:
                print("❌ Failed to update profile picture in database")

    def refresh_profile_button(self):
        """Refresh the profile button with current user's picture"""
        if not hasattr(self, "profile_btn"):
            return

        profile_pic = self.get_display_profile_picture()

        if is_emoji_profile(self.current_user.profile_picture if self.current_user else "🏈"):
            # Update to emoji
            self.profile_btn.setText(profile_pic)
            self.profile_btn.setIcon(QIcon())  # Clear any icon
            self.profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    color: #e3e3e3;
                    border: 2px solid #565656;
                    border-radius: 20px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 20px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )
        else:
            # Update to custom image
            self.profile_btn.setText("")  # Clear text
            pixmap = self.load_profile_pixmap(profile_pic)
            if pixmap:
                self.profile_btn.setIcon(QIcon(pixmap))
                self.profile_btn.setIconSize(QSize(36, 36))

            self.profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    border: 2px solid #565656;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )

    def show_settings_dialog(self):
        """Show profile dropdown menu"""
        print("👤 Opening profile dropdown...")

        # Create dropdown menu
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 8px 0px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                min-width: 280px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 12px 20px;
                margin: 2px 8px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2a2a2a;
                color: #29d28c;
            }
            QMenu::item:pressed {
                background-color: #29d28c;
                color: #151515;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2a2a2a;
                margin: 8px 16px;
            }
        """
        )

        # Profile section with real user data
        display_name = self.current_user.display_name if self.current_user else "Guest"
        username = f"@{self.current_user.username}" if self.current_user else "@guest"

        profile_section = QLabel(display_name)
        profile_section.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 12px 20px 4px 20px;
            }
        """
        )

        # User info with subscription status
        subscription_status = ""
        if self.current_user and self.current_user.is_premium:
            subscription_status = f" • {self.current_user.subscription_type.upper()}"

        user_info = QLabel(f"{username}{subscription_status}")
        user_info.setStyleSheet(
            """
            QLabel {
                color: #767676;
                font-size: 12px;
                padding: 0px 20px 8px 20px;
            }
        """
        )

        # Create custom widget actions for labels
        profile_widget_action = QWidgetAction(menu)
        profile_widget_action.setDefaultWidget(profile_section)
        menu.addAction(profile_widget_action)

        user_widget_action = QWidgetAction(menu)
        user_widget_action.setDefaultWidget(user_info)
        menu.addAction(user_widget_action)

        menu.addSeparator()

        # Profile Picture submenu
        profile_pic_menu = menu.addMenu("🖼️ Change Profile Picture")
        profile_pic_menu.setStyleSheet(menu.styleSheet())  # Inherit parent style

        # Upload custom picture option
        upload_action = profile_pic_menu.addAction("📤 Upload Custom Picture...")
        upload_action.triggered.connect(self.upload_custom_profile_picture)

        profile_pic_menu.addSeparator()

        # Default emoji options
        default_profiles = self.profile_manager.get_default_emoji_profiles()

        for pic, name in default_profiles:
            action = profile_pic_menu.addAction(f"{pic} {name}")
            action.triggered.connect(lambda checked, p=pic: self.update_profile_picture(p, "emoji"))

        menu.addSeparator()

        # Account management with subscription-specific options
        if self.current_user and not self.current_user.is_premium:
            upgrade_action = menu.addAction("⭐ Upgrade to Premium")
            upgrade_action.triggered.connect(self.show_upgrade_info)
        else:
            manage_sub_action = menu.addAction("⭐ Manage Subscription")
            manage_sub_action.triggered.connect(self.show_subscription_info)

        profile_action = menu.addAction("👤 Manage Profile")
        profile_action.triggered.connect(self.show_profile_info)

        purchases_action = menu.addAction("💳 Purchases and memberships")
        purchases_action.triggered.connect(self.show_purchases_info)

        menu.addSeparator()

        # App settings
        settings_action = menu.addAction("⚙️ Settings")
        settings_action.triggered.connect(self.show_app_settings)

        help_action = menu.addAction("❓ Help")
        help_action.triggered.connect(self.show_help)

        feedback_action = menu.addAction("📝 Send feedback")
        feedback_action.triggered.connect(self.show_feedback)

        menu.addSeparator()

        # Sign out
        signout_action = menu.addAction("🚪 Sign out")
        signout_action.triggered.connect(self.sign_out)

        # Position menu to appear below and aligned to the right edge of the profile button
        # This keeps it within the app window bounds
        button_global_pos = self.profile_btn.mapToGlobal(self.profile_btn.rect().bottomRight())
        menu_pos = QPoint(
            button_global_pos.x() - 280, button_global_pos.y() + 5
        )  # 280px is menu width
        menu.exec(menu_pos)

    def show_upgrade_info(self):
        """Show upgrade information"""
        print("⭐ Showing upgrade options...")

        # Create upgrade dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Upgrade to Premium")
        dialog.setFixedSize(450, 500)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34e89a;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("🏆 Upgrade to SpygateAI Premium")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Features list
        features_text = """
        ✅ Unlimited video analysis
        ✅ Advanced AI coaching insights
        ✅ Custom formation builder
        ✅ Export clips and highlights
        ✅ Priority customer support
        ✅ Beta features access
        """

        features_label = QLabel(features_text)
        features_label.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(features_label)

        # Pricing
        price_label = QLabel("Only $19.99/month")
        price_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #29d28c; text-align: center;"
        )
        price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(price_label)

        # Buttons
        button_layout = QHBoxLayout()

        upgrade_btn = QPushButton("Upgrade Now")
        upgrade_btn.clicked.connect(lambda: self.process_upgrade(dialog))
        button_layout.addWidget(upgrade_btn)

        cancel_btn = QPushButton("Maybe Later")
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        cancel_btn.clicked.connect(dialog.close)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def process_upgrade(self, dialog):
        """Process the upgrade to premium"""
        if self.current_user:
            # Create premium subscription
            subscription_id = self.user_db.create_subscription(
                self.current_user.user_id, "premium", 19.99, 12
            )

            # Refresh current user data
            self.current_user = self.user_db.get_user_by_id(self.current_user.user_id)

            print(f"✅ Upgraded to Premium! Subscription ID: {subscription_id}")

            # Show success message
            QMessageBox.information(
                self,
                "Upgrade Successful!",
                "🎉 Welcome to SpygateAI Premium!\n\nYou now have access to all premium features.",
                QMessageBox.StandardButton.Ok,
            )

            dialog.close()

    def show_subscription_info(self):
        """Show subscription management information"""
        print("⭐ Showing subscription management...")

        if not self.current_user:
            return

        # Get subscription details
        subscription_status = self.user_db.check_subscription_status(self.current_user.user_id)

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Subscription")
        dialog.setFixedSize(450, 400)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("📋 Subscription Details")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Current plan
        plan_info = f"""
        Current Plan: {subscription_status.get('plan', 'Unknown').upper()}
        Status: {subscription_status.get('status', 'Unknown').upper()}
        """

        if subscription_status.get("expires_at"):
            from datetime import datetime

            expires_date = datetime.fromisoformat(
                subscription_status["expires_at"].replace("Z", "")
            )
            plan_info += f"Expires: {expires_date.strftime('%B %d, %Y')}"

        plan_label = QLabel(plan_info)
        plan_label.setStyleSheet(
            "font-size: 14px; background-color: #2a2a2a; padding: 15px; border-radius: 8px;"
        )
        layout.addWidget(plan_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_profile_info(self):
        """Show profile management"""
        print("👤 Showing profile management...")

        dialog = QDialog(self)
        dialog.setWindowTitle("Profile Settings")
        dialog.setFixedSize(400, 300)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        if self.current_user:
            profile_info = f"""
            Username: {self.current_user.username}
            Display Name: {self.current_user.display_name}
            Email: {self.current_user.email}
            Member Since: {self.current_user.created_at[:10]}
            Last Login: {self.current_user.last_login[:10]}
            """

            info_label = QLabel(profile_info)
            info_label.setStyleSheet(
                "font-size: 14px; background-color: #2a2a2a; padding: 15px; border-radius: 8px;"
            )
            layout.addWidget(info_label)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_purchases_info(self):
        """Show purchases and subscriptions"""
        print("💳 Showing purchases...")

        dialog = QDialog(self)
        dialog.setWindowTitle("Purchase History")
        dialog.setFixedSize(500, 400)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("💳 Purchase History")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        if self.current_user:
            subscriptions = self.user_db.get_user_subscriptions(self.current_user.user_id)

            if subscriptions:
                for sub in subscriptions:
                    sub_info = f"""
                    Plan: {sub.plan_type.upper()}
                    Price: ${sub.price_paid:.2f}
                    Started: {sub.started_at[:10]}
                    Status: {sub.status.upper()}
                    """

                    sub_label = QLabel(sub_info)
                    sub_label.setStyleSheet(
                        "font-size: 12px; background-color: #2a2a2a; padding: 10px; border-radius: 6px; margin-bottom: 5px;"
                    )
                    layout.addWidget(sub_label)
            else:
                no_purchases = QLabel("No purchase history found.")
                no_purchases.setStyleSheet("font-size: 14px; color: #767676; text-align: center;")
                layout.addWidget(no_purchases)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_app_settings(self):
        """Show application settings"""
        print("⚙️ Showing app settings...")

    def show_help(self):
        """Show help information"""
        print("❓ Showing help...")

    def show_feedback(self):
        """Show feedback form"""
        print("📝 Showing feedback form...")

    def sign_out(self):
        """Sign out user"""
        print("🚪 Signing out...")

    def create_left_sidebar(self, parent_layout):
        # Left sidebar frame
        left_sidebar = QFrame()
        left_sidebar.setFixedWidth(250)
        left_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Header area
        header_widget = QWidget()
        header_widget.setFixedHeight(50)
        header_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: #0b0c0f;
            }}
        """
        )
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 5, 20, 5)
        header_layout.setSpacing(0)

        # Try to load custom logo, fallback to text logo
        logo_widget = self.create_logo_widget()
        header_layout.addWidget(logo_widget)
        header_layout.addStretch()

        sidebar_layout.addWidget(header_widget)

        # Navigation items (SpygateAI specific)
        nav_items = [
            ("", "Dashboard"),
            ("", "Analysis"),
            ("", "Gameplan"),
            ("", "Learn"),
            ("", "Clips"),
            ("", "Stats"),
            ("🔍", "Debug"),
        ]

        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(5)

        # Store nav buttons for selection management
        self.nav_buttons = []

        for icon, text in nav_items:
            nav_button = self.create_nav_button(icon, text)
            nav_layout.addWidget(nav_button)
            self.nav_buttons.append(nav_button)

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_widget)

        parent_layout.addWidget(left_sidebar)

    def create_logo_widget(self):
        """Create hoverable logo widget - tries to load image logos, falls back to text"""
        # Define paths for default and hover logos
        default_logo_paths = [
            "assets/logo/spygate-logo.png",
            "assets/logo/spygate_logo.png",
            "assets/spygate-logo.png",
            "logo.png",
        ]

        hover_logo_path = "assets/logo/SpygateAI2.png"

        # Try to find default logo
        default_logo = None
        for logo_path in default_logo_paths:
            if Path(logo_path).exists():
                default_logo = logo_path
                break

        # Check if both logos exist for hover functionality
        if default_logo and Path(hover_logo_path).exists():
            try:
                # Create hoverable logo widget
                logo_label = HoverableLogoLabel(default_logo, hover_logo_path)
                print(f"✅ Loaded hoverable logo: {default_logo} → {hover_logo_path}")
                return logo_label

            except Exception as e:
                print(f"❌ Failed to create hoverable logo: {e}")

                # Fallback: try regular logo without hover
                if default_logo:
                    try:
                        logo_label = QLabel()
                        pixmap = QPixmap(default_logo)

                        if not pixmap.isNull():
                            # Scale logo to fit nicely in header (max 180x50)
                            scaled_pixmap = pixmap.scaled(
                                180,
                                50,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                            logo_label.setPixmap(scaled_pixmap)
                            logo_label.setToolTip("SpygateAI Desktop")
                            print(f"✅ Loaded logo from: {default_logo} (no hover)")
                            return logo_label

                    except Exception as e:
                        print(f"❌ Failed to load logo from {default_logo}: {e}")

        # Fallback to text logo
        print("📝 Using text logo (no image found)")
        logo_label = QLabel("🏈 SPYGATE")
        logo_label.setStyleSheet(
            """
            QLabel {
                color: #1ce783;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )
        logo_label.setToolTip("SpygateAI Desktop")
        return logo_label

    def create_nav_button(self, icon, text):
        button = QPushButton(text if not icon else f"{icon}  {text}")
        button.setFixedHeight(45)
        button.setCheckable(True)  # Make button checkable for selected state

        # Set first button (Dashboard) as selected by default
        if text == "Dashboard":
            button.setChecked(True)

        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: #767676;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                text-align: left;
                padding-left: 20px;
                border: none;
                border-radius: 0px;
            }}
            QPushButton:checked {{
                color: #ffffff;
                background-color: #1a1a1a;
            }}
            QPushButton:hover {{
                background-color: #1a1a1a;
                color: #1ce783;
            }}
            QPushButton:pressed {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
        """
        )

        # Connect button click to handle selection
        button.clicked.connect(lambda: self.handle_nav_selection(button, text.lower()))
        return button

    def handle_nav_selection(self, selected_button, content_type):
        """Handle navigation tab selection - only one tab selected at a time"""
        for button in self.nav_buttons:
            button.setChecked(False)
        selected_button.setChecked(True)

        # Update main content based on selection
        self.current_content = content_type
        self.update_main_content()
        self.update_right_sidebar()  # Update right sidebar based on current tab

    def create_main_content(self, parent_layout):
        # Main content area
        self.main_content = QFrame()
        self.main_content.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
            }}
        """
        )

        self.content_layout = QVBoxLayout(self.main_content)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(20)

        # Initial content (Analysis)
        self.update_main_content()

        parent_layout.addWidget(self.main_content, 1)  # Takes remaining space

    def update_main_content(self):
        """Update main content based on current selection"""
        # Stop zoom timer if it exists before clearing content
        if hasattr(self, "zoom_timer") and self.zoom_timer.isActive():
            self.zoom_timer.stop()

        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        if self.current_content == "analysis":
            content_widget = self.create_analysis_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "dashboard":
            content_widget = self.create_dashboard_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "gameplan":
            content_widget = self.create_gameplan_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "learn":
            content_widget = self.create_learn_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "debug":
            content_widget = self.create_debug_content()
            self.content_layout.addWidget(content_widget)
        else:
            content_widget = self.create_default_content()
            self.content_layout.addWidget(content_widget)

    def create_analysis_content(self):
        """Create the analysis tab content - either clips viewer or upload interface"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Check if we have detected clips to show
        if hasattr(self, "detected_clips") and self.detected_clips:
            return self.create_clips_viewer_content()
        else:
            return self.create_upload_interface_content()

    def create_clips_viewer_content(self):
        """Create YouTube-style clips viewer with thumbnail grid"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header with video info
        header_layout = QHBoxLayout()

        header = QLabel(f"🎬 Detected Clips: {len(self.detected_clips)} plays found")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        header_layout.addWidget(header)

        header_layout.addStretch()

        # View controls
        grid_view_btn = QPushButton("⊞ Grid View")
        grid_view_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                margin-right: 8px;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        header_layout.addWidget(grid_view_btn)

        # Export all button
        export_all_btn = QPushButton("📥 Export All")
        export_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                margin-right: 8px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        header_layout.addWidget(export_all_btn)

        # New video button
        new_video_btn = QPushButton("📤 New Video")
        new_video_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #767676;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #868686; }
        """
        )
        new_video_btn.clicked.connect(self.browse_file)
        header_layout.addWidget(new_video_btn)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Video info
        if hasattr(self, "current_video_path"):
            video_info = QLabel(f"📁 {Path(self.current_video_path).name}")
            video_info.setStyleSheet(
                """
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
                padding: 5px 0;
            """
            )
            layout.addWidget(video_info)

        # YouTube-style grid in scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar::vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #565656;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #29d28c;
            }
        """
        )

        # Grid container
        clips_widget = QWidget()
        grid_layout = QGridLayout(clips_widget)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(0, 10, 15, 0)

        # Add clips in 3-column grid
        if self.detected_clips:
            columns = 3
            for i, clip in enumerate(self.detected_clips):
                row = i // columns
                col = i % columns

                # Create YouTube-style clip card
                clip_card = YouTubeStyleClipCard(
                    i, clip, getattr(self, "current_video_path", ""), self
                )
                clip_card.preview_requested.connect(self.open_video_preview)
                clip_card.export_requested.connect(self.export_clip)

                grid_layout.addWidget(clip_card, row, col)

            # Add stretch to remaining columns in last row
            last_row = (len(self.detected_clips) - 1) // columns
            for col in range(len(self.detected_clips) % columns, columns):
                grid_layout.setColumnStretch(col, 1)

        else:
            # No clips message with better styling
            no_clips_widget = QWidget()
            no_clips_layout = QVBoxLayout(no_clips_widget)
            no_clips_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            icon_label = QLabel("🎬")
            icon_label.setStyleSheet(
                """
                color: #565656;
                font-size: 48px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_clips_layout.addWidget(icon_label)

            no_clips_label = QLabel("No clips detected yet")
            no_clips_label.setStyleSheet(
                """
                color: #ffffff;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 15px 0 5px 0;
            """
            )
            no_clips_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_clips_layout.addWidget(no_clips_label)

            subtitle_label = QLabel("Run video analysis to generate clips")
            subtitle_label.setStyleSheet(
                """
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_clips_layout.addWidget(subtitle_label)

            grid_layout.addWidget(no_clips_widget, 0, 0, 1, 3)

        # Add stretch to push content to top
        grid_layout.setRowStretch(grid_layout.rowCount(), 1)

        scroll_area.setWidget(clips_widget)
        layout.addWidget(scroll_area)

        return content

    def open_video_preview(self, index, clip_data):
        """Open video player dialog for clip preview with resource management"""
        # Prevent multiple dialogs from opening simultaneously
        if hasattr(self, "_active_video_dialog") and self._active_video_dialog:
            print("🎬 Video dialog already open, closing previous one...")
            self._active_video_dialog.close()
            self._active_video_dialog = None

        if not hasattr(self, "current_video_path") or not self.current_video_path:
            QMessageBox.warning(
                self,
                "No Video",
                "No video file is currently loaded. Please analyze a video first.",
                QMessageBox.StandardButton.Ok,
            )
            return

        # Check if video file exists
        if not Path(self.current_video_path).exists():
            QMessageBox.warning(
                self,
                "Video Not Found",
                f"Video file not found:\n{self.current_video_path}",
                QMessageBox.StandardButton.Ok,
            )
            return

        try:
            # Create and show video player dialog
            self._active_video_dialog = VideoPlayerDialog(
                index, clip_data, self.current_video_path, self
            )

            # Connect finished signal to cleanup
            self._active_video_dialog.finished.connect(
                lambda: setattr(self, "_active_video_dialog", None)
            )

            print(f"🎬 Opening video dialog for clip #{index + 1}")
            self._active_video_dialog.exec()

        except Exception as e:
            print(f"❌ Video preview error: {e}")
            self._active_video_dialog = None
            # Fallback to simple preview dialog
            self.show_simple_clip_preview(index, clip_data)

    def show_simple_clip_preview(self, index, clip_data):
        """Simple fallback preview dialog without video player"""
        # Extract clip info
        if hasattr(clip_data, "situation"):
            situation = clip_data.situation
            start_time = clip_data.start_time
            end_time = clip_data.end_time
        else:
            situation = clip_data.get("situation", "Unknown")
            start_time = clip_data.get("start_time", 0)
            end_time = clip_data.get("end_time", 0)

        duration = end_time - start_time

        # Create simple dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Clip #{index + 1}: {situation}")
        dialog.setFixedSize(400, 250)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel(f"🎬 {situation}")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Clip details
        details = QLabel(
            f"""
⏱️ Start Time: {start_time:.1f}s
⏱️ End Time: {end_time:.1f}s
⏱️ Duration: {duration:.1f}s

📁 Video: {Path(self.current_video_path).name if hasattr(self, 'current_video_path') else 'Unknown'}
        """
        )
        details.setStyleSheet(
            """
            color: #ffffff;
            font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
        """
        )
        layout.addWidget(details)

        # Buttons
        button_layout = QHBoxLayout()

        # Open external button
        external_btn = QPushButton("▶️ Open in External Player")
        external_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        external_btn.clicked.connect(lambda: self.play_clip_external(start_time, end_time))
        button_layout.addWidget(external_btn)

        # Export button
        export_btn = QPushButton("📥 Export")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        export_btn.clicked.connect(lambda: self.export_clip(index, clip_data))
        button_layout.addWidget(export_btn)

        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #767676;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #868686; }
        """
        )
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def create_clip_item(self, index, clip):
        """Create a single clip item widget"""
        item = QWidget()
        item.setFixedHeight(80)
        item.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
                border: 1px solid #333;
            }
            QWidget:hover {
                background-color: #2a2a2a;
                border: 1px solid #29d28c;
            }
        """
        )

        layout = QHBoxLayout(item)
        layout.setContentsMargins(15, 10, 15, 10)

        # Clip info
        info_layout = QVBoxLayout()

        # Title with situation - handle both DetectedClip objects and dictionaries
        if hasattr(clip, "situation"):
            situation = clip.situation
            start_time = clip.start_time
            end_time = clip.end_time
        else:
            situation = clip.get("situation", "Play Detection")
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)

        title = QLabel(f"#{index + 1}: {situation}")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-weight: bold;
            font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        info_layout.addWidget(title)

        # Time info
        duration = end_time - start_time
        time_info = QLabel(f"⏱️ {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s duration)")
        time_info.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        info_layout.addWidget(time_info)

        layout.addLayout(info_layout)
        layout.addStretch()

        # Button layout for preview and export
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)

        # Preview button
        preview_btn = QPushButton("👁️ Preview Clip")
        preview_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        preview_btn.clicked.connect(lambda: self.preview_clip(index, clip))
        button_layout.addWidget(preview_btn)

        # Export button
        export_btn = QPushButton("📥 Export Clip")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        export_btn.clicked.connect(lambda: self.export_clip(index, clip))
        button_layout.addWidget(export_btn)

        layout.addLayout(button_layout)

        return item

    def preview_clip(self, index, clip):
        """Preview a specific clip in a dialog window"""
        # Handle both DetectedClip objects and dictionaries
        if hasattr(clip, "situation"):
            situation = clip.situation
            start_time = clip.start_time
            end_time = clip.end_time
        else:
            situation = clip.get("situation", "Unknown")
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)

        print(f"🎬 Previewing clip #{index + 1}: {situation}")

        # Create preview dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview Clip #{index + 1}: {situation}")
        dialog.setFixedSize(800, 600)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header with clip info
        header = QLabel(f"🎬 {situation}")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        layout.addWidget(header)

        # Time info
        duration = end_time - start_time
        time_info = QLabel(f"⏱️ {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s duration)")
        time_info.setStyleSheet(
            """
            color: #767676;
            font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 15px;
        """
        )
        layout.addWidget(time_info)

        # Video preview placeholder (since we don't have full video player integration)
        preview_area = QLabel("🎥 Video Preview\n\n(Full video player integration coming soon)")
        preview_area.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
            background-color: #2a2a2a;
            border: 2px dashed #565656;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
        """
        )
        preview_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_area.setMinimumHeight(300)
        layout.addWidget(preview_area)

        # Control buttons
        button_layout = QHBoxLayout()

        # Play in external player button
        play_external_btn = QPushButton("▶️ Open in External Player")
        play_external_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        play_external_btn.clicked.connect(lambda: self.play_clip_external(start_time, end_time))
        button_layout.addWidget(play_external_btn)

        # Export button
        export_btn = QPushButton("📥 Export This Clip")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        export_btn.clicked.connect(lambda: (self.export_clip(index, clip), dialog.close()))
        button_layout.addWidget(export_btn)

        # Close button
        close_btn = QPushButton("❌ Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #767676;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #868686; }
        """
        )
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def play_clip_external(self, start_time, end_time):
        """Play clip in external video player"""
        if not hasattr(self, "current_video_path"):
            QMessageBox.warning(
                self,
                "No Video",
                "No video file is currently loaded.",
                QMessageBox.StandardButton.Ok,
            )
            return

        try:
            import subprocess
            import sys

            video_path = self.current_video_path

            # Try to open with system default player at specific time
            if sys.platform.startswith("win"):
                # Windows - try VLC or default player
                try:
                    subprocess.run(
                        ["vlc", "--start-time", str(int(start_time)), video_path], check=False
                    )
                except FileNotFoundError:
                    subprocess.run(["start", video_path], shell=True, check=False)
            elif sys.platform.startswith("darwin"):
                # macOS - try with QuickTime Player or default
                subprocess.run(["open", video_path], check=False)
            else:
                # Linux - try common video players
                try:
                    subprocess.run(
                        ["vlc", "--start-time", str(int(start_time)), video_path], check=False
                    )
                except FileNotFoundError:
                    try:
                        subprocess.run(["mpv", "--start", str(start_time), video_path], check=False)
                    except FileNotFoundError:
                        subprocess.run(["xdg-open", video_path], check=False)

            print(f"🎬 Opened video in external player at {start_time}s")

        except Exception as e:
            QMessageBox.warning(
                self,
                "Player Error",
                f"Could not open video in external player:\n\n{str(e)}",
                QMessageBox.StandardButton.Ok,
            )

    def export_clip(self, index, clip):
        """Export a specific clip"""
        # Handle both DetectedClip objects and dictionaries
        if hasattr(clip, "situation"):
            situation = clip.situation
            start_time = clip.start_time
            end_time = clip.end_time
        else:
            situation = clip.get("situation", "Unknown")
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)

        print(f"🎬 Exporting clip #{index + 1}: {situation}")

        # Get output file path from user
        video_name = (
            Path(self.current_video_path).stem if hasattr(self, "current_video_path") else "clip"
        )
        default_name = f"{video_name}_clip_{index + 1}_{situation.replace(' ', '_')}.mp4"

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Clip As", default_name, "Video Files (*.mp4);;All Files (*)"
        )

        if output_path:
            # Show export progress
            progress = QProgressDialog("Exporting clip...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()

            try:
                # Extract clip using ffmpeg (simplified - you may want to use the full worker)
                import subprocess

                # Handle both DetectedClip objects and dictionaries
                if hasattr(clip, "start_time"):
                    start_time = clip.start_time
                    end_time = clip.end_time
                else:
                    start_time = clip.get("start_time", 0)
                    end_time = clip.get("end_time", 0)

                duration = end_time - start_time

                cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output
                    "-i",
                    self.current_video_path,
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",  # Copy streams for speed
                    output_path,
                ]

                # Run ffmpeg
                result = subprocess.run(cmd, capture_output=True, text=True)

                progress.close()

                if result.returncode == 0:
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Clip exported successfully!\n\nSaved to: {output_path}",
                        QMessageBox.StandardButton.Ok,
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        f"Failed to export clip.\n\nError: {result.stderr}",
                        QMessageBox.StandardButton.Ok,
                    )

            except Exception as e:
                progress.close()
                QMessageBox.warning(
                    self,
                    "Export Error",
                    f"An error occurred during export:\n\n{str(e)}",
                    QMessageBox.StandardButton.Ok,
                )

    def create_upload_interface_content(self):
        """Create upload interface when no clips are available"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel("Video Analysis")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 10px 0;
        """
        )
        layout.addWidget(header)

        # YouTube-style upload area
        upload_container = QWidget()
        upload_container.setMaximumWidth(600)
        upload_container.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 12px;
                padding: 40px;
            }
        """
        )

        container_layout = QVBoxLayout(upload_container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload icon
        upload_icon = QLabel("📤")
        upload_icon.setStyleSheet(
            """
            font-size: 48px;
            color: #1ce783;
            font-family: "Minork Sans", sans-serif;
        """
        )
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(upload_icon)

        # Title
        title = QLabel("Upload your Madden gameplay")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        """
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Drag and drop video files here, or click to browse")
        subtitle.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 14px;
            margin-bottom: 20px;
        """
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle)

        # Create main upload content layout
        main_content_layout = QVBoxLayout()

        # Center the upload container (without browse button)
        centered_layout = QHBoxLayout()
        centered_layout.addStretch()
        centered_layout.addWidget(upload_container)
        centered_layout.addStretch()

        upload_widget = QWidget()
        upload_widget.setLayout(centered_layout)
        main_content_layout.addWidget(upload_widget)

        # Add Clip Selection UI
        clip_selection_widget = self.create_clip_selection_ui()
        main_content_layout.addWidget(clip_selection_widget)

        # Add stretch to push browse button down
        main_content_layout.addStretch()

        # Browse button positioned lower (where stop button would be)
        browse_btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        browse_btn.clicked.connect(self.browse_file)

        # Center the browse button horizontally
        browse_btn_layout.addStretch()
        browse_btn_layout.addWidget(browse_btn)
        browse_btn_layout.addStretch()

        browse_btn_widget = QWidget()
        browse_btn_widget.setLayout(browse_btn_layout)
        main_content_layout.addWidget(browse_btn_widget)

        # Add some bottom margin
        main_content_layout.addSpacing(80)

        # Create final container
        final_container = QWidget()
        final_container.setLayout(main_content_layout)
        layout.addWidget(final_container)

        layout.addStretch()
        return content

    def create_clip_selection_ui(self):
        """Create the improved clip selection interface with Select All/Clear All"""
        container = QWidget()
        container.setMaximumWidth(900)
        container.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
            }
        """
        )

        layout = QVBoxLayout(container)

        # Header with controls
        header_layout = QHBoxLayout()

        header = QLabel("🎯 Select Clips to Detect")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 18px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Quick action buttons
        select_all_btn = QPushButton("✅ Select All")
        select_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                margin-right: 8px;
            }
            QPushButton:hover { background-color: #34e89a; }
        """
        )
        select_all_btn.clicked.connect(self.select_all_clips)
        header_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("❌ Clear All")
        clear_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                margin-right: 8px;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        clear_all_btn.clicked.connect(self.clear_all_clips)
        header_layout.addWidget(clear_all_btn)

        # Preset buttons
        preset_btn = QPushButton("⚡ Quick Presets")
        preset_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #767676;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #868686; }
        """
        )
        preset_btn.clicked.connect(self.show_clip_presets)
        header_layout.addWidget(preset_btn)

        layout.addLayout(header_layout)

        # Initialize clip preferences if not exists
        if not hasattr(self, "clip_preferences"):
            self.clip_preferences = {
                "1st_down": True,
                "3rd_down": True,
                "3rd_long": True,
                "4th_down": True,
                "red_zone": True,
                "goal_line": True,
                "touchdown": True,
                "turnover": True,
                "two_minute_drill": True,
            }

        # Store checkboxes for bulk operations
        self.clip_checkboxes = {}

        # Create clip tags dictionary
        self.clip_tags = {
            # Downs
            "1st_down": {"icon": "🥇", "name": "1st Down", "category": "Downs"},
            "2nd_down": {"icon": "🥈", "name": "2nd Down", "category": "Downs"},
            "3rd_down": {"icon": "🥉", "name": "3rd Down", "category": "Downs"},
            "3rd_long": {"icon": "📏", "name": "3rd & Long (7+ yards)", "category": "Downs"},
            "4th_down": {"icon": "🔥", "name": "4th Down", "category": "Downs"},
            # Field Position
            "red_zone": {
                "icon": "🎯",
                "name": "Red Zone (25 yard line)",
                "category": "Field Position",
            },
            "goal_line": {
                "icon": "🏁",
                "name": "Goal Line (10 yard line)",
                "category": "Field Position",
            },
            "midfield": {"icon": "⚖️", "name": "Midfield", "category": "Field Position"},
            "deep_territory": {
                "icon": "🏠",
                "name": "Deep Territory",
                "category": "Field Position",
            },
            # Scoring
            "touchdown": {"icon": "🏈", "name": "Touchdown", "category": "Scoring"},
            "field_goal": {"icon": "🥅", "name": "Field Goal", "category": "Scoring"},
            "pat": {"icon": "➕", "name": "PAT", "category": "Scoring"},
            "safety": {"icon": "🛡️", "name": "Safety", "category": "Scoring"},
            # Game Situations
            "two_minute_drill": {
                "icon": "⏰",
                "name": "Two Minute Drill",
                "category": "Game Situations",
            },
            "overtime": {"icon": "🕐", "name": "Overtime", "category": "Game Situations"},
            "penalty": {"icon": "🚩", "name": "Penalty", "category": "Game Situations"},
            "turnover": {"icon": "🔄", "name": "Turnover", "category": "Game Situations"},
            "sack": {"icon": "💥", "name": "Sack", "category": "Game Situations"},
            # Strategy
            "blitz": {"icon": "⚡", "name": "Blitz", "category": "Strategy"},
            "play_action": {"icon": "🎭", "name": "Play Action", "category": "Strategy"},
            "screen_pass": {"icon": "🕸️", "name": "Screen Pass", "category": "Strategy"},
            "trick_play": {"icon": "🎪", "name": "Trick Play", "category": "Strategy"},
            # Performance
            "big_play": {"icon": "💨", "name": "Big Play (20+ yards)", "category": "Performance"},
            "explosive_play": {
                "icon": "💥",
                "name": "Explosive Play (40+ yards)",
                "category": "Performance",
            },
            "three_and_out": {"icon": "🚫", "name": "Three & Out", "category": "Performance"},
            "sustained_drive": {
                "icon": "🚂",
                "name": "Sustained Drive (8+ plays)",
                "category": "Performance",
            },
        }

        # Group by category
        categories = {}
        for tag_id, tag_info in self.clip_tags.items():
            category = tag_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((tag_id, tag_info))

        # Create tabbed interface for better organization
        from PyQt6.QtWidgets import QTabWidget

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #565656;
                background-color: #2a2a2a;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #e3e3e3;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #29d28c;
                color: #151515;
            }
            QTabBar::tab:hover {
                background-color: #565656;
            }
        """
        )

        # Create tabs for each category
        for category, tags in categories.items():
            tab_widget_content = QWidget()
            tab_layout = QVBoxLayout(tab_widget_content)
            tab_layout.setContentsMargins(15, 15, 15, 15)
            tab_layout.setSpacing(10)

            # Category description
            category_descriptions = {
                "Downs": "Track specific down situations and critical conversion attempts",
                "Field Position": "Monitor field position changes and territory advantages",
                "Scoring": "Capture all scoring plays and point-after attempts",
                "Game Situations": "Detect high-pressure and special game moments",
                "Strategy": "Identify strategic plays and tactical decisions",
                "Performance": "Measure explosive plays and drive efficiency",
            }

            desc_label = QLabel(category_descriptions.get(category, ""))
            desc_label.setStyleSheet(
                """
                color: #767676;
                font-size: 12px;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-bottom: 10px;
            """
            )
            tab_layout.addWidget(desc_label)

            # Create grid layout for checkboxes (4 columns)
            grid_layout = QGridLayout()
            grid_layout.setSpacing(12)

            for i, (tag_id, tag_info) in enumerate(tags):
                row = i // 4
                col = i % 4

                checkbox = QCheckBox(f"{tag_info['icon']} {tag_info['name']}")
                checkbox.setChecked(self.clip_preferences.get(tag_id, False))
                checkbox.setStyleSheet(
                    """
                    QCheckBox {
                        color: #e3e3e3;
                        font-family: 'Minork Sans', Arial, sans-serif;
                        font-size: 12px;
                        spacing: 8px;
                        padding: 6px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border-radius: 4px;
                        border: 2px solid #565656;
                        background-color: #2a2a2a;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #29d28c;
                        border-color: #29d28c;
                    }
                    QCheckBox::indicator:hover {
                        border-color: #29d28c;
                    }
                """
                )

                # Store checkbox reference and connect signal
                self.clip_checkboxes[tag_id] = checkbox
                checkbox.tag_id = tag_id
                checkbox.toggled.connect(
                    lambda checked, cb=checkbox: self.update_clip_preference(cb.tag_id, checked)
                )

                grid_layout.addWidget(checkbox, row, col)

            tab_layout.addLayout(grid_layout)
            tab_layout.addStretch()

            # Add tab with emoji icon
            category_icons = {
                "Downs": "🏈",
                "Field Position": "📍",
                "Scoring": "🎯",
                "Game Situations": "⚡",
                "Strategy": "🧠",
                "Performance": "📊",
            }
            tab_widget.addTab(
                tab_widget_content, f"{category_icons.get(category, '📂')} {category}"
            )

        layout.addWidget(tab_widget)

        # Selection summary and speed info
        summary_layout = QHBoxLayout()

        self.selection_summary = QLabel("0 clips selected")
        self.selection_summary.setStyleSheet(
            """
            color: #29d28c;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        summary_layout.addWidget(self.selection_summary)

        summary_layout.addStretch()

        speed_info = QLabel("💡 Fewer selections = faster analysis")
        speed_info.setStyleSheet(
            """
            color: #767676;
            font-size: 11px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        summary_layout.addWidget(speed_info)

        layout.addLayout(summary_layout)

        # Initialize summary
        self.update_selection_summary()

        # Center the container
        centered_layout = QHBoxLayout()
        centered_layout.addStretch()
        centered_layout.addWidget(container)
        centered_layout.addStretch()

        centered_widget = QWidget()
        centered_widget.setLayout(centered_layout)

        return centered_widget

    def update_clip_preference(self, tag_id, checked):
        """Update clip preference when checkbox is toggled"""
        if not hasattr(self, "clip_preferences"):
            self.clip_preferences = {}
        self.clip_preferences[tag_id] = checked

        # Also update the selected_clips dict used by the analysis worker
        if not hasattr(self, "selected_clips"):
            self.selected_clips = {}
        self.selected_clips[tag_id] = checked

        # Update selection summary
        self.update_selection_summary()
        print(f"🎯 Clip preference updated: {tag_id} = {checked}")

    def update_selection_summary(self):
        """Update the selection summary label"""
        if hasattr(self, "selection_summary"):
            selected_count = sum(1 for checked in self.clip_preferences.values() if checked)
            total_count = len(self.clip_preferences)
            self.selection_summary.setText(f"{selected_count} of {total_count} clips selected")

    def select_all_clips(self):
        """Select all clip preferences"""
        if hasattr(self, "clip_checkboxes"):
            for tag_id, checkbox in self.clip_checkboxes.items():
                checkbox.setChecked(True)
                self.clip_preferences[tag_id] = True
                if not hasattr(self, "selected_clips"):
                    self.selected_clips = {}
                self.selected_clips[tag_id] = True
            self.update_selection_summary()
            print("✅ All clips selected")

    def clear_all_clips(self):
        """Clear all clip preferences"""
        if hasattr(self, "clip_checkboxes"):
            for tag_id, checkbox in self.clip_checkboxes.items():
                checkbox.setChecked(False)
                self.clip_preferences[tag_id] = False
                if not hasattr(self, "selected_clips"):
                    self.selected_clips = {}
                self.selected_clips[tag_id] = False
            self.update_selection_summary()
            print("❌ All clips cleared")

    def show_clip_presets(self):
        """Show quick preset options"""
        from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Quick Presets")
        dialog.setFixedSize(400, 350)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                border-radius: 12px;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("⚡ Quick Clip Presets")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        layout.addWidget(header)

        # Preset options
        presets = {
            "🏈 Essential": ["3rd_down", "4th_down", "red_zone", "touchdown", "turnover"],
            "🎯 Scoring Focus": ["red_zone", "goal_line", "touchdown", "field_goal", "pat"],
            "⚡ High Pressure": [
                "3rd_down",
                "3rd_long",
                "4th_down",
                "two_minute_drill",
                "turnover",
            ],
            "📊 Performance": ["big_play", "explosive_play", "three_and_out", "sustained_drive"],
            "🧠 Strategy": ["blitz", "play_action", "screen_pass", "trick_play"],
            "📍 Field Position": ["red_zone", "goal_line", "midfield", "deep_territory"],
        }

        for preset_name, tags in presets.items():
            btn = QPushButton(preset_name)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2a2a2a;
                    color: #e3e3e3;
                    border: 1px solid #565656;
                    border-radius: 6px;
                    padding: 10px 15px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 12px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #29d28c;
                    color: #151515;
                    border-color: #29d28c;
                }
            """
            )
            btn.clicked.connect(
                lambda checked, preset_tags=tags: self.apply_preset(preset_tags, dialog)
            )
            layout.addWidget(btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #666666; }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def apply_preset(self, preset_tags, dialog=None):
        """Apply a preset selection"""
        # First clear all
        self.clear_all_clips()

        # Then select preset tags
        if hasattr(self, "clip_checkboxes"):
            for tag_id in preset_tags:
                if tag_id in self.clip_checkboxes:
                    self.clip_checkboxes[tag_id].setChecked(True)
                    self.clip_preferences[tag_id] = True
                    if not hasattr(self, "selected_clips"):
                        self.selected_clips = {}
                    self.selected_clips[tag_id] = True

        self.update_selection_summary()
        print(f"⚡ Applied preset with {len(preset_tags)} clips")

        # Close dialog if provided
        if dialog:
            dialog.close()

    def create_dashboard_content(self):
        """Create comprehensive dashboard content"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Welcome header with user info
        welcome_layout = QHBoxLayout()

        welcome_text = (
            f"Welcome back, {self.current_user.display_name if self.current_user else 'Guest'}!"
        )
        welcome_header = QLabel(welcome_text)
        welcome_header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        welcome_layout.addWidget(welcome_header)

        # Premium badge if applicable
        if self.current_user and self.current_user.is_premium:
            premium_badge = QLabel("⭐ PREMIUM")
            premium_badge.setStyleSheet(
                """
                color: #29d28c;
                font-size: 12px;
                font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
                background-color: rgba(41, 210, 140, 0.2);
                padding: 4px 8px;
                border-radius: 4px;
                margin-left: 15px;
            """
            )
            welcome_layout.addWidget(premium_badge)

        welcome_layout.addStretch()

        welcome_widget = QWidget()
        welcome_widget.setLayout(welcome_layout)
        layout.addWidget(welcome_widget)

        # Quick Action Buttons Row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(15)

        # Upload Video Button (primary action)
        upload_btn = QPushButton("Upload New Video")
        upload_btn.setFixedHeight(50)
        upload_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #34e89a;
            }
            QPushButton:pressed {
                background-color: #1fc47d;
            }
        """
        )
        upload_btn.clicked.connect(lambda: self.switch_to_tab("analysis"))
        actions_row.addWidget(upload_btn)

        # Play Builder Button
        play_builder_btn = QPushButton("🏈 Play Builder")
        play_builder_btn.setFixedHeight(50)
        play_builder_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
            border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        play_builder_btn.clicked.connect(lambda: self.switch_to_tab("gameplan"))
        actions_row.addWidget(play_builder_btn)

        # View Analysis Button
        analysis_btn = QPushButton("📊 View Analysis")
        analysis_btn.setFixedHeight(50)
        analysis_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        analysis_btn.clicked.connect(lambda: self.switch_to_tab("analysis"))
        actions_row.addWidget(analysis_btn)

        actions_widget = QWidget()
        actions_widget.setLayout(actions_row)
        layout.addWidget(actions_widget)

        # Stats Cards Row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(15)

        # Get cache performance data
        cache_data = self.get_cache_performance_data()

        # Create stat cards with cache performance
        stat_cards = [
            ("📹", "Videos Analyzed", "23", "#29d28c"),
            ("⚡", "Cache Hit Rate", f"{cache_data['hit_rate']:.1%}", "#1ce783"),
            ("🏈", "Formations Used", "12", "#17d474"),
            ("📈", "Win Rate", "67%", "#34e89a"),
        ]

        for icon, title, value, color in stat_cards:
            card = self.create_stat_card(icon, title, value, color)
            stats_row.addWidget(card)

        stats_widget = QWidget()
        stats_widget.setLayout(stats_row)
        layout.addWidget(stats_widget)

        # Content Row (Recent Activity + Performance Charts)
        content_row = QHBoxLayout()
        content_row.setSpacing(20)

        # Recent Activity Panel
        recent_activity = self.create_recent_activity_panel()
        content_row.addWidget(recent_activity, 1)

        # Performance Summary Panel
        performance_panel = self.create_performance_panel()
        content_row.addWidget(performance_panel, 1)

        content_widget = QWidget()
        content_widget.setLayout(content_row)
        layout.addWidget(content_widget)

        # Premium Features Showcase (if premium user)
        if self.current_user and self.current_user.is_premium:
            premium_panel = self.create_premium_features_panel()
            layout.addWidget(premium_panel)

        layout.addStretch()
        return content

    def create_stat_card(self, icon, title, value, color):
        """Create a stat card widget"""
        card = QWidget()
        card.setFixedHeight(100)
        card.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        # Icon and value row
        top_row = QHBoxLayout()

        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 24px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        top_row.addWidget(icon_label)

        top_row.addStretch()

        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 20px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        top_row.addWidget(value_label)

        top_widget = QWidget()
        top_widget.setLayout(top_row)
        layout.addWidget(top_widget)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(title_label)

        return card

    def create_recent_activity_panel(self):
        """Create recent activity panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("🕒 Recent Activity")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Activity items
        activities = [
            ("📹", "Analyzed 'Saints vs Panthers' - 4th Quarter", "2 hours ago"),
            ("🏈", "Created Gun Bunch formation", "Yesterday"),
            ("📊", "Generated Red Zone report", "Yesterday"),
            ("🎬", "Exported highlight reel", "2 days ago"),
            ("📈", "Updated win rate stats", "3 days ago"),
        ]

        for icon, description, time in activities:
            activity_item = self.create_activity_item(icon, description, time)
            layout.addWidget(activity_item)

        layout.addStretch()
        return panel

    def create_activity_item(self, icon, description, time):
        """Create an activity item"""
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(10)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            """
            color: #29d28c;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(icon_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(desc_label)

        layout.addStretch()

        # Time
        time_label = QLabel(time)
        time_label.setStyleSheet(
            """
            color: #767676;
            font-size: 10px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(time_label)

        return item

    def create_performance_panel(self):
        """Create performance summary panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("📈 Performance Summary")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Performance metrics
        metrics = [
            ("🎯", "Red Zone Efficiency", "72%", "#29d28c"),
            ("🏃", "3rd Down Conversion", "45%", "#1ce783"),
            ("⚡", "Big Play Rate", "18%", "#17d474"),
            ("🛡️", "Defensive Stops", "62%", "#34e89a"),
            ("⏱️", "Time of Possession", "58%", "#29d28c"),
        ]

        for icon, metric, value, color in metrics:
            metric_item = self.create_metric_item(icon, metric, value, color)
            layout.addWidget(metric_item)

        layout.addStretch()
        return panel

    def create_metric_item(self, icon, metric, value, color):
        """Create a performance metric item"""
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(15)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 16px;
                    font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(icon_label)

        # Metric name
        metric_label = QLabel(metric)
        metric_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(metric_label)

        layout.addStretch()

        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 14px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(value_label)

        return item

    def create_premium_features_panel(self):
        """Create premium features showcase panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: rgba(41, 210, 140, 0.1);
                border-radius: 8px;
                border: 1px solid #29d28c;
                }
            """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()

        header = QLabel("⭐ Premium Features Active")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Subscription status
        if self.current_user:
            status_label = QLabel(f"{self.current_user.subscription_type.upper()} Plan")
            status_label.setStyleSheet(
                """
                color: #29d28c;
                    font-size: 12px;
                    font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                background-color: rgba(41, 210, 140, 0.2);
                padding: 4px 8px;
                border-radius: 4px;
            """
            )
            header_layout.addWidget(status_label)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Feature highlights
        features_layout = QHBoxLayout()
        features_layout.setSpacing(15)

        features = [
            ("🚀", "Unlimited Analysis"),
            ("🎯", "Advanced AI Coaching"),
            ("📊", "Custom Reports"),
            ("🏆", "Beta Features Access"),
        ]

        for icon, feature in features:
            feature_item = QWidget()
            feature_layout = QHBoxLayout(feature_item)
            feature_layout.setContentsMargins(0, 0, 0, 0)
            feature_layout.setSpacing(8)

            icon_label = QLabel(icon)
            icon_label.setStyleSheet(
                """
                color: #29d28c;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            feature_layout.addWidget(icon_label)

            text_label = QLabel(feature)
            text_label.setStyleSheet(
                """
                color: #ffffff;
                font-size: 11px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            feature_layout.addWidget(text_label)

            features_layout.addWidget(feature_item)

        features_widget = QWidget()
        features_widget.setLayout(features_layout)
        layout.addWidget(features_widget)

        return panel

    def switch_to_gameplan_and_play_builder(self):
        """Switch to gameplan tab and launch play builder"""
        # Switch to gameplan tab first
        self.switch_to_tab("gameplan")

        # Small delay to ensure tab switch is complete, then show play builder
        QTimer.singleShot(100, self.show_play_builder)

    def switch_to_tab(self, tab_name):
        """Switch to a specific tab"""
        # Find and activate the corresponding navigation button
        for button in self.nav_buttons:
            button.setChecked(False)
            if button.text().lower().find(tab_name) != -1:
                button.setChecked(True)

        # Update content
        self.current_content = tab_name
        self.update_main_content()
        self.update_right_sidebar()

    def create_gameplan_content(self):
        """Create gameplan tab content with embedded interactive field"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        header = QLabel("🏈 Interactive Play Planner")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Create main horizontal layout for field and controls
        main_layout = QHBoxLayout()

        # Left side: Interactive field
        field_widget = self.create_interactive_field()
        main_layout.addWidget(field_widget, 2)  # Take up more space

        # Right side: Controls
        controls_widget = self.create_play_planner_controls()
        main_layout.addWidget(controls_widget, 1)

        layout.addLayout(main_layout)

        return content

    def create_interactive_field(self):
        """Create the interactive football field widget"""
        field_container = QWidget()
        field_layout = QVBoxLayout(field_container)
        field_layout.setContentsMargins(10, 10, 10, 10)

        # Field controls row
        controls_row = QHBoxLayout()

        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setFixedSize(40, 30)
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #e3e3e3;
                border: 1px solid #666;
                border-radius: 4px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QPushButton:hover { background-color: #3d3d3d; border-color: #1ce783; }
        """
        )

        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setFixedSize(40, 30)
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #e3e3e3;
                border: 1px solid #666;
                border-radius: 4px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QPushButton:hover { background-color: #3d3d3d; border-color: #1ce783; }
        """
        )

        reset_btn = QPushButton("⟲")
        reset_btn.setFixedSize(40, 30)
        reset_btn.setToolTip("Reset View")
        reset_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #e3e3e3;
                border: 1px solid #666;
                border-radius: 4px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QPushButton:hover { background-color: #3d3d3d; border-color: #1ce783; }
        """
        )

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-weight: bold;
            padding: 5px;
        """
        )

        controls_row.addWidget(zoom_in_btn)
        controls_row.addWidget(zoom_out_btn)
        controls_row.addWidget(reset_btn)
        controls_row.addWidget(self.zoom_label)
        controls_row.addStretch()

        field_layout.addLayout(controls_row)

        # Interactive graphics view
        self.field_view = ZoomableGraphicsView()
        self.field_scene = QGraphicsScene()
        self.field_view.setScene(self.field_scene)
        self.field_view.setFixedSize(700, 500)
        self.field_view.setStyleSheet(
            """
            QGraphicsView {
                border: 2px solid #666;
                border-radius: 8px;
                background-color: #1a1a1a;
            }
        """
        )

        # Connect zoom controls
        zoom_in_btn.clicked.connect(self.zoom_in_field)
        zoom_out_btn.clicked.connect(self.zoom_out_field)
        reset_btn.clicked.connect(self.reset_field_zoom)

        field_layout.addWidget(self.field_view)

        # Create the field and players (coordinates display will be set up later)
        self.create_football_field()

        return field_container

    def create_play_planner_controls(self):
        """Create the play planner control panel"""
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(10, 10, 10, 10)

        # Formation presets
        presets_label = QLabel("Formation Presets")
        presets_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        controls_layout.addWidget(presets_label)

        # Preset buttons
        gun_bunch_btn = QPushButton("Gun Bunch")
        gun_bunch_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 2px;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        gun_bunch_btn.clicked.connect(lambda: self.load_field_formation("Gun Bunch"))
        controls_layout.addWidget(gun_bunch_btn)

        gun_trips_btn = QPushButton("Gun Trips TE")
        gun_trips_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 2px;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        gun_trips_btn.clicked.connect(lambda: self.load_field_formation("Gun Trips Te"))
        controls_layout.addWidget(gun_trips_btn)

        i_formation_btn = QPushButton("I-Formation")
        i_formation_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 2px;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        i_formation_btn.clicked.connect(lambda: self.load_field_formation("I-Formation"))
        controls_layout.addWidget(i_formation_btn)

        # Player info
        info_label = QLabel("Player Positions")
        info_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-top: 20px;
        """
        )
        controls_layout.addWidget(info_label)

        # Coordinates display
        self.coordinates_display = QLabel("Drag players to position...")
        self.coordinates_display.setStyleSheet(
            """
            color: #767676;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 12px;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            background-color: #1a1a1a;
        """
        )
        self.coordinates_display.setWordWrap(True)
        controls_layout.addWidget(self.coordinates_display)

        # Action buttons
        save_btn = QPushButton("💾 Save Formation")
        save_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px;
                border: 2px solid #666;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 5px 0;
            }
            QPushButton:hover { border-color: #1ce783; background-color: #3d3d3d; }
        """
        )
        save_btn.clicked.connect(self.save_current_formation)
        controls_layout.addWidget(save_btn)

        reset_formation_btn = QPushButton("🔄 Reset Players")
        reset_formation_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 10px;
                border: 2px solid #666;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 5px 0;
            }
            QPushButton:hover { border-color: #1ce783; background-color: #3d3d3d; }
        """
        )
        reset_formation_btn.clicked.connect(self.reset_players_to_default)
        controls_layout.addWidget(reset_formation_btn)

        # Instructions
        instructions = QLabel(
            """
🏈 Instructions:
• Drag players to desired positions
• Use mouse wheel to zoom in/out
• Select formation presets
• Save custom formations
        """
        )
        instructions.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 15px;
            border: 1px solid #444;
            border-radius: 6px;
            background-color: #1a1a1a;
            margin-top: 20px;
        """
        )
        controls_layout.addWidget(instructions)

        controls_layout.addStretch()

        # Now add the players after coordinates display is created
        self.add_draggable_players()

        return controls_container

    def create_football_field(self):
        """Create the football field graphics"""
        # Field background (green)
        field = QGraphicsRectItem(0, 0, 600, 400)
        field.setBrush(QBrush(QColor("#228B22")))
        field.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(field)

        # End zones
        end_zone_1 = QGraphicsRectItem(0, 0, 600, 40)
        end_zone_1.setBrush(QBrush(QColor("#1e7e1e")))
        end_zone_1.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(end_zone_1)

        end_zone_2 = QGraphicsRectItem(0, 360, 600, 40)
        end_zone_2.setBrush(QBrush(QColor("#1e7e1e")))
        end_zone_2.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(end_zone_2)

        # Yard lines every 40 pixels (10 yards)
        for yard in range(1, 10):
            y_pos = 40 + (yard * 32)  # Scale down for widget
            line = QGraphicsLineItem(0, y_pos, 600, y_pos)
            line.setPen(QPen(QColor("#ffffff"), 1))
            self.field_scene.addItem(line)

        # 50-yard line (midfield)
        midfield_line = QGraphicsLineItem(0, 200, 600, 200)
        midfield_line.setPen(QPen(QColor("#ffffff"), 3))
        self.field_scene.addItem(midfield_line)

        # Line of scrimmage (highlight at 25-yard line)
        los_line = QGraphicsLineItem(0, 120, 600, 120)
        los_line.setPen(QPen(QColor("#ff6b35"), 4))
        self.field_scene.addItem(los_line)

        # Add field labels
        los_label = QGraphicsTextItem("Line of Scrimmage")
        los_label.setDefaultTextColor(QColor("#ff6b35"))
        los_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        los_label.setPos(10, 95)
        self.field_scene.addItem(los_label)

    def add_draggable_players(self):
        """Add draggable player icons to the field"""
        from PyQt6.QtGui import QBrush, QPen
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem

        # Default Gun Bunch formation positions (scaled for widget)
        default_positions = {
            "QB": (300, 100, QColor("#0066cc")),  # Blue for QB
            "RB": (300, 85, QColor("#cc6600")),  # Orange for RB
            "WR1": (120, 120, QColor("#cc0066")),  # Pink for WRs
            "WR2": (180, 120, QColor("#cc0066")),
            "WR3": (420, 120, QColor("#cc0066")),
            "TE": (450, 120, QColor("#9900cc")),  # Purple for TE
            "LT": (250, 120, QColor("#666666")),  # Gray for O-line
            "LG": (275, 120, QColor("#666666")),
            "C": (300, 120, QColor("#666666")),
            "RG": (325, 120, QColor("#666666")),
            "RT": (350, 120, QColor("#666666")),
        }

        self.field_players = {}

        for position, (x, y, color) in default_positions.items():
            # Create player circle
            player = QGraphicsEllipseItem(0, 0, 20, 20)
            player.setPos(x - 10, y - 10)  # Center the circle
            player.setBrush(QBrush(color))
            player.setPen(QPen(QColor("#ffffff"), 1))

            # Make draggable
            player.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
            player.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
            player.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

            # Add text label
            label = QGraphicsTextItem(position, player)
            label.setDefaultTextColor(QColor("#ffffff"))
            label.setFont(QFont("Arial", 7, QFont.Weight.Bold))
            label.setPos(2, 2)  # Center text in circle

            # Store reference
            player.position = position
            player.label = label
            self.field_players[position] = player

            self.field_scene.addItem(player)

        self.update_field_coordinates()

    def zoom_in_field(self):
        """Zoom in the field view"""
        self.field_view.zoom_in()
        zoom_percent = int(self.field_view.current_zoom * 100)
        self.zoom_label.setText(f"{zoom_percent}%")

    def zoom_out_field(self):
        """Zoom out the field view"""
        self.field_view.zoom_out()
        zoom_percent = int(self.field_view.current_zoom * 100)
        self.zoom_label.setText(f"{zoom_percent}%")

    def reset_field_zoom(self):
        """Reset field zoom to 100%"""
        self.field_view.reset_zoom()
        self.zoom_label.setText("100%")

    def load_field_formation(self, formation_name):
        """Load a formation preset on the field"""
        print(f"🏈 Loading formation preset: {formation_name}")

        formations = {
            "Gun Bunch": {
                "QB": (300, 100),
                "RB": (300, 85),
                "WR1": (120, 120),
                "WR2": (180, 120),
                "WR3": (420, 120),
                "TE": (450, 120),
                "LT": (250, 120),
                "LG": (275, 120),
                "C": (300, 120),
                "RG": (325, 120),
                "RT": (350, 120),
            },
            "Gun Trips Te": {
                "QB": (300, 100),
                "RB": (300, 85),
                "WR1": (420, 120),
                "WR2": (450, 120),
                "WR3": (480, 120),
                "TE": (380, 120),
                "LT": (250, 120),
                "LG": (275, 120),
                "C": (300, 120),
                "RG": (325, 120),
                "RT": (350, 120),
            },
            "I-Formation": {
                "QB": (300, 100),
                "RB": (300, 140),
                "WR1": (100, 120),
                "WR2": (500, 120),
                "WR3": (450, 120),
                "TE": (380, 120),
                "LT": (250, 120),
                "LG": (275, 120),
                "C": (300, 120),
                "RG": (325, 120),
                "RT": (350, 120),
            },
        }

        if formation_name in formations:
            formation = formations[formation_name]
            for position, (x, y) in formation.items():
                if position in self.field_players:
                    self.field_players[position].setPos(x - 10, y - 10)
            self.update_field_coordinates()
            print(f"✅ Loaded {formation_name}: {len(formation)} players positioned")
        else:
            print(f"❌ Formation preset '{formation_name}' not found")

    def update_field_coordinates(self):
        """Update the coordinates display"""
        coords_text = ""
        for position in ["QB", "RB", "WR1", "WR2", "WR3", "TE", "LT", "LG", "C", "RG", "RT"]:
            if position in self.field_players:
                player = self.field_players[position]
                x = player.pos().x() + 10  # Add offset to get center
                y = player.pos().y() + 10
                coords_text += f"{position}: ({int(x)}, {int(y)})\n"

        self.coordinates_display.setText(coords_text)

    def save_current_formation(self):
        """Save the current formation"""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        formation_name, ok = QInputDialog.getText(self, "Save Formation", "Enter formation name:")

        if ok and formation_name:
            # Here you would save the formation
            QMessageBox.information(
                self, "Formation Saved", f"Formation '{formation_name}' has been saved!"
            )
            print(f"💾 Formation '{formation_name}' saved")

    def reset_players_to_default(self):
        """Reset players to default Gun Bunch positions"""
        self.load_field_formation("Gun Bunch")

    def create_learn_content(self):
        """Create learn tab content placeholder"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        header = QLabel("Learning Center")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        placeholder = QLabel("📚 Tutorials, guides, and learning resources will be available here")
        placeholder.setStyleSheet(
            """
            color: #767676;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
            text-align: center;
            padding: 40px;
        """
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()
        return content

    def create_debug_content(self):
        """Create debug tool content for analyzing clips and logs"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("🔍 Debug Clip Analyzer")
        title.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-size: 32px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        # Description
        description = QLabel(
            "Analyze clips with synchronized logs, OCR results, and game state data to understand why clips are labeled incorrectly."
        )
        description.setStyleSheet(
            """
            QLabel {
                color: #767676;
                font-size: 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-bottom: 20px;
            }
        """
        )
        description.setWordWrap(True)

        # Launch button
        launch_button = QPushButton("🚀 Launch Debug Tool")
        launch_button.setFixedHeight(50)
        launch_button.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #0b0c0f;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #17d174;
            }
            QPushButton:pressed {
                background-color: #14b863;
            }
        """
        )
        launch_button.clicked.connect(self.launch_debug_tool)

        # Export debug data button
        export_button = QPushButton("📤 Export Debug Data")
        export_button.setFixedHeight(40)
        export_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2a2a2a;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #1ce783;
            }
        """
        )
        export_button.clicked.connect(self.export_debug_data)

        # Status info
        status_label = QLabel()
        if (
            hasattr(self, "analysis_worker")
            and self.analysis_worker
            and hasattr(self.analysis_worker, "analyzer")
        ):
            if (
                hasattr(self.analysis_worker.analyzer, "debug_mode")
                and self.analysis_worker.analyzer.debug_mode
            ):
                status_text = "✅ Debug mode is ENABLED - collecting detailed analysis data"
                status_color = "#1ce783"
            else:
                status_text = "⚠️ Debug mode is DISABLED - limited debug data available"
                status_color = "#ffa500"
        else:
            status_text = "ℹ️ No analysis session active - run video analysis first"
            status_color = "#767676"

        status_label.setText(status_text)
        status_label.setStyleSheet(
            f"""
            QLabel {{
                color: {status_color};
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
                padding: 10px;
                background-color: #1a1a1a;
                border-radius: 6px;
                border-left: 4px solid {status_color};
            }}
        """
        )

        # Features list
        features_label = QLabel("Debug Tool Features:")
        features_label.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-top: 20px;
            }
        """
        )

        features_list = QLabel(
            """
• 🎬 Watch clips with synchronized analysis logs
• 🔍 View OCR results and YOLO detections frame-by-frame
• 📊 Analyze game state data and confidence scores
• ✏️ Annotate clips with corrections and explanations
• 🐛 Identify exactly why clips are mislabeled
• 📈 Track detection accuracy and performance metrics
• 💾 Export debug reports for further analysis
        """
        )
        features_list.setStyleSheet(
            """
            QLabel {
                color: #e3e3e3;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
                line-height: 1.6;
                padding: 15px;
                background-color: #1a1a1a;
                border-radius: 6px;
            }
        """
        )

        # Layout
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(status_label)
        layout.addWidget(launch_button)
        layout.addWidget(export_button)
        layout.addWidget(features_label)
        layout.addWidget(features_list)
        layout.addStretch()

        return content

    def launch_debug_tool(self):
        """Launch the debug clip analyzer tool"""
        try:
            # Import and launch the debug tool
            from debug_clip_analyzer import ClipDebugAnalyzer

            # Get debug data from analyzer if available
            debug_data = None
            if (
                hasattr(self, "analysis_worker")
                and self.analysis_worker
                and hasattr(self.analysis_worker, "analyzer")
            ):
                if hasattr(self.analysis_worker.analyzer, "debug_data"):
                    debug_data = self.analysis_worker.analyzer.debug_data
                    print(
                        f"🔍 Loaded debug data: {len(debug_data.get('clips', []))} clips, {len(debug_data.get('logs', []))} log entries"
                    )

            # Launch debug tool
            self.debug_tool = ClipDebugAnalyzer()
            if debug_data:
                self.debug_tool.load_debug_data(debug_data)
            self.debug_tool.show()

            print("🚀 Debug tool launched successfully")

        except ImportError as e:
            print(f"❌ Failed to import debug tool: {e}")
            # Show error message to user
            from PyQt6.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Debug Tool Error")
            msg.setText("Debug tool is not available.")
            msg.setInformativeText(
                "The debug_clip_analyzer.py file is missing or has import errors."
            )
            msg.exec()

        except Exception as e:
            print(f"❌ Failed to launch debug tool: {e}")
            from PyQt6.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Debug Tool Error")
            msg.setText(f"Failed to launch debug tool: {str(e)}")
            msg.exec()

    def export_debug_data(self):
        """Export debug data to file"""
        try:
            if (
                hasattr(self, "analysis_worker")
                and self.analysis_worker
                and hasattr(self.analysis_worker, "analyzer")
            ):
                if hasattr(self.analysis_worker.analyzer, "export_debug_data"):
                    output_dir = "debug_export"
                    self.analysis_worker.analyzer.export_debug_data(output_dir)
                    print(f"📤 Debug data exported to {output_dir}")

                    # Show success message
                    from PyQt6.QtWidgets import QMessageBox

                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Information)
                    msg.setWindowTitle("Export Complete")
                    msg.setText(f"Debug data exported successfully to '{output_dir}' folder.")
                    msg.exec()
                else:
                    print("⚠️ Debug export not available - analyzer doesn't support export")
            else:
                print("⚠️ No analysis session active - nothing to export")
                from PyQt6.QtWidgets import QMessageBox

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("Export Error")
                msg.setText(
                    "No analysis session active. Run video analysis first to generate debug data."
                )
                msg.exec()

        except Exception as e:
            print(f"❌ Failed to export debug data: {e}")
            from PyQt6.QtWidgets import QMessageBox

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Export Error")
            msg.setText(f"Failed to export debug data: {str(e)}")
            msg.exec()

    def create_default_content(self):
        """Create default content for unimplemented tabs"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        placeholder = QLabel("🚧 This feature is coming soon!")
        placeholder.setStyleSheet(
            """
            color: #767676;
            font-size: 18px;
            font-family: 'Minork Sans', Arial, sans-serif;
            text-align: center;
            padding: 40px;
        """
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()
        return content

    def create_right_sidebar(self, parent_layout):
        """Create collapsible right sidebar"""
        # Right sidebar frame
        self.right_sidebar = QFrame()
        self.right_sidebar.setFixedWidth(300)
        self.right_sidebar.setStyleSheet(
            """
            QFrame {
                background-color: #0b0c0f;
            }
        """
        )

        self.right_sidebar_layout = QVBoxLayout(self.right_sidebar)
        self.right_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.right_sidebar_layout.setSpacing(0)

        # Update right sidebar content
        self.update_right_sidebar()

        parent_layout.addWidget(self.right_sidebar)

    def update_right_sidebar(self):
        """Update right sidebar content based on current tab"""
        # Clear existing content
        for i in reversed(range(self.right_sidebar_layout.count())):
            item = self.right_sidebar_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        # Add content based on current tab
        if self.current_content == "dashboard":
            self.create_dashboard_sidebar()
        elif self.current_content == "analysis":
            self.create_analysis_sidebar()
        else:
            self.create_default_sidebar()

    def create_dashboard_sidebar(self):
        """Create dashboard-specific sidebar content"""
        header = QLabel("Quick Actions")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(header)

        # Recent files or quick actions would go here
        placeholder = QLabel("Dashboard sidebar content")
        placeholder.setStyleSheet(
            """
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(placeholder)
        self.right_sidebar_layout.addStretch()

    def create_analysis_sidebar(self):
        """Create analysis-specific sidebar content"""
        header = QLabel("Analysis Tools")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(header)

        # Cache Performance Section
        cache_header = QLabel("🚀 Cache Performance")
        cache_header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 14px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 10px 20px 5px 20px;
        """
        )
        self.right_sidebar_layout.addWidget(cache_header)

        # Cache status label (will be updated by timer)
        cache_data = self.get_cache_performance_data()
        self.cache_status_label = QLabel(
            f"Status: {cache_data['status'].title()}\nHit Rate: {cache_data['hit_rate']:.1%}\nOperations: {cache_data['total_operations']}"
        )
        self.cache_status_label.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 5px 20px 15px 20px;
            line-height: 1.4;
        """
        )
        self.right_sidebar_layout.addWidget(self.cache_status_label)

        # Hybrid OCR Performance Section
        ocr_header = QLabel("🧠 Hybrid OCR System")
        ocr_header.setStyleSheet(
            """
            color: #1ce783;
            font-size: 14px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 10px 20px 5px 20px;
        """
        )
        self.right_sidebar_layout.addWidget(ocr_header)

        # OCR features status
        self.ocr_status_label = QLabel(
            "✅ PAT Detection\n✅ Penalty Detection\n✅ Temporal Validation\n✅ Color Analysis\n✅ Drive Intelligence\n✅ Yard Line Extraction"
        )
        self.ocr_status_label.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 5px 20px 15px 20px;
            line-height: 1.4;
        """
        )
        self.right_sidebar_layout.addWidget(self.ocr_status_label)

        # 8-Class Model Section
        model_header = QLabel("🎯 8-Class YOLOv8 Model")
        model_header.setStyleSheet(
            """
            color: #17d474;
            font-size: 14px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 10px 20px 5px 20px;
        """
        )
        self.right_sidebar_layout.addWidget(model_header)

        # Model classes status
        self.model_status_label = QLabel(
            "🎮 HUD Detection\n📍 Possession Triangle\n🗺️ Territory Triangle\n⏸️ Pre-play Indicator\n📋 Play Call Screen\n🏈 Down/Distance Area\n⏰ Game Clock Area\n⏱️ Play Clock Area"
        )
        self.model_status_label.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 5px 20px 15px 20px;
            line-height: 1.4;
        """
        )
        self.right_sidebar_layout.addWidget(self.model_status_label)

        self.right_sidebar_layout.addStretch()

    def create_default_sidebar(self):
        """Create default sidebar content"""
        placeholder = QLabel("Sidebar content")
        placeholder.setStyleSheet(
            """
                color: #767676;
                font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(placeholder)
        self.right_sidebar_layout.addStretch()


class ZoomableGraphicsView(QGraphicsView):
    """Enhanced Graphics View with comprehensive zoom and navigation controls"""

    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.zoom_step = 0.15
        self.min_zoom = 0.25
        self.max_zoom = 5.0
        self.pan_mode = False
        self.grid_visible = True
        self.snap_to_grid = False
        self.last_pan_point = QPointF()

        # Setup the view
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

        # Enable mouse tracking for pan operations
        self.setMouseTracking(True)

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for zoom and navigation"""
        # Zoom shortcuts
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, self.zoom_in)  # Alternative
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
        QShortcut(QKeySequence("Ctrl+9"), self, self.fit_to_view)

        # Navigation shortcuts
        QShortcut(QKeySequence("Space"), self, self.toggle_pan_mode)
        QShortcut(QKeySequence("Ctrl+G"), self, self.toggle_grid)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, self.toggle_snap)
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom with Ctrl modifier"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom in/out with Ctrl+scroll
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Normal scroll behavior
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for pan mode"""
        if self.pan_mode and event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_point = event.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for pan mode"""
        if self.pan_mode and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self.last_pan_point
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self.last_pan_point = event.position()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for pan mode"""
        if self.pan_mode:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(event)

    def zoom_in(self):
        """Zoom in by zoom_step"""
        if self.zoom_factor < self.max_zoom:
            factor = 1 + self.zoom_step
            self.scale(factor, factor)
            self.zoom_factor *= factor

    def zoom_out(self):
        """Zoom out by zoom_step"""
        if self.zoom_factor > self.min_zoom:
            factor = 1 / (1 + self.zoom_step)
            self.scale(factor, factor)
            self.zoom_factor *= factor

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.resetTransform()
        self.zoom_factor = 1.0

    def fit_to_view(self):
        """Fit entire scene to view"""
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            transform = self.transform()
            self.zoom_factor = transform.m11()  # Get scaling factor

    def toggle_pan_mode(self):
        """Toggle pan/hand tool mode"""
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def toggle_grid(self):
        """Toggle grid visibility"""
        self.grid_visible = not self.grid_visible
        self.viewport().update()

    def toggle_snap(self):
        """Toggle snap to grid"""
        self.snap_to_grid = not self.snap_to_grid

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showFullScreen()

    def get_zoom_percentage(self):
        """Get current zoom as percentage"""
        return int(self.zoom_factor * 100)


class FloatingZoomControls(QWidget):
    """Floating zoom control widget with comprehensive features"""

    # Signals for communication with parent
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    reset_zoom_requested = pyqtSignal()
    fit_view_requested = pyqtSignal()
    pan_mode_toggled = pyqtSignal(bool)
    grid_toggled = pyqtSignal(bool)
    snap_toggled = pyqtSignal(bool)
    fullscreen_toggled = pyqtSignal()
    mini_map_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_percentage = 100
        self.pan_mode = False
        self.grid_visible = True
        self.snap_enabled = False
        self.mini_map_visible = False

        self.setup_ui()
        self.setup_styling()

    def setup_ui(self):
        """Setup the floating control UI with clear icons"""
        self.setFixedSize(220, 120)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Core Zoom Controls Row
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)

        # Zoom Out button
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(28, 28)
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out_requested.emit)
        zoom_row.addWidget(self.zoom_out_btn)

        # Zoom percentage display
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(40)
        self.zoom_label.setToolTip("Current Zoom Level")
        zoom_row.addWidget(self.zoom_label)

        # Zoom In button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(28, 28)
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in_requested.emit)
        zoom_row.addWidget(self.zoom_in_btn)

        # Reset button
        self.reset_btn = QPushButton("◯")
        self.reset_btn.setFixedSize(28, 28)
        self.reset_btn.setToolTip("Reset Zoom (100%)")
        self.reset_btn.clicked.connect(self.reset_zoom_requested.emit)
        zoom_row.addWidget(self.reset_btn)

        # Fit to view button
        self.fit_btn = QPushButton("⬜")
        self.fit_btn.setFixedSize(28, 28)
        self.fit_btn.setToolTip("Fit Field to View")
        self.fit_btn.clicked.connect(self.fit_view_requested.emit)
        zoom_row.addWidget(self.fit_btn)

        layout.addLayout(zoom_row)

        # Navigation Controls Row
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        # Pan/Hand tool toggle
        self.pan_btn = QPushButton("✋")
        self.pan_btn.setFixedSize(28, 28)
        self.pan_btn.setCheckable(True)
        self.pan_btn.setToolTip("Pan/Drag Tool")
        self.pan_btn.clicked.connect(self.toggle_pan_mode)
        nav_row.addWidget(self.pan_btn)

        # Grid toggle
        self.grid_btn = QPushButton("⊞")
        self.grid_btn.setFixedSize(28, 28)
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.setToolTip("Toggle Grid")
        self.grid_btn.clicked.connect(self.toggle_grid)
        nav_row.addWidget(self.grid_btn)

        # Snap to grid toggle
        self.snap_btn = QPushButton("⊡")
        self.snap_btn.setFixedSize(28, 28)
        self.snap_btn.setCheckable(True)
        self.snap_btn.setToolTip("Snap to Grid")
        self.snap_btn.clicked.connect(self.toggle_snap)
        nav_row.addWidget(self.snap_btn)

        # Mini-map toggle
        self.map_btn = QPushButton("◐")
        self.map_btn.setFixedSize(28, 28)
        self.map_btn.setCheckable(True)
        self.map_btn.setToolTip("Mini-Map Overview")
        self.map_btn.clicked.connect(self.toggle_mini_map)
        nav_row.addWidget(self.map_btn)

        # Fullscreen toggle
        self.fullscreen_btn = QPushButton("⤢")
        self.fullscreen_btn.setFixedSize(28, 28)
        self.fullscreen_btn.setToolTip("Toggle Fullscreen")
        self.fullscreen_btn.clicked.connect(self.fullscreen_toggled.emit)
        nav_row.addWidget(self.fullscreen_btn)

        layout.addLayout(nav_row)

    def setup_styling(self):
        """Apply styling to the floating controls"""
        # Main container styling with #565656 background
        self.setStyleSheet(
            f"""
            FloatingZoomControls {{
                background-color: rgba(86, 86, 86, 220);
                border: 1px solid #29d28c;
                border-radius: 12px;
            }}

            QPushButton {{
                background-color: #565656;
                color: #e3e3e3;
                border: 1px solid #404040;
                border-radius: 6px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }}

            QPushButton:hover {{
                background-color: #6a6a6a;
                border-color: #29d28c;
            }}

            QPushButton:pressed {{
                background-color: #4a4a4a;
            }}

            QPushButton:checked {{
                background-color: #29d28c;
                color: #151515;
                border-color: #1fc47d;
            }}

            QLabel {{
                color: #e3e3e3;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                background-color: transparent;
            }}
        """
        )

    def toggle_pan_mode(self):
        """Toggle pan mode and emit signal"""
        self.pan_mode = not self.pan_mode
        self.pan_mode_toggled.emit(self.pan_mode)

    def toggle_grid(self):
        """Toggle grid visibility and emit signal"""
        self.grid_visible = not self.grid_visible
        self.grid_toggled.emit(self.grid_visible)

    def toggle_snap(self):
        """Toggle snap to grid and emit signal"""
        self.snap_enabled = not self.snap_enabled
        self.snap_toggled.emit(self.snap_enabled)

    def toggle_mini_map(self):
        """Toggle mini-map visibility and emit signal"""
        self.mini_map_visible = not self.mini_map_visible
        self.mini_map_toggled.emit(self.mini_map_visible)

    def update_zoom_display(self, percentage):
        """Update the zoom percentage display"""
        self.zoom_percentage = percentage
        self.zoom_label.setText(f"{percentage}%")

    def update_pan_mode(self, enabled):
        """Update pan mode button state"""
        self.pan_mode = enabled
        self.pan_btn.setChecked(enabled)

    def update_grid_state(self, visible):
        """Update grid button state"""
        self.grid_visible = visible
        self.grid_btn.setChecked(visible)

    def update_snap_state(self, enabled):
        """Update snap button state"""
        self.snap_enabled = enabled
        self.snap_btn.setChecked(enabled)


class AnimatedClockWidget(QSvgWidget):
    """Animated clock widget using your exact clock.svg file"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)  # Larger size to see details

        # Load your exact clock.svg file
        svg_path = "assets/other/clock.svg"
        if os.path.exists(svg_path):
            self.load(svg_path)
        else:
            print(f"⚠️ Clock SVG not found at: {svg_path}")

        # Animation properties
        self.hand_rotation = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_hand_rotation)

        # Make it semi-transparent overlay
        self.setStyleSheet(
            """
            QSvgWidget {
                background-color: rgba(45, 45, 45, 180);
                border: 2px solid #29d28c;
                border-radius: 60px;
            }
        """
        )

    def start_animation(self):
        """Start the clock hand animation"""
        self.animation_timer.start(400)  # Update every 400ms (every few frames)

    def stop_animation(self):
        """Stop the clock hand animation"""
        self.animation_timer.stop()

    def update_hand_rotation(self):
        """Update the hand rotation by modifying the SVG transform"""
        self.hand_rotation = (self.hand_rotation + 30) % 360  # Move 30 degrees

        # Create CSS transformation for the hand layer
        transform_style = f"""
            QSvgWidget {{
                background-color: rgba(45, 45, 45, 180);
                border: 2px solid #29d28c;
                border-radius: 60px;
                transform: rotate({self.hand_rotation}deg);
                transform-origin: center;
            }}
        """
        self.setStyleSheet(transform_style)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SpygateDesktop()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

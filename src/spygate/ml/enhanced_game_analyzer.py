"""
Enhanced game analyzer module for SpygateAI.
Handles game state detection and analysis using YOLOv8 and OpenCV.

OCR Integration: Uses MAIN optimal preprocessing pipeline (0.939 score from 20K sweep)
- All OCR operations use self.ocr (EnhancedOCR) with optimal parameters
- Scale=3.5x, CLAHE clip=1.0 grid=(4,4), Blur=(3,3), Gamma=0.8
- Threshold=adaptive_mean block=13 C=3, Morphological=(3,3) closing
- NO duplicate OCR instances - consolidated to single optimal system
"""

import json
import logging
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract  # Added for new 8-class OCR functionality
import torch
from ultralytics import YOLO

from ..core.hardware import HardwareDetector, HardwareTier
from ..core.optimizer import TierOptimizer

# Import the simple PaddleOCR wrapper
from .simple_paddle_ocr import SimplePaddleOCRWrapper

# Removed SituationalPredictor - using pure OCR detection only
from .template_triangle_detector import TemplateTriangleDetector
from .temporal_extraction_manager import ExtractionResult, TemporalExtractionManager
from .yolov8_model import MODEL_CONFIGS, UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

# Configure logging
logger = logging.getLogger(__name__)

# Hardware tier configurations exactly matching PRD
HARDWARE_TIERS = {
    "ULTRA_LOW": {
        "model_size": "n",
        "img_size": 320,
        "batch_size": 1,
        "device": "cpu",
        "conf": 0.4,
        "target_fps": 0.2,
    },
    "LOW": {
        "model_size": "n",
        "img_size": 416,
        "batch_size": 2,
        "device": "auto",
        "conf": 0.3,
        "target_fps": 0.5,
    },
    "MEDIUM": {
        "model_size": "s",
        "img_size": 640,
        "batch_size": 4,
        "half": True,
        "quantize": True,
        "target_fps": 1.0,
    },
    "HIGH": {
        "model_size": "m",
        "img_size": 832,
        "batch_size": 8,
        "compile": True,
        "target_fps": 2.0,
    },
    "ULTRA": {
        "model_size": "l",
        "img_size": 1280,
        "batch_size": 16,
        "optimize": True,
        "target_fps": 2.5,
    },
}

# UI element classes imported from yolov8_model.py (single source of truth)
# UI_CLASSES now comes from the import above - no duplicate definition needed

# Enhanced color mapping for 8-class visualization (matches trained model)
ENHANCED_COLORS = {
    "hud": (0, 255, 0),  # Green - Main HUD bar
    "possession_triangle_area": (255, 0, 0),  # Blue
    "territory_triangle_area": (0, 0, 255),  # Red
    "preplay_indicator": (255, 255, 0),  # Cyan
    "play_call_screen": (255, 0, 255),  # Magenta
    "down_distance_area": (0, 255, 255),  # Yellow
    "game_clock_area": (255, 128, 0),  # Orange
    "play_clock_area": (128, 255, 0),  # Lime
}


class PerformanceTier(Enum):
    """Hidden performance tier classification."""

    ELITE_PRO = "MCS Championship Level"
    PRO_LEVEL = "Tournament Competitive"
    ADVANCED = "High-Level Competitive"
    INTERMEDIATE = "Solid Fundamentals"
    DEVELOPING = "Learning Strategy"
    BEGINNER = "Basic Understanding"
    LEARNING = "New to Competitive"


@dataclass
class GameState:
    """Represents the current state of the game."""

    possession_team: Optional[str] = None
    territory: Optional[str] = None
    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[int] = None
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    home_team: Optional[str] = None  # Team abbreviation (e.g., "KC", "SF")
    away_team: Optional[str] = None  # Team abbreviation (e.g., "DEN", "LAR")
    quarter: Optional[int] = None
    time: Optional[str] = None
    confidence: float = 0.0
    visualization_layers: dict[str, np.ndarray] = field(default_factory=dict)
    state_indicators: dict[str, bool] = field(default_factory=dict)
    # Additional fields for state persistence
    score_visible: bool = False
    play_elements_visible: bool = False
    quarter_end_detected: bool = False
    is_valid_gameplay: bool = True
    can_track_stats: bool = True
    can_generate_clips: bool = True
    last_valid_down: Optional[int] = None
    last_valid_possession: Optional[str] = None
    last_valid_territory: Optional[str] = None
    # Additional metadata fields
    timestamp: Optional[float] = None
    frame_number: Optional[int] = None
    data_source: Optional[str] = None
    ocr_confidence: Optional[float] = None


@dataclass
class TriangleValidation:
    """Enhanced triangle validation parameters."""

    min_area: float = 100.0
    max_area: float = 1000.0
    min_aspect: float = 0.5
    max_aspect: float = 2.0
    angle_tolerance: float = 15.0  # degrees
    min_confidence: float = 0.3


@dataclass
class DetectionMetrics:
    """Tracks detection performance metrics."""

    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=50))

    def get_average_metrics(self) -> dict[str, float]:
        return {
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0.0,
            "avg_accuracy": np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0,
        }


@dataclass
class DetectionResult:
    """Detection result from YOLOv8."""

    class_id: int
    confidence: float
    bbox: list[int]  # [x, y, w, h]


@dataclass
class SituationContext:
    """Enhanced situation context with possession/territory awareness."""

    possession_team: str = "unknown"  # "user" or "opponent"
    territory: str = "unknown"  # "own" or "opponent"
    situation_type: str = "normal"
    pressure_level: str = "low"  # "low", "medium", "high", "critical"
    leverage_index: float = 0.5  # 0.0-1.0 situational importance
    special_situations: list[str] = field(default_factory=list)  # penalties, turnovers, etc.


@dataclass
class HiddenMMRMetrics:
    """Hidden performance metrics for MMR calculation."""

    # Situational IQ
    red_zone_efficiency: float = 0.0
    third_down_conversion: float = 0.0
    turnover_avoidance: float = 0.0
    clock_management: float = 0.0
    field_position_awareness: float = 0.0

    # Execution Quality
    pressure_performance: float = 0.0
    clutch_factor: float = 0.0
    consistency: float = 0.0
    adaptability: float = 0.0
    momentum_management: float = 0.0

    # Strategic Depth
    formation_diversity: float = 0.0
    situational_play_calling: float = 0.0
    opponent_exploitation: float = 0.0
    game_flow_reading: float = 0.0
    risk_reward_balance: float = 0.0

    def calculate_overall_mmr(self) -> float:
        """Calculate overall MMR score (0-100)."""
        situational_iq = (
            self.red_zone_efficiency
            + self.third_down_conversion
            + self.turnover_avoidance
            + self.clock_management
            + self.field_position_awareness
        ) / 5

        execution_quality = (
            self.pressure_performance
            + self.clutch_factor
            + self.consistency
            + self.adaptability
            + self.momentum_management
        ) / 5

        strategic_depth = (
            self.formation_diversity
            + self.situational_play_calling
            + self.opponent_exploitation
            + self.game_flow_reading
            + self.risk_reward_balance
        ) / 5

        # Weighted average: 40% situational, 35% execution, 25% strategic
        return (situational_iq * 0.4 + execution_quality * 0.35 + strategic_depth * 0.25) * 100

    def get_performance_tier(self) -> PerformanceTier:
        """Get performance tier based on MMR score."""
        mmr = self.calculate_overall_mmr()

        if mmr >= 95:
            return PerformanceTier.ELITE_PRO
        elif mmr >= 85:
            return PerformanceTier.PRO_LEVEL
        elif mmr >= 75:
            return PerformanceTier.ADVANCED
        elif mmr >= 65:
            return PerformanceTier.INTERMEDIATE
        elif mmr >= 50:
            return PerformanceTier.DEVELOPING
        elif mmr >= 35:
            return PerformanceTier.BEGINNER
        else:
            return PerformanceTier.LEARNING


class EnhancedGameAnalyzer:
    """Enhanced game analyzer with hardware-adaptive processing and full game state extraction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hardware: Optional["HardwareDetector"] = None,
        optimization_config: Optional["OptimizationConfig"] = None,
        debug_output_dir: Optional[Path] = None,
    ):
        """
        Initialize the EnhancedGameAnalyzer.

        Args:
            model_path: Path to YOLO model weights
            hardware: HardwareDetector instance (NOT HardwareTier)
            optimization_config: OptimizationConfig instance
            debug_output_dir: Directory for debug output
        Raises:
            TypeError: If hardware is a HardwareTier instead of HardwareDetector
        """
        # Enforce correct type for hardware
        if (
            hardware is not None
            and hasattr(hardware, "__class__")
            and "HardwareTier" in str(type(hardware))
        ):
            raise TypeError(
                f"Expected HardwareDetector instance, got HardwareTier: {hardware}. "
                f"Pass HardwareDetector() instance instead."
            )

        self.hardware = hardware or HardwareDetector()
        self.optimization_config = optimization_config

        # Use custom trained HUD model if no specific path provided
        if model_path is None:
            model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

        # Initialize YOLO model with hardware optimization
        self.model = EnhancedYOLOv8(
            model_path=model_path,
            hardware_tier=self.hardware.detect_tier(),  # Use detect_tier() method
        )
        # Set confidence threshold if supported
        if hasattr(self.model, "conf"):
            self.model.conf = 0.25  # Default confidence threshold

        # Initialize MAIN OCR engine with simple PaddleOCR wrapper
        self.ocr = SimplePaddleOCRWrapper()  # PRIMARY OCR system using PaddleOCR

        # Initialize game state tracking with SIZE LIMITS to prevent memory leaks
        self.current_state = None
        self.state_history = deque(maxlen=200)  # Keep only recent 200 states (bounded)
        self.confidence_threshold = 0.25
        self.detection_interval = 30

        # Add colors for visualization - Updated for 8-class model
        self.colors = {
            "hud": (0, 255, 0),  # Green
            "possession_triangle_area": (255, 0, 0),  # Blue
            "territory_triangle_area": (0, 0, 255),  # Red
            "preplay_indicator": (255, 255, 0),  # Cyan
            "play_call_screen": (255, 0, 255),  # Magenta
            "down_distance_area": (0, 255, 255),  # Yellow - NEW
            "game_clock_area": (255, 128, 0),  # Orange - NEW
            "play_clock_area": (128, 255, 0),  # Lime - NEW
        }

        # UI Classes for our 8-class enhanced model
        self.ui_classes = UI_CLASSES.copy()

        logger.info(
            f"Enhanced Game Analyzer initialized for {self.hardware.detect_tier().name} hardware"
        )

        self.triangle_detector = TemplateTriangleDetector(debug_output_dir)

        # Advanced situation tracking with SIZE LIMITS to prevent memory leaks
        self.hidden_mmr = HiddenMMRMetrics()
        self.situation_history = deque(maxlen=100)  # Keep only recent 100 situations (bounded)
        # Initialize drive tracking with bounded collections to prevent memory leaks
        self.drive_tracking = {
            "current_drive": None,
            "drive_history": deque(maxlen=50),  # Keep only recent 50 drives (bounded)
            "possession_changes": deque(
                maxlen=100
            ),  # Keep only recent 100 possession changes (bounded)
        }

        # Performance tracking (hidden from user)
        self.performance_stats = {
            "offensive_situations": defaultdict(list),
            "defensive_situations": defaultdict(list),
            "transition_moments": [],
            "decision_quality": [],
        }

        # User context (set by application)
        self.user_team = None  # "home" or "away"
        self.analysis_context = "self"  # "self", "opponent", "pro_study"
        self.last_possession_direction = None
        self.last_territory_direction = None
        self.direction_confidence_threshold = 0.4

        # Initialize game state dictionary
        self.game_state = {}

        # Initialize class mapping for YOLO detections - Updated for 8-class model
        self.class_map = {
            "hud": 0,
            "possession_triangle_area": 1,
            "territory_triangle_area": 2,
            "preplay_indicator": 3,
            "play_call_screen": 4,
            "down_distance_area": 5,  # NEW
            "game_clock_area": 6,  # NEW
            "play_clock_area": 7,  # NEW
        }

        # Initialize key moments and clip queues with SIZE LIMITS to prevent memory leaks
        self.key_moments = deque(maxlen=100)  # Keep only recent 100 key moments
        self.clip_queue = deque(maxlen=50)  # Keep only recent 50 clips

        # Initialize HUD state tracking
        self.hud_state = {
            "is_visible": True,
            "frames_since_visible": 0,
            "in_game_interruption": False,
            "max_frames_without_hud": 75,  # 2.5 seconds at 30fps
            "game_interruption_frames": 75,
            "last_valid_state": {},
        }

        # Initialize performance metrics
        self.performance_metrics = {
            "inference_times": deque(maxlen=100),
            "confidence_scores": deque(maxlen=100),
        }

        # Initialize temporal extraction manager for smart OCR
        self.temporal_manager = TemporalExtractionManager()
        logger.info("🧠 Temporal extraction manager initialized for smart OCR voting")

        # Removed situational predictor - using pure OCR detection only
        logger.info("🎯 Pure OCR detection mode - no predictive logic")

        # Initialize temporal validation tracking for game clock with SIZE LIMIT
        self.game_clock_history = deque(
            maxlen=5
        )  # Track last 5 game clock readings for temporal validation
        self.max_clock_history = 5  # Keep last 5 readings for validation
        logger.info("⏰ Game clock temporal validation initialized")

        # Initialize OPTIMIZED burst consensus voting system with SIZE LIMIT
        self.burst_results = deque(maxlen=20)  # Store results from multiple frames for consensus
        self.max_burst_frames = 10  # Maximum frames to store for consensus

        # OPTIMIZATION: Hardware-adaptive burst parameters
        hardware_tier = self.hardware.detect_tier()
        if hardware_tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            self.max_burst_frames = 6  # Reduce memory usage on low-end hardware
        elif hardware_tier == HardwareTier.MEDIUM:
            self.max_burst_frames = 8  # Balanced performance
        else:  # HIGH and ULTRA
            self.max_burst_frames = 12  # More frames for better consensus

        logger.info(
            f"🎯 OPTIMIZED Burst consensus system initialized (max_frames: {self.max_burst_frames})"
        )

        # Initialize memory pools for performance optimization
        self.object_pools = {
            "game_state_pool": [GameState() for _ in range(10)],
            "numpy_arrays": {},  # Cache for common array sizes
            "cv2_kernels": {},  # Cache for morphological kernels
            "roi_cache": {},  # Cache for preprocessed ROIs
        }
        self.pool_index = 0
        logger.info("🚀 Memory pools initialized for performance optimization")

        # Initialize preprocessing cache
        self.preprocessing_cache = {}
        self.previous_regions = {}  # For smart region filtering
        logger.info("⚡ Preprocessing cache initialized")

        # Initialize simple in-memory caching (removed Redis dependency)
        self.advanced_cache = None
        self.cache_enabled = False
        logger.info("🎯 Using simple in-memory caching (Redis cache disabled)")

        # Initialize debug counter for OCR debugging
        self._debug_frame_counter = 0

        # 🎯 CRITICAL FIX: Initialize clip configuration and play timing for intelligent clip boundaries
        self.clip_config = {
            "pre_play_buffer": 3.0,  # seconds before play starts
            "post_play_buffer": 2.0,  # seconds after play ends
            "min_play_duration": 2.0,  # minimum clip duration
            "max_play_duration": 15.0,  # maximum clip duration
            "validation_buffer": 1.0,  # buffer for validation methods
        }

        self.play_timing = {
            "last_preplay_frame": None,
            "last_playcall_frame": None,
            "play_start_frame": None,
            "play_end_frame": None,
            "current_play_duration": 0.0,
        }

        # Initialize frame buffer for clip validation (needed for validate_clip_timing)
        self.frame_buffer = deque(maxlen=450)  # ~15 seconds at 30fps

        logger.info("🎬 Intelligent clip boundary system initialized with game-state validation")

        # Debug data collection
        self.debug_mode = False
        self.debug_data = {
            "clips": [],
            "frame_analysis": {},
            "logs": [],
            "ocr_results": {},
            "yolo_detections": {},
        }

        # Initialize tracking attributes for extract_game_state
        self.last_zone = None
        self.last_formation = None
        self.tracking_metrics = {
            "zone_changes": [],
            "formation_sequences": [],
            "zone_stats": defaultdict(
                lambda: defaultdict(lambda: {"plays": 0, "yards": 0, "scores": 0})
            ),
        }
        self.formation_history = deque(maxlen=10)

        # Initialize last game state for clip validation
        self.last_game_state = None

        # Initialize state persistence attributes
        self.detection_history = defaultdict(
            lambda: {"last_seen": 0, "state_frames": 0, "confidence": 0.0, "persisted_state": False}
        )
        self.state_persistence = {
            "max_frames_without_hud": 9,  # 0.3 seconds at 30fps
            "min_frames_for_state": 3,
            "detection_threshold": 0.5,
        }
        self.hud_occlusion = {
            "regions": {
                "left": {
                    "elements": ["possession_triangle_area", "score_away"],
                    "is_visible": True,
                    "confidence": 1.0,
                },
                "center": {
                    "elements": ["down_distance_area", "game_clock_area"],
                    "is_visible": True,
                    "confidence": 1.0,
                },
                "right": {
                    "elements": ["territory_triangle_area", "play_clock_area"],
                    "is_visible": True,
                    "confidence": 1.0,
                },
            },
            "occlusion_pattern": None,
            "last_known_values": {},
            "min_visible_regions": 2,
            "critical_pairs": [
                ["down_distance_area", "game_clock_area"],
                ["possession_triangle_area", "territory_triangle_area"],
            ],
        }

        # Initialize frame timestamps for clip creation
        self.frame_timestamps = deque(maxlen=450)  # ~15 seconds at 30fps

        # Initialize game history for natural clip boundaries
        self.game_history = deque(maxlen=100)  # Keep last 100 game states

        # Initialize play state tracking
        self.play_state = {
            "is_play_active": False,
            "play_start_time": None,
            "play_count": 0,
            "current_play_duration": 0.0,
            "last_preplay_time": None,
            "last_playcall_time": None,
        }

        # Initialize state indicators for tracking UI elements
        self.state_indicators = {
            "preplay_indicator": False,
            "play_call_screen": False,
            "hud_visible": False,
            "possession_triangle": False,
            "territory_triangle": False,
            "play_in_progress": False,
        }

        # Initialize field zones for field position analytics
        self.field_zones = {
            "own": {
                "red_zone": (0, 20),  # Own 0-20 yard line (defending red zone)
                "short_field": (21, 35),  # Own 21-35 yard line
                "midfield": (36, 50),  # Own 36-50 yard line
            },
            "opponent": {
                "midfield": (0, 15),  # Opponent 50-35 yard line
                "short_field": (16, 30),  # Opponent 34-20 yard line
                "red_zone": (31, 50),  # Opponent 19-0 yard line (attacking red zone)
            },
        }

    def enable_debug_mode(self, enabled=True):
        """Enable or disable debug data collection"""
        self.debug_mode = enabled
        if enabled:
            print("🔍 Debug mode enabled - collecting detailed analysis data")

    def _log_debug(self, message, category="general"):
        """Log debug information"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] [{category.upper()}] {message}"
            self.debug_data["logs"].append(log_entry)
            print(log_entry)

    def _save_debug_frame_data(self, frame_number, frame, game_state, yolo_results, ocr_results):
        """Save detailed frame analysis data for debugging"""
        if not self.debug_mode:
            return

        try:
            # Save frame image
            debug_dir = "debug_frames"
            os.makedirs(debug_dir, exist_ok=True)
            frame_path = os.path.join(debug_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            # Store frame analysis data
            self.debug_data["frame_analysis"][frame_number] = {
                "image_path": frame_path,
                "game_state": (
                    game_state.__dict__ if hasattr(game_state, "__dict__") else str(game_state)
                ),
                "yolo_detections": yolo_results,
                "ocr_results": ocr_results,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in separate collections for easy access
            self.debug_data["yolo_detections"][frame_number] = yolo_results
            self.debug_data["ocr_results"][frame_number] = ocr_results

        except Exception as e:
            self._log_debug(f"Error saving debug frame data: {str(e)}", "error")

    def analyze_frame(self, frame, current_time=None, frame_number=None):
        """
        Enhanced frame analysis with OCR data preservation for clip creation.

        This method now implements a dual-mode system:
        1. Normal mode: Uses temporal smoothing and caching for performance
        2. Clip creation mode: Uses only fresh OCR data to prevent contamination
        """
        try:
            # Check if we're in clip creation mode (set by analysis worker)
            creating_clip = getattr(self, "_creating_clip", False)

            if creating_clip:
                print(f"🧊 CLIP CREATION MODE: Using fresh OCR only at frame {frame_number}")
                # Force fresh OCR extraction without temporal smoothing
                return self._analyze_frame_fresh_ocr(frame, current_time, frame_number)
            else:
                # Normal analysis with temporal smoothing and caching
                return self._analyze_frame_normal(frame, current_time, frame_number)

        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return None

    def _analyze_frame_fresh_ocr(self, frame, current_time=None, frame_number=None):
        """
        Fresh OCR analysis for clip creation - no temporal smoothing or caching.
        This ensures clips get labeled with the exact OCR data that triggered detection.
        """
        try:
            # Step 1: YOLO detection (always fresh)
            detections = self.model.detect(frame)

            if not detections or len(detections) == 0:
                return None

            # Step 2: Fresh OCR extraction using 8-class detections (bypass all caching)
            fresh_ocr_data = self._extract_fresh_ocr_data(detections, frame, frame_number)

            if not fresh_ocr_data:
                return None

            # Step 4: Create game state from fresh data
            game_state = self._create_game_state_from_fresh_ocr(
                fresh_ocr_data, detections, current_time, frame_number, frame
            )

            # Step 5: Store this as the authoritative clip data
            if game_state:
                self._store_clip_detection_data(game_state, frame_number)
                print(
                    f"✅ FRESH OCR PRESERVED: Down {game_state.down} & {game_state.distance} at frame {frame_number}"
                )

            return game_state

        except Exception as e:
            logger.error(f"Fresh OCR analysis error: {e}")
            return None

    def _analyze_frame_normal(self, frame, current_time=None, frame_number=None):
        """
        Normal frame analysis with temporal smoothing and caching for performance.
        """
        try:
            # Use existing analysis logic with temporal smoothing
            detections = self.model.detect(frame)

            if not detections or len(detections) == 0:
                return None

            # Extract game state using existing method
            game_state_dict = self.extract_game_state(frame)

            # Convert dict to GameState object if needed
            if isinstance(game_state_dict, dict):
                game_state = GameState(
                    down=game_state_dict.get("down"),
                    distance=game_state_dict.get("distance"),
                    yard_line=game_state_dict.get("yard_line"),
                    territory=game_state_dict.get("territory"),
                    quarter=game_state_dict.get("quarter"),
                    time=game_state_dict.get("game_clock"),  # Fixed: use 'time' field
                    possession_team=game_state_dict.get("possession_team"),
                    timestamp=current_time,
                    frame_number=frame_number,
                    confidence=game_state_dict.get("confidence", 0.0),
                )
            else:
                game_state = game_state_dict

            return game_state

        except Exception as e:
            logger.error(f"Normal frame analysis error: {e}")
            return None

    def _extract_fresh_ocr_data(self, detections, frame, frame_number):
        """
        Extract OCR data without any temporal smoothing or caching using 8-class YOLO detections.
        This is the authoritative OCR extraction for clip creation.
        """
        try:
            fresh_data = {}

            # Process each detection to extract OCR data from precise regions
            for detection in detections:
                class_name = detection["class_name"]
                bbox = detection["bbox"]
                confidence = detection["confidence"]

                # Extract region from frame
                region_roi = self._extract_region(frame, bbox)
                if region_roi is None:
                    continue

                region_data = {"roi": region_roi, "bbox": bbox, "confidence": confidence}

                # Extract down and distance from precise down_distance_area
                if class_name == "down_distance_area":
                    down_result = self._extract_down_distance_from_region(
                        region_data, current_time=None
                    )
                    if down_result:
                        fresh_data["down"] = down_result.get("down")
                        fresh_data["distance"] = down_result.get("distance")
                        fresh_data["down_distance_text"] = down_result.get("raw_text", "")
                        fresh_data["down_distance_confidence"] = down_result.get("confidence", 0.0)
                        print(
                            f"🔍 FRESH DOWN/DISTANCE: '{down_result.get('raw_text')}' -> Down {fresh_data['down']} & {fresh_data['distance']}"
                        )

                # Extract game clock from precise game_clock_area
                elif class_name == "game_clock_area":
                    clock_result = self._extract_game_clock_from_region(
                        region_data, current_time=None
                    )
                    if clock_result:
                        fresh_data["quarter"] = clock_result.get("quarter")
                        fresh_data["game_clock"] = clock_result.get("time")
                        fresh_data["clock_text"] = clock_result.get("raw_text", "")
                        fresh_data["clock_confidence"] = clock_result.get("confidence", 0.0)
                        print(
                            f"🔍 FRESH GAME CLOCK: '{clock_result.get('raw_text')}' -> Q{fresh_data['quarter']} {fresh_data['game_clock']}"
                        )

                # Extract yard line from territory_triangle_area (where yard line is displayed)
                elif class_name == "territory_triangle_area":
                    yard_line_result = self._extract_yard_line_from_region(region_data)
                    if yard_line_result:
                        fresh_data["yard_line"] = yard_line_result.get("yard_line")
                        fresh_data["territory"] = yard_line_result.get("territory")
                        fresh_data["yard_line_text"] = yard_line_result.get("raw_text", "")
                        fresh_data["yard_line_confidence"] = yard_line_result.get("confidence", 0.0)
                        print(
                            f"🔍 FRESH YARD LINE: '{yard_line_result.get('raw_text')}' -> {fresh_data['territory']} {fresh_data['yard_line']}"
                        )

            # Add timestamp for tracking
            fresh_data["extracted_at_frame"] = frame_number
            fresh_data["extraction_method"] = "fresh_ocr_8class"

            return fresh_data

        except Exception as e:
            logger.error(f"Fresh OCR extraction error: {e}")
            return None

    def _create_game_state_from_fresh_ocr(
        self, fresh_ocr_data, detections, current_time, frame_number, frame
    ):
        """
        Create a GameState object from fresh OCR data without any temporal smoothing.
        """
        try:
            # Extract possession and territory from triangles (these are visual, not OCR)
            possession_team = self._analyze_possession_triangles(detections, frame)
            territory_info = self._analyze_territory_triangles(detections, frame)

            # Create game state with fresh OCR data
            game_state = GameState(
                # Fresh OCR data
                down=fresh_ocr_data.get("down"),
                distance=fresh_ocr_data.get("distance"),
                yard_line=fresh_ocr_data.get("yard_line"),
                territory=fresh_ocr_data.get("territory"),
                quarter=fresh_ocr_data.get("quarter"),
                time=fresh_ocr_data.get("game_clock"),  # Fixed: use 'time' field
                # Visual detection data
                possession_team=possession_team,
                # Metadata
                timestamp=current_time,
                frame_number=frame_number,
                confidence=self._calculate_overall_confidence(fresh_ocr_data),
                # Mark as fresh data
                data_source="fresh_ocr",
                ocr_confidence=fresh_ocr_data.get("down_distance_confidence", 0.0),
            )

            return game_state

        except Exception as e:
            logger.error(f"Game state creation from fresh OCR error: {e}")
            return None

    def _extract_region(self, frame, bbox):
        """
        Extract a region from the frame using bounding box coordinates.

        Args:
            frame: Input frame (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Extracted region as numpy array
        """
        try:
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are integers and within frame bounds
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = frame.shape[:2]

            # Clamp coordinates to frame bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Extract region
            region = frame[y1:y2, x1:x2]
            return region

        except Exception as e:
            logger.error(f"Error extracting region: {e}")
            return None

    def _locate_down_distance_region(self, hud_region):
        """
        Locate the down and distance region within the HUD.
        This is a simplified implementation - in practice, you'd use more sophisticated region detection.
        """
        try:
            if hud_region is None or hud_region.size == 0:
                return None

            # For now, use the left portion of the HUD where down/distance typically appears
            h, w = hud_region.shape[:2]
            # Down/distance is typically in the left 40% of the HUD
            return hud_region[:, : int(w * 0.4)]

        except Exception as e:
            logger.error(f"Error locating down/distance region: {e}")
            return None

    def _locate_yard_line_region(self, hud_region):
        """
        Locate the yard line region within the HUD.
        """
        try:
            if hud_region is None or hud_region.size == 0:
                return None

            # Yard line is typically in the right portion of the HUD
            h, w = hud_region.shape[:2]
            # Yard line is typically in the right 30% of the HUD
            return hud_region[:, int(w * 0.7) :]

        except Exception as e:
            logger.error(f"Error locating yard line region: {e}")
            return None

    def _locate_game_clock_region(self, hud_region):
        """
        Locate the game clock region within the HUD.
        """
        try:
            if hud_region is None or hud_region.size == 0:
                return None

            # Game clock is typically in the center-right portion of the HUD
            h, w = hud_region.shape[:2]
            # Game clock is typically in the center portion of the HUD
            return hud_region[:, int(w * 0.4) : int(w * 0.7)]

        except Exception as e:
            logger.error(f"Error locating game clock region: {e}")
            return None

    def _extract_possession_from_triangles(self, detections):
        """
        Extract possession information from triangle detections.
        """
        try:
            for detection in detections:
                if detection["class_name"] == "possession_triangle_area":
                    # Extract the region and analyze triangle direction
                    bbox = detection["bbox"]
                    # Note: We need the frame to extract the region, but it's not passed here
                    # This will be handled in the calling code
                    return "user"  # Placeholder - actual analysis happens in calling code
            return None
        except Exception as e:
            logger.error(f"Error extracting possession from triangles: {e}")
            return None

    def _extract_territory_from_triangles(self, detections):
        """
        Extract territory information from triangle detections.
        """
        try:
            for detection in detections:
                if detection["class_name"] == "territory_triangle_area":
                    # Extract the region and analyze triangle direction
                    bbox = detection["bbox"]
                    # Note: We need the frame to extract the region, but it's not passed here
                    # This will be handled in the calling code
                    return "opponent"  # Placeholder - actual analysis happens in calling code
            return None
        except Exception as e:
            logger.error(f"Error extracting territory from triangles: {e}")
            return None

    def _analyze_triangle_direction(self, region_roi, triangle_type):
        """
        Analyze triangle direction within a detected region.

        Args:
            region_roi: The extracted region containing the triangle
            triangle_type: "possession" or "territory"

        Returns:
            Direction string: "left", "right", "up", "down", or None
        """
        try:
            if region_roi is None or region_roi.size == 0:
                return None

            # Convert to grayscale for analysis
            if len(region_roi.shape) == 3:
                gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_roi

            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return self._template_match_triangles(region_roi, triangle_type)

            # Find the largest contour (likely the triangle)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < 10:  # Too small to be a triangle
                return self._template_match_triangles(region_roi, triangle_type)

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) >= 3:
                # Calculate moments for centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get bounding rectangle for aspect ratio
                    rect = cv2.boundingRect(largest_contour)
                    aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0

                    # Get triangle vertices
                    vertices = approx.reshape(-1, 2)

                    if triangle_type == "possession":
                        # Horizontal pointing triangles (◀ ▶)
                        if aspect_ratio > 1.2:  # Wide triangle
                            leftmost_idx = np.argmin(vertices[:, 0])
                            rightmost_idx = np.argmax(vertices[:, 0])

                            # Check if leftmost point is actually pointing left
                            if vertices[leftmost_idx, 0] < cx:
                                return "left"  # ◀ User team has ball
                            else:
                                return "right"  # ▶ Opponent team has ball

                    elif triangle_type == "territory":
                        # Vertical pointing triangles (▲ ▼)
                        if aspect_ratio < 0.8:  # Tall triangle
                            topmost_idx = np.argmin(vertices[:, 1])
                            bottommost_idx = np.argmax(vertices[:, 1])

                            # Check if topmost point is actually pointing up
                            if vertices[topmost_idx, 1] < cy:
                                return "up"  # ▲ Own territory
                            else:
                                return "down"  # ▼ Opponent territory

            # Fallback to template matching if contour analysis fails
            return self._template_match_triangles(region_roi, triangle_type)

        except Exception as e:
            logger.error(f"Error analyzing triangle direction: {e}")
            return None

    def _template_match_triangles(self, roi, triangle_type):
        """
        Template matching for specific triangle shapes as fallback.
        """
        try:
            if roi is None or roi.size == 0:
                return None

            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # Create triangle templates
            template_size = min(roi.shape[0], roi.shape[1], 20)
            if template_size < 8:
                return None

            templates = {}

            # Left pointing triangle ◀
            left_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array(
                [
                    [template_size - 2, 2],
                    [2, template_size // 2],
                    [template_size - 2, template_size - 2],
                ]
            )
            cv2.fillPoly(left_template, [points], 255)
            templates["left"] = left_template

            # Right pointing triangle ▶
            right_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array(
                [[2, 2], [template_size - 2, template_size // 2], [2, template_size - 2]]
            )
            cv2.fillPoly(right_template, [points], 255)
            templates["right"] = right_template

            # Up pointing triangle ▲
            up_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array(
                [
                    [template_size // 2, 2],
                    [2, template_size - 2],
                    [template_size - 2, template_size - 2],
                ]
            )
            cv2.fillPoly(up_template, [points], 255)
            templates["up"] = up_template

            # Down pointing triangle ▼
            down_template = np.zeros((template_size, template_size), dtype=np.uint8)
            points = np.array(
                [[2, 2], [template_size - 2, 2], [template_size // 2, template_size - 2]]
            )
            cv2.fillPoly(down_template, [points], 255)
            templates["down"] = down_template

            # Match templates based on triangle type
            best_match = None
            best_score = 0.3  # Minimum threshold

            for direction, template in templates.items():
                # Skip irrelevant directions based on triangle type
                if triangle_type == "possession" and direction in ["up", "down"]:
                    continue
                if triangle_type == "territory" and direction in ["left", "right"]:
                    continue

                try:
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        best_match = direction
                except:
                    continue

            return best_match

        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return None

    def _analyze_possession_triangles(self, detections, frame):
        """
        Analyze possession triangles to determine which team has the ball.

        Args:
            detections: List of YOLO detections
            frame: Current frame for region extraction

        Returns:
            String: "user", "opponent", or None
        """
        try:
            for detection in detections:
                if detection["class_name"] == "possession_triangle_area":
                    # Extract the region containing the possession triangle
                    bbox = detection["bbox"]
                    region_roi = self._extract_region(frame, bbox)

                    if region_roi is not None:
                        # Analyze triangle direction
                        direction = self._analyze_triangle_direction(region_roi, "possession")

                        if direction == "left":
                            return "user"  # User team has the ball
                        elif direction == "right":
                            return "opponent"  # Opponent team has the ball
                        else:
                            print(f"⚠️ Unknown possession triangle direction: {direction}")
                            return None

            return None  # No possession triangle detected

        except Exception as e:
            logger.error(f"Error analyzing possession triangles: {e}")
            return None

    def _analyze_territory_triangles(self, detections, frame):
        """
        Analyze territory triangles to determine field position context.

        Args:
            detections: List of YOLO detections
            frame: Current frame for region extraction

        Returns:
            String: "own", "opponent", or None
        """
        try:
            for detection in detections:
                if detection["class_name"] == "territory_triangle_area":
                    # Extract the region containing the territory triangle
                    bbox = detection["bbox"]
                    region_roi = self._extract_region(frame, bbox)

                    if region_roi is not None:
                        # Analyze triangle direction
                        direction = self._analyze_triangle_direction(region_roi, "territory")

                        if direction == "up":
                            return "opponent"  # In opponent's territory
                        elif direction == "down":
                            return "own"  # In own territory
                        else:
                            print(f"⚠️ Unknown territory triangle direction: {direction}")
                            return None

            return None  # No territory triangle detected

        except Exception as e:
            logger.error(f"Error analyzing territory triangles: {e}")
            return None

    def _store_clip_detection_data(self, game_state, frame_number):
        """
        Store the authoritative game state data for clip creation.
        This prevents data contamination from subsequent frames.
        """
        try:
            if not hasattr(self, "_clip_detection_data"):
                self._clip_detection_data = {}

            # Store with frame number as key
            self._clip_detection_data[frame_number] = {
                "game_state": game_state,
                "down": game_state.down,
                "distance": game_state.distance,
                "yard_line": game_state.yard_line,
                "territory": game_state.territory,
                "quarter": game_state.quarter,
                "game_clock": game_state.time,  # Fixed: use 'time' field
                "confidence": game_state.confidence,
                "stored_at": time.time(),
                "data_source": "fresh_ocr_for_clip",
            }

            # Keep only recent data (last 100 frames)
            if len(self._clip_detection_data) > 100:
                oldest_frame = min(self._clip_detection_data.keys())
                del self._clip_detection_data[oldest_frame]

        except Exception as e:
            logger.error(f"Error storing clip detection data: {e}")

    def get_clip_detection_data(self, frame_number):
        """
        Retrieve the authoritative game state data for a specific frame.
        This is used by the clip creation system to get contamination-free data.
        """
        if hasattr(self, "_clip_detection_data"):
            return self._clip_detection_data.get(frame_number)
        return None

    def _calculate_ocr_confidence(self, text):
        """
        Calculate confidence score for OCR text based on various factors.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        confidence = 0.5  # Base confidence

        # Length factor (reasonable length text is more confident)
        if 2 <= len(text.strip()) <= 10:
            confidence += 0.2

        # Character type factor (alphanumeric is more confident)
        if any(c.isdigit() for c in text):
            confidence += 0.2

        # Common patterns factor
        if any(pattern in text.upper() for pattern in ["ST", "ND", "RD", "TH", "&"]):
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_overall_confidence(self, fresh_ocr_data):
        """
        Calculate overall confidence based on all extracted OCR data.
        """
        confidences = []

        for key in ["down_distance_confidence", "yard_line_confidence", "clock_confidence"]:
            if key in fresh_ocr_data:
                confidences.append(fresh_ocr_data[key])

        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return 0.5  # Default confidence

    def _calculate_frame_confidence(self, game_state: GameState) -> float:
        """Calculate overall confidence score for a frame's analysis results."""
        confidence_factors = []

        # Down/distance confidence
        if game_state.down is not None and game_state.distance is not None:
            confidence_factors.append(0.8)  # High confidence for complete down/distance
        elif game_state.down is not None or game_state.distance is not None:
            confidence_factors.append(0.5)  # Medium confidence for partial

        # Clock confidence
        if game_state.time:
            confidence_factors.append(0.7)

        # Yard line confidence
        if game_state.yard_line is not None:
            confidence_factors.append(0.7)

        # Possession/territory confidence
        if game_state.possession_team:
            confidence_factors.append(0.8)
        if game_state.territory:
            confidence_factors.append(0.6)

        # Return average confidence, or 0.1 if no factors
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.1

    def _process_detections(
        self, detections: list[dict], layers: dict[str, np.ndarray], current_time: float = None
    ) -> GameState:
        """Process detections and update visualization layers with 7-class system."""
        # Ensure cv2 is available in this method scope
        import cv2

        # DEBUG: Log all detections to see what the model is actually detecting
        if detections:
            detection_summary = []
            for det in detections:
                detection_summary.append(f"{det['class_name']}({det['confidence']:.2f})")
            print(f"🔍 YOLO DETECTIONS: {', '.join(detection_summary)}")
        else:
            print(f"❌ NO YOLO DETECTIONS in this frame")

        game_state = GameState()
        total_confidence = 0
        num_detections = 0

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class_name"]

            # Update total confidence
            total_confidence += conf
            num_detections += 1

            # Draw detection on appropriate layer
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[class_name]

            # Extract region for OCR processing
            region_roi = (
                layers["original_frame"][y1:y2, x1:x2] if "original_frame" in layers else None
            )

            if "triangle" in class_name:
                cv2.rectangle(layers["triangle_detection"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    layers["triangle_detection"],
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Update possession/territory based on triangle detection
                if class_name == "possession_triangle_area":
                    # Use proven triangle detection on YOLO-detected region
                    if region_roi is not None:
                        possession_direction = self._analyze_triangle_direction(
                            region_roi, "possession"
                        )
                        if possession_direction:
                            game_state.possession_team = self._get_team_with_ball(
                                possession_direction
                            )
                            # Store in game_state dict for hybrid logic
                            if not hasattr(self, "game_state"):
                                self.game_state = {}
                            if "possession" not in self.game_state:
                                self.game_state["possession"] = {}
                            self.game_state["possession"][
                                "team_with_ball"
                            ] = game_state.possession_team

                        # Extract scores from possession region
                        region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}
                        score_result = self._extract_scores_from_possession_region(region_data)
                        if score_result:
                            game_state.score_away = score_result.get("away_score")
                            game_state.score_home = score_result.get("home_score")
                            game_state.away_team = score_result.get("away_team")
                            game_state.home_team = score_result.get("home_team")

                elif class_name == "territory_triangle_area":
                    # Use proven triangle detection on YOLO-detected region
                    if region_roi is not None:
                        territory_direction = self._analyze_triangle_direction(
                            region_roi, "territory"
                        )
                        if territory_direction:
                            game_state.territory = self._get_field_context(territory_direction)
                            # Store in game_state dict for hybrid logic
                            if not hasattr(self, "game_state"):
                                self.game_state = {}
                            if "territory" not in self.game_state:
                                self.game_state["territory"] = {}
                            self.game_state["territory"]["field_context"] = game_state.territory

                        # Extract yard line from territory region (where yard line is displayed)
                        region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}
                        yard_line_result = self._extract_yard_line_from_region(region_data)
                        if yard_line_result:
                            game_state.yard_line = yard_line_result.get("yard_line")
                            logger.debug(f"🏈 Enhanced Yard Line Detection: {yard_line_result}")

            elif class_name == "down_distance_area":
                cv2.rectangle(layers["ocr_results"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    layers["ocr_results"],
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Extract down & distance using temporal manager
                if region_roi is not None:
                    logger.debug(
                        f"🎯 Extracting down/distance from region with confidence {conf:.2f}"
                    )
                    region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}

                    # FIXED: Use direct OCR extraction with error handling
                    try:
                        # CRITICAL DEBUG: Save the region being processed every 300 frames
                        self._debug_frame_counter += 1

                        if self._debug_frame_counter % 300 == 0:
                            try:
                                import os

                                debug_dir = "debug_ocr_regions"
                                os.makedirs(debug_dir, exist_ok=True)

                                # Save the actual region being processed
                                region_roi = region_data["roi"]
                                cv2.imwrite(
                                    f"{debug_dir}/down_distance_region_frame_{self._debug_frame_counter}.jpg",
                                    region_roi,
                                )
                                print(
                                    f"💾 SAVED DOWN/DISTANCE REGION: {debug_dir}/down_distance_region_frame_{self._debug_frame_counter}.jpg"
                                )

                                # Save bbox info
                                bbox = region_data["bbox"]
                                print(
                                    f"📍 BBOX: {bbox}, Confidence: {region_data['confidence']:.3f}"
                                )

                            except Exception as e:
                                print(f"❌ Debug save error: {e}")

                        down_result = self._extract_down_distance_from_region(
                            region_data, current_time
                        )

                        # CRITICAL DEBUG: Show what OCR actually extracted every 300 frames
                        if self._debug_frame_counter % 300 == 0:
                            print(f"🔤 RAW OCR RESULT from _extract_down_distance_from_region:")
                            print(f"   📊 Result: {down_result}")
                            if down_result:
                                print(
                                    f"   🎯 Down: {down_result.get('down')}, Distance: {down_result.get('distance')}"
                                )
                            else:
                                print(f"   ❌ No result returned from OCR extraction")

                        # CRITICAL DEBUG: Show raw OCR results every 300 frames
                        if self._debug_frame_counter % 300 == 0:
                            try:
                                # Try manual OCR on the region to see what's happening
                                region_roi = region_data["roi"]

                                # Raw Tesseract
                                try:
                                    import pytesseract

                                    raw_text = pytesseract.image_to_string(
                                        region_roi, config="--psm 7"
                                    ).strip()
                                    print(f"   🔤 RAW TESSERACT TEXT: '{raw_text}'")
                                except Exception as e:
                                    print(f"   ❌ TESSERACT ERROR: {e}")

                                # Raw EasyOCR if available
                                try:
                                    if hasattr(self.ocr, "reader") and self.ocr.reader:
                                        easy_results = self.ocr.reader.readtext(region_roi)
                                        print(f"   🔤 RAW EASYOCR RESULTS: {easy_results}")
                                        for bbox, text, conf in easy_results:
                                            print(
                                                f"      📝 Text: '{text}', Confidence: {conf:.3f}"
                                            )
                                except Exception as e:
                                    print(f"   ❌ EASYOCR ERROR: {e}")

                            except Exception as e:
                                print(f"❌ RAW OCR DEBUG ERROR: {e}")

                        if down_result:
                            game_state.down = down_result.get("down")
                            game_state.distance = down_result.get("distance")
                            game_state.yard_line = down_result.get("yard_line")
                            logger.debug(
                                f"✅ Down/Distance extracted: down={game_state.down}, distance={game_state.distance}, yard_line={game_state.yard_line}"
                            )
                            # IMPORTANT: Print to console so desktop app can see successful OCR
                            print(
                                f"🎯 OCR SUCCESS: Down={game_state.down}, Distance={game_state.distance}, Yard Line={game_state.yard_line}"
                            )
                        else:
                            logger.debug("❌ No down/distance extracted from region")
                    except Exception as e:
                        logger.debug(f"❌ OCR extraction error: {e}")
                        # Continue without crashing

            elif class_name == "game_clock_area":
                cv2.rectangle(layers["ocr_results"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    layers["ocr_results"],
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Extract game clock using temporal manager
                if region_roi is not None:
                    region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}
                    clock_result = self._extract_game_clock_from_region(region_data, current_time)

                    if clock_result:
                        # Map the clock result fields correctly
                        game_state.time = clock_result.get("time_string") or clock_result.get(
                            "game_clock"
                        )
                        game_state.quarter = clock_result.get("quarter")
                        # IMPORTANT: Print to console so desktop app can see successful OCR
                        print(
                            f"🕐 CLOCK SUCCESS: Time={game_state.time}, Quarter={game_state.quarter}"
                        )

            elif class_name == "play_clock_area":
                cv2.rectangle(layers["ocr_results"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    layers["ocr_results"],
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Extract play clock using temporal manager
                if region_roi is not None:
                    region_data = {"roi": region_roi, "bbox": bbox, "confidence": conf}
                    play_clock_result = self._extract_play_clock_from_region(
                        region_data, current_time
                    )
                    if play_clock_result:
                        # Play clock info can be stored in game state if needed
                        pass

            else:  # preplay_indicator or play_call_screen
                cv2.rectangle(layers["ocr_results"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    layers["ocr_results"],
                    f"{class_name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Update state indicators for play state tracking
                if class_name == "preplay_indicator":
                    game_state.state_indicators["preplay_indicator"] = True
                    print(f"🎮 PREPLAY INDICATOR DETECTED (confidence: {conf:.2f})")
                elif class_name == "play_call_screen":
                    game_state.state_indicators["play_call_screen"] = True
                    print(f"📋 PLAY CALL SCREEN DETECTED (confidence: {conf:.2f})")

        # Update overall confidence
        game_state.confidence = total_confidence / num_detections if num_detections > 0 else 0.0

        # Set additional state indicators for play state tracking
        game_state.state_indicators["hud_visible"] = any(
            detection["class_name"] == "hud" for detection in detections
        )
        game_state.state_indicators["possession_triangle"] = any(
            detection["class_name"] == "possession_triangle_area" for detection in detections
        )
        game_state.state_indicators["territory_triangle"] = any(
            detection["class_name"] == "territory_triangle_area" for detection in detections
        )

        # Update play state indicators for play tracking
        current_state_dict = {
            "preplay_detected": game_state.state_indicators.get("preplay_indicator", False),
            "play_call_screen": game_state.state_indicators.get("play_call_screen", False),
            "hud_visible": game_state.state_indicators.get("hud_visible", False),
            "possession_triangle_detected": game_state.state_indicators.get(
                "possession_triangle", False
            ),
            "territory_triangle_detected": game_state.state_indicators.get(
                "territory_triangle", False
            ),
        }
        self._update_state_indicators(current_state_dict)

        return game_state

    def _extract_hud_info(self, hud_region: np.ndarray, game_state: GameState) -> None:
        """Extract game state information from HUD region using enhanced OCR with batch processing."""
        try:
            # Use the MAIN OCR system (already initialized with optimal parameters)
            # NO DUPLICATE OCR INSTANCES - Use self.ocr which has the 0.939 score preprocessing

            # OPTIMIZATION: Batch process multiple HUD regions if available
            if hasattr(self, "_pending_hud_regions") and len(self._pending_hud_regions) > 1:
                # Process multiple regions in batch for better performance
                batch_results = self.ocr.process_hud_regions_batch(self._pending_hud_regions)
                ocr_results = batch_results[0]  # Get current region result
                self._pending_hud_regions.clear()
            else:
                # Process single region with our proven optimal OCR pipeline
                ocr_results = self.ocr.process_hud_region(hud_region)

            # Extract down and distance with high accuracy
            if "down" in ocr_results:
                down_data = ocr_results["down"]
                if down_data["confidence"] >= 0.7:  # High confidence threshold
                    game_state.down = down_data["text"]

            if "distance" in ocr_results:
                distance_data = ocr_results["distance"]
                if distance_data["confidence"] >= 0.7:
                    game_state.distance = distance_data["text"]

            # Extract team scores
            if "left_score" in ocr_results:
                score_data = ocr_results["left_score"]
                if score_data["confidence"] >= 0.6:  # Slightly lower threshold for scores
                    game_state.score_away = score_data["text"]

            if "right_score" in ocr_results:
                score_data = ocr_results["right_score"]
                if score_data["confidence"] >= 0.6:
                    game_state.score_home = score_data["text"]

            # Extract team abbreviations
            if "left_team" in ocr_results:
                team_data = ocr_results["left_team"]
                if team_data["confidence"] >= 0.8:  # High threshold for team names
                    game_state.away_team = team_data["text"]

            if "right_team" in ocr_results:
                team_data = ocr_results["right_team"]
                if team_data["confidence"] >= 0.8:
                    game_state.home_team = team_data["text"]

            # Extract yard line information
            if "yard_line" in ocr_results:
                yard_data = ocr_results["yard_line"]
                if yard_data["confidence"] >= 0.6:
                    game_state.yard_line = int(yard_data["text"])

            # Apply professional-grade down detection enhancement
            self._enhance_down_detection(hud_region, game_state, ocr_results)

            # Calculate overall confidence based on successful extractions
            successful_extractions = sum(
                1
                for key in ["down", "distance", "left_score", "right_score"]
                if key in ocr_results and ocr_results[key]["confidence"] >= 0.6
            )
            game_state.confidence = min(0.95, successful_extractions / 4.0)

            # Log extraction results for debugging
            if successful_extractions >= 2:
                logger.debug(
                    f"HUD extraction successful: {successful_extractions}/4 elements detected"
                )
                if game_state.down and game_state.distance:
                    logger.debug(f"Down & Distance: {game_state.down} & {game_state.distance}")
            else:
                logger.debug(
                    f"HUD extraction partial: only {successful_extractions}/4 elements detected"
                )

        except Exception as e:
            logger.error(f"Error in HUD extraction: {e}")
            # Fallback to previous values or defaults
            game_state.confidence = 0.0

    def _enhance_down_detection(
        self, hud_region: np.ndarray, game_state: GameState, ocr_results: dict[str, Any]
    ) -> None:
        """
        Professional-grade down detection enhancement using static HUD positioning.

        This method applies the same precision approach we used for triangle detection:
        1. Static coordinate targeting (HUD doesn't move)
        2. Multi-engine OCR with fallback
        3. Advanced pattern matching and validation
        4. Temporal smoothing for consistency
        5. Confidence-based selection
        """
        try:
            # Initialize down detection history if not exists
            if not hasattr(self, "down_detection_history"):
                self.down_detection_history = deque(maxlen=10)
                self.down_confidence_threshold = 0.75

            # Use precise static coordinates for down/distance region (proven to work)
            h, w = hud_region.shape[:2]

            # These coordinates are based on our previous successful implementations
            down_x1 = int(w * 0.750)  # 75% across (column 15)
            down_x2 = int(w * 0.900)  # 90% across (column 17)
            down_y1 = int(h * 0.200)  # 20% from top
            down_y2 = int(h * 0.800)  # 80% from top

            # Extract the precise down/distance region
            down_region = hud_region[down_y1:down_y2, down_x1:down_x2]

            # Apply professional-grade preprocessing (same as triangle detection)
            enhanced_region = self._preprocess_down_region(down_region)

            # Multi-engine OCR approach with fallback
            down_results = self._multi_engine_down_detection(enhanced_region)

            # Advanced pattern matching and validation
            validated_results = self._validate_down_results(down_results)

            # Temporal smoothing for consistency
            smoothed_results = self._apply_temporal_smoothing(validated_results)

            # Update game state with best results
            if (
                smoothed_results
                and smoothed_results["confidence"] >= self.down_confidence_threshold
            ):
                if "down" in smoothed_results and smoothed_results["down"] is not None:
                    game_state.down = smoothed_results["down"]
                if "distance" in smoothed_results and smoothed_results["distance"] is not None:
                    game_state.distance = smoothed_results["distance"]

                logger.debug(
                    f"Enhanced down detection: {game_state.down} & {game_state.distance} (confidence: {smoothed_results['confidence']:.3f})"
                )

        except Exception as e:
            logger.error(f"Error in enhanced down detection: {e}")

    def _preprocess_down_region(self, down_region: np.ndarray) -> np.ndarray:
        """Apply professional-grade preprocessing to down/distance region."""
        # Scale up for better OCR (same approach as triangle detection)
        scale_factor = 5
        scaled_height, scaled_width = (
            down_region.shape[0] * scale_factor,
            down_region.shape[1] * scale_factor,
        )
        scaled_region = cv2.resize(
            down_region, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        if len(scaled_region.shape) == 3:
            gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = scaled_region

        # Apply high-contrast preprocessing for clean OCR
        _, thresh_region = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh_region

    def _multi_engine_down_detection(self, enhanced_region: np.ndarray) -> list[dict[str, Any]]:
        """Multi-engine OCR approach with fallback (similar to triangle detection)."""
        results = []

        try:
            # Engine 1: EasyOCR with OPTIMAL preprocessing (primary)
            if hasattr(self.ocr, "reader"):
                # Apply optimal preprocessing FIRST, then use EasyOCR
                optimally_preprocessed_easy = self.ocr.preprocess_image(enhanced_region)
                easyocr_results = self.ocr.reader.readtext(optimally_preprocessed_easy)
                for bbox, text, conf in easyocr_results:
                    results.append(
                        {
                            "text": text.strip(),
                            "confidence": conf,
                            "source": "easyocr",
                            "bbox": bbox,
                        }
                    )
        except Exception as e:
            logger.debug(f"EasyOCR failed for down detection: {e}")

        try:
            # Engine 2: Use MAIN OCR preprocessing pipeline for Tesseract (optimal 0.939 score params)
            import pytesseract
            from PIL import Image

            # Apply our optimal preprocessing FIRST, then use Tesseract
            optimally_preprocessed = self.ocr.preprocess_image(enhanced_region)

            # Optimal Tesseract config for Madden HUD text
            custom_config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP "

            pil_image = Image.fromarray(optimally_preprocessed)
            tesseract_text = pytesseract.image_to_string(pil_image, config=custom_config).strip()

            if tesseract_text:
                results.append(
                    {
                        "text": tesseract_text,
                        "confidence": 0.8,  # Default confidence for Tesseract
                        "source": "tesseract",
                        "bbox": None,
                    }
                )

        except Exception as e:
            logger.debug(f"Tesseract failed for down detection: {e}")

        return results

    def _validate_down_results(self, down_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Advanced pattern matching and validation (similar to triangle selection)."""
        best_result = {"confidence": 0.0, "down": None, "distance": None}

        # Comprehensive down/distance patterns (from our previous implementations)
        down_patterns = [
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)",  # "1st & 10", "3rd & 8"
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*Goal",  # "1st & Goal"
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)",  # Variations
            r"(\d+)\s*&\s*(\d+)",  # Simple "3 & 8"
            r"(\d+)(?:nd|rd|th|st)\s*&\s*(\d+)",  # OCR variations
        ]

        for result in down_results:
            text = result["text"]
            confidence = result["confidence"]

            for pattern in down_patterns:
                import re

                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        down = int(match.group(1))

                        # Validate down (1-4)
                        if not (1 <= down <= 4):
                            continue

                        # Handle distance
                        distance = None
                        if len(match.groups()) > 1:
                            distance_text = match.group(2)
                            if distance_text.lower() == "goal":
                                distance = 0  # Goal line
                            else:
                                try:
                                    distance = int(distance_text)
                                    # Validate distance (0-99)
                                    if not (0 <= distance <= 99):
                                        continue
                                except ValueError:
                                    continue

                        # Calculate weighted confidence (similar to triangle scoring)
                        weighted_confidence = confidence * 0.7  # Base confidence

                        # Bonus for complete down & distance
                        if distance is not None:
                            weighted_confidence += 0.2

                        # Bonus for common patterns
                        if down in [1, 3] and distance in [10, 8, 7, 5, 3]:
                            weighted_confidence += 0.1

                        # Update best result if this is better
                        if weighted_confidence > best_result["confidence"]:
                            best_result = {
                                "confidence": weighted_confidence,
                                "down": down,
                                "distance": distance,
                                "source": result["source"],
                                "raw_text": text,
                            }

                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing down/distance: {e}")
                        continue

        return best_result

    def _apply_temporal_smoothing(self, current_result: dict[str, Any]) -> dict[str, Any]:
        """Apply temporal smoothing for consistency (similar to triangle detection)."""
        if not current_result or current_result["confidence"] < 0.5:
            return current_result

        # Add to history with size limit to prevent memory leaks
        self.down_detection_history.append(current_result)
        if len(self.down_detection_history) > 50:  # Keep only recent 50 entries
            self.down_detection_history = self.down_detection_history[-25:]

        # If we have enough history, apply smoothing
        if len(self.down_detection_history) >= 3:
            # Get recent results
            recent_results = list(self.down_detection_history)[-5:]

            # Find most common down value
            down_values = [r["down"] for r in recent_results if r["down"] is not None]
            if down_values:
                from collections import Counter

                most_common_down = Counter(down_values).most_common(1)[0][0]

                # If current result matches most common, boost confidence
                if current_result["down"] == most_common_down:
                    current_result["confidence"] = min(0.95, current_result["confidence"] + 0.1)

                # If current result differs significantly, reduce confidence
                elif abs(current_result["down"] - most_common_down) > 1:
                    current_result["confidence"] *= 0.8

        return current_result

    def optimize_memory(self):
        """Optimize memory usage based on hardware tier."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up caches periodically
        self.cleanup_caches()

        hardware_tier = self.hardware.detect_tier()
        if hardware_tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            # Aggressive optimization for low-end systems
            self.detection_interval = 45  # Reduce detection frequency
            torch.cuda.empty_cache()  # Clear CUDA cache

        elif hardware_tier == HardwareTier.MEDIUM:
            self.detection_interval = 30

        else:  # HIGH and ULTRA
            self.detection_interval = 15  # More frequent updates

    def cleanup_caches(self):
        """Clean up memory caches to prevent memory leaks."""
        # Clean preprocessing cache (keep only recent 100 entries)
        if len(self.preprocessing_cache) > 100:
            # Keep only the most recent 50 entries
            cache_items = list(self.preprocessing_cache.items())
            self.preprocessing_cache = dict(cache_items[-50:])

        # Clean previous regions cache
        if len(self.previous_regions) > 50:
            region_items = list(self.previous_regions.items())
            self.previous_regions = dict(region_items[-25:])

        # Clean object pools - reset to initial state
        self.object_pools["game_state_pool"] = [GameState() for _ in range(10)]
        self.object_pools["numpy_arrays"].clear()
        self.object_pools["cv2_kernels"].clear()
        self.object_pools["roi_cache"].clear()

        # Reset pool index
        self.pool_index = 0

        logger.debug("🧹 Memory caches cleaned up")

    def log_performance_metrics(self):
        """Log current performance metrics."""
        if self.performance_metrics["inference_times"]:
            avg_inference = np.mean(self.performance_metrics["inference_times"][-100:])
            avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0

            logger.info(f"Performance Metrics:")
            logger.info(f"- Average FPS: {avg_fps:.1f}")
            logger.info(f"- Average Inference Time: {avg_inference*1000:.1f}ms")
            logger.info(f"- Detection Rate: {self.detection_interval} frames")
            logger.info(
                f"- Average Confidence: {np.mean(self.performance_metrics['confidence_scores'][-100:]):.3f}"
            )

    def set_user_team(self, team_abbrev: str, is_home: bool) -> None:
        """Set the user's team for analysis.

        Args:
            team_abbrev: User's team abbreviation (e.g., 'KC', 'SF')
            is_home: Whether the user is the home team (True) or away team (False)
        """
        self.user_team = team_abbrev.strip().upper()
        self.is_user_home = is_home

    def set_analysis_context(
        self, analysis_type: str, player_name: str = None, is_home: bool = None
    ) -> None:
        """Set the context for game analysis.

        This should be called at the start of analysis with a simple UI prompt like:

        "What are you analyzing?"
        [My Gameplay] [Opponent's Gameplay]

        If [My Gameplay] selected:
        "Were you home or away?"
        [Home Team (Right Side)] [Away Team (Left Side)]

        Args:
            analysis_type: Either 'self' or 'opponent'
            player_name: Name of the player being analyzed (optional)
            is_home: If analysis_type is 'self', whether user was home (right) or away (left)
                    If not provided for 'self' analysis, raises ValueError
        """
        if analysis_type not in ["self", "opponent"]:
            raise ValueError("analysis_type must be either 'self' or 'opponent'")

        self.analysis_type = analysis_type
        self.analyzed_player = player_name or "Unknown Player"

        if analysis_type == "self":
            if is_home is None:
                raise ValueError("Must specify is_home when analyzing own gameplay")
            self.is_user_home = is_home
        else:
            # For opponent analysis, we don't need home/away as we're just observing
            self.is_user_home = None

    def _update_frame_buffer(self, frame: np.ndarray) -> None:
        """Maintain rolling buffer of recent frames for pre-event context."""
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def detect_key_moment(self, current_state: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Analyze game state changes to detect key moments that should generate clips.

        Args:
            current_state: Current game state

        Returns:
            Dict with clip metadata if key moment detected, None otherwise
        """
        if not self.last_game_state:
            self.last_game_state = current_state
            return None

        key_moment = None
        triggers = []

        # Update state indicators first
        self._update_state_indicators(current_state)

        # Check scoring plays (detect score changes)
        if self.key_moment_triggers["scoring_play"]:
            analyzed_score_changed = (
                current_state["scores"]["analyzed_player"]
                != self.last_game_state["scores"]["analyzed_player"]
            )
            opponent_score_changed = (
                current_state["scores"]["opponent"] != self.last_game_state["scores"]["opponent"]
            )
            if analyzed_score_changed or opponent_score_changed:
                triggers.append("scoring_play")

        # Check possession changes
        if (
            self.key_moment_triggers["possession_change"]
            and current_state["possession"] != self.last_game_state["possession"]
        ):
            triggers.append("possession_change")

        # Check for critical situations (only during active play)
        if (
            self.key_moment_triggers["critical_situation"]
            and self.state_indicators["play_in_progress"]
        ):
            situation = current_state.get("situation", {})
            if (
                situation.get("down") in [3, 4]
                or situation.get("in_redzone")
                or self._is_two_minute_drill(situation.get("game_clock", ""))
            ):
                triggers.append("critical_situation")

        # Track zone changes (no clip generated)
        if current_state.get("field_zone") != self.last_game_state.get("field_zone"):
            self._track_zone_change(current_state)

        # Track formation sequences (no clip generated)
        if current_state.get("formation"):
            self._track_formation_sequence(current_state)

        if triggers:
            key_moment = {
                "triggers": triggers,
                "frame_buffer": self.frame_buffer.copy(),  # Include pre-event context
                "current_state": current_state,
                "previous_state": self.last_game_state,
                "timestamp": time.time(),
                "analysis_context": {
                    "type": self.analysis_type,
                    "analyzed_player": self.analyzed_player,
                    "is_home": self.is_user_home,
                },
            }

        self.last_game_state = current_state
        return key_moment

    def _update_state_indicators(self, current_state: dict[str, Any]) -> None:
        """Update game state indicators based on current frame analysis.
        These map directly to what we can detect with our YOLO classes."""
        current_time = time.time()

        # Previous states
        was_preplay = self.state_indicators["preplay_indicator"]
        was_playcall = self.state_indicators["play_call_screen"]

        # Update current states based on direct YOLO detections
        self.state_indicators["preplay_indicator"] = current_state.get("preplay_detected", False)
        self.state_indicators["play_call_screen"] = current_state.get("play_call_screen", False)
        self.state_indicators["hud_visible"] = current_state.get("hud_visible", False)
        self.state_indicators["possession_triangle"] = current_state.get(
            "possession_triangle_detected", False
        )
        self.state_indicators["territory_triangle"] = current_state.get(
            "territory_triangle_detected", False
        )

        # Track timing of indicators
        if self.state_indicators["preplay_indicator"]:
            self.play_state["last_preplay_time"] = current_time
        if self.state_indicators["play_call_screen"]:
            self.play_state["last_playcall_time"] = current_time

        # Detect play start: preplay indicator disappears after being visible
        if was_preplay and not self.state_indicators["preplay_indicator"]:
            self.play_state["is_play_active"] = True
            self.play_state["play_start_time"] = current_time
            self.play_state["play_count"] += 1
            logger.info(f"Play {self.play_state['play_count']} started")

        # Detect play end: either preplay or playcall appears after play was active
        if self.play_state["is_play_active"] and (
            (not was_preplay and self.state_indicators["preplay_indicator"])
            or (not was_playcall and self.state_indicators["play_call_screen"])
        ):
            self.play_state["is_play_active"] = False
            play_duration = current_time - self.play_state["play_start_time"]
            logger.info(
                f"Play {self.play_state['play_count']} ended after {play_duration:.2f} seconds"
            )

    def is_play_active(self) -> bool:
        """Check if a play is currently active."""
        return self.play_state["is_play_active"]

    def get_current_play_info(self) -> dict[str, Any]:
        """Get information about the current/last play."""
        return {
            "play_number": self.play_state["play_count"],
            "is_active": self.play_state["is_play_active"],
            "play_duration": (
                time.time() - self.play_state["play_start_time"]
                if self.play_state["is_play_active"]
                else 0
            ),
            "last_preplay_time": self.play_state["last_preplay_time"],
            "last_playcall_time": self.play_state["last_playcall_time"],
        }

    def _track_zone_change(self, current_state: dict[str, Any]) -> None:
        """Track zone changes without generating clips."""
        if current_state.get("yard_line") and current_state.get("territory"):
            current_territory, current_zone = self._get_field_zone(
                current_state["yard_line"], current_state["territory"]
            )

            # Track zone change
            current_zone_full = f"{current_territory}_{current_zone}"
            if self.last_zone != current_zone_full:
                self.tracking_metrics["zone_changes"].append(
                    {
                        "timestamp": time.time(),
                        "from_zone": self.last_zone,
                        "to_zone": current_zone_full,
                        "yard_line": current_state["yard_line"],
                    }
                )
                self.last_zone = current_zone_full

    def _track_formation_sequence(self, current_state: dict[str, Any]) -> None:
        """Track formation sequences without generating clips."""
        if current_state.get("formation"):
            formation_matched = self._detect_formation_match(current_state["formation"])
            if formation_matched:
                self.tracking_metrics["formation_sequences"].append(
                    {
                        "timestamp": time.time(),
                        "formation": current_state["formation"],
                        "previous_formation": self.last_formation,
                        "field_zone": self.last_zone,  # Include zone context
                    }
                )
            self.last_formation = current_state["formation"]

    def _is_two_minute_drill(self, game_clock: str) -> bool:
        """Check if current time indicates two-minute drill situation."""
        try:
            minutes, seconds = map(int, game_clock.split(":"))
            return minutes < 2
        except (ValueError, AttributeError):
            return False

    def _get_field_zone(self, yard_line: int, territory: str) -> tuple[str, str]:
        """Determine specific field zone based on yard line and territory.

        Args:
            yard_line: Current yard line (0-50)
            territory: Current territory ('own' or 'opponent')

        Returns:
            Tuple of (territory, zone_name)
        """
        zones = self.field_zones[territory]
        for zone_name, (start, end) in zones.items():
            if start <= yard_line <= end:
                return territory, zone_name
        return territory, "unknown"

    def _update_zone_stats(self, game_state: dict[str, Any]) -> None:
        """Update statistics for the current field zone.

        Args:
            game_state: Current game state including yard line and play result
        """
        if not game_state.get("yard_line") or not game_state.get("territory"):
            return

        territory, zone = self._get_field_zone(game_state["yard_line"], game_state["territory"])

        # Update play count
        self.tracking_metrics["zone_stats"][territory][zone]["plays"] += 1

        # Update yards gained if available
        if game_state.get("yards_gained"):
            self.tracking_metrics["zone_stats"][territory][zone]["yards"] += game_state[
                "yards_gained"
            ]

        # Update scores if touchdown or field goal
        if game_state.get("score_type"):
            self.tracking_metrics["zone_stats"][territory][zone]["scores"] += 1

    def extract_game_state(self, frame: np.ndarray) -> dict[str, Any]:
        """Extract complete game state from a frame."""
        # FIXED: Removed circular call to analyze_frame to prevent infinite recursion
        # Instead, directly process the frame using YOLO and OCR

        # Direct YOLO detection without calling analyze_frame
        detections = self.model.detect(frame)

        # Create required visualization layers for _process_detections
        h, w = frame.shape[:2]
        layers = {
            "original_frame": frame,
            "triangle_detection": np.zeros((h, w, 3), dtype=np.uint8),
            "ocr_results": np.zeros((h, w, 3), dtype=np.uint8),
        }

        # Process detections to get game state
        game_state_obj = self._process_detections(detections, layers, None)

        # Convert GameState object to dictionary
        game_state = {}
        if game_state_obj:
            # Extract all relevant fields from GameState
            game_state = {
                "possession_team": game_state_obj.possession_team,
                "territory": game_state_obj.territory,
                "down": game_state_obj.down,
                "distance": game_state_obj.distance,
                "yard_line": game_state_obj.yard_line,
                "score_home": game_state_obj.score_home,
                "score_away": game_state_obj.score_away,
                "home_team": game_state_obj.home_team,
                "away_team": game_state_obj.away_team,
                "quarter": game_state_obj.quarter,
                "time": game_state_obj.time,
                "confidence": game_state_obj.confidence,
                "visualization_layers": game_state_obj.visualization_layers,
            }

            # Get current field zone if yard line available
            if game_state.get("yard_line") and game_state.get("territory"):
                current_territory, current_zone = self._get_field_zone(
                    game_state["yard_line"], game_state["territory"]
                )

                # Track zone change
                current_zone_full = f"{current_territory}_{current_zone}"
                if hasattr(self, "last_zone") and self.last_zone != current_zone_full:
                    if not hasattr(self, "tracking_metrics"):
                        self.tracking_metrics = {"zone_changes": [], "formation_sequences": []}
                    self.tracking_metrics["zone_changes"].append(
                        {
                            "timestamp": time.time(),
                            "from_zone": getattr(self, "last_zone", None),
                            "to_zone": current_zone_full,
                            "yard_line": game_state["yard_line"],
                        }
                    )
                    self.last_zone = current_zone_full

                # Update zone statistics
                self._update_zone_stats(game_state)

                # Add zone info to game state
                game_state["field_zone"] = {
                    "territory": current_territory,
                    "zone_name": current_zone,
                }

            # Track formation matches internally
            if game_state.get("formation"):
                formation_matched = self._detect_formation_match(game_state["formation"])
                if formation_matched:
                    if not hasattr(self, "tracking_metrics"):
                        self.tracking_metrics = {"zone_changes": [], "formation_sequences": []}
                    self.tracking_metrics["formation_sequences"].append(
                        {
                            "timestamp": time.time(),
                            "formation": game_state["formation"],
                            "previous_formation": getattr(self, "last_formation", None),
                            "field_zone": getattr(self, "last_zone", None),  # Include zone context
                        }
                    )

        return game_state

    def set_key_moment_triggers(self, **triggers) -> None:
        """Configure which events should trigger key moment clips.

        Args:
            **triggers: Boolean flags for each trigger type
            e.g., possession_change=True, scoring_play=False
        """
        for trigger, enabled in triggers.items():
            if trigger in self.key_moment_triggers:
                self.key_moment_triggers[trigger] = enabled

    def _detect_formation_match(self, current_formation: str) -> bool:
        """Detect when formations are matched/countered.

        Args:
            current_formation: Current formation detected

        Returns:
            True if this formation appears to be matching/countering previous one
        """
        if not current_formation or not self.last_formation:
            self.last_formation = current_formation
            return False

        # Add to formation history
        self.formation_history.append(
            {"formation": current_formation, "timestamp": time.time(), "is_matching": False}
        )

        # Keep only last 10 formations
        if len(self.formation_history) > 10:
            self.formation_history.pop(0)

        # Check if this formation is a known counter to the last one
        is_matching = self._check_formation_counter_database(
            previous=self.last_formation, current=current_formation
        )

        self.last_formation = current_formation
        return is_matching

    def _check_formation_counter_database(self, previous: str, current: str) -> bool:
        """Check if current formation is a known counter to previous formation.

        Args:
            previous: Previous formation name
            current: Current formation name

        Returns:
            True if current formation is a known counter, False otherwise
        """
        # Example formation counter database (to be expanded)
        formation_counters = {
            "shotgun_bunch": ["cover_3", "cover_4_quarters"],
            "i_form_close": ["4_4", "5_2"],
            "trips_te": ["cover_2_man", "cover_3_match"],
            # Add more formation matchups
        }

        if previous.lower() in formation_counters:
            return current.lower() in formation_counters[previous.lower()]

        return False

    def update_play_state(self, current_state: dict[str, Any], timestamp: float) -> None:
        """Update play state based on UI element sequence.

        Args:
            current_state: Current detection state
            timestamp: Current frame timestamp
        """
        preplay_visible = current_state.get("preplay_indicator", False)
        playcall_visible = current_state.get("play_call_screen", False)

        # Track timing of indicators
        if preplay_visible:
            self.play_state["last_preplay_time"] = timestamp
        if playcall_visible:
            self.play_state["last_playcall_time"] = timestamp

        # Detect play start: preplay disappears after being visible
        if not preplay_visible and self.play_timing["last_preplay_frame"] is not None:
            if not self.play_state["is_play_active"]:
                self.play_state["is_play_active"] = True
                self.play_state["play_start_time"] = timestamp
                self.play_state["play_count"] += 1

        # Detect play end: either indicator appears
        if self.play_state["is_play_active"]:
            if preplay_visible or playcall_visible:
                self.play_state["is_play_active"] = False
                if self.play_state["play_start_time"] is not None:
                    self.play_state["current_play_duration"] = (
                        timestamp - self.play_state["play_start_time"]
                    )
                self.play_state["play_start_time"] = None

    def process_frame(self, frame: np.ndarray, frame_number: int) -> dict[str, Any]:
        """Process a video frame and update game state.

        Args:
            frame: Video frame to analyze
            frame_number: Current frame number

        Returns:
            Updated game state with persistence and tracking metrics
        """
        # Get initial detections
        detections = self.detect_objects(frame)
        current_state = self.parse_detections(detections)

        # Apply state persistence and recovery logic
        current_state = self.update_detection_state(current_state, frame_number)

        # Track persistence metrics
        self.track_persistence_metrics(current_state, frame_number)

        # Update play state based on persisted detections
        self.update_play_state(current_state)

        # Add metrics to output
        metrics_summary = self.get_persistence_metrics_summary()
        current_state.detection_metrics = {
            "persistence": metrics_summary,
            "frame_number": frame_number,
            "confidence_scores": {
                element: self.detection_history[element]["confidence"]
                for element in self.detection_history.keys()
            },
        }

        return current_state

    def validate_clip_timing(self, current_state: GameState, frame_number: int) -> dict[str, Any]:
        """Validate clip start/end timing using game state changes and play detection."""
        # 🎯 FIXED: Use actual game state changes to determine clip boundaries
        timing_info = {
            "is_valid": False,
            "start_frame": None,
            "end_frame": None,
            "confidence": 0.0,
            "validation_methods": [],
            "reason": "No validation methods succeeded",
        }

        # Method 1: Down/Distance Change Detection (Most Reliable)
        if (
            self.last_game_state
            and current_state.down is not None
            and current_state.distance is not None
        ):
            if (
                current_state.down != self.last_game_state.down
                or current_state.distance != self.last_game_state.distance
            ):
                # Play boundary detected - down/distance changed
                timing_info["is_valid"] = True
                timing_info["start_frame"] = max(
                    0, frame_number - int(self.clip_config["pre_play_buffer"] * 30)
                )
                timing_info["end_frame"] = frame_number + int(
                    self.clip_config["post_play_buffer"] * 30
                )
                timing_info["validation_methods"].append("down_distance_change")
                timing_info["confidence"] = 0.9
                timing_info["reason"] = (
                    f"Down/distance changed from {self.last_game_state.down}&{self.last_game_state.distance} to {current_state.down}&{current_state.distance}"
                )

        # Method 2: Quarter Change Detection
        elif self.last_game_state and current_state.quarter != self.last_game_state.quarter:
            timing_info["is_valid"] = True
            timing_info["start_frame"] = max(
                0, frame_number - int(self.clip_config["pre_play_buffer"] * 30)
            )
            timing_info["end_frame"] = frame_number + int(self.clip_config["post_play_buffer"] * 30)
            timing_info["validation_methods"].append("quarter_change")
            timing_info["confidence"] = 0.8
            timing_info["reason"] = (
                f"Quarter changed from {self.last_game_state.quarter} to {current_state.quarter}"
            )

        # Method 3: Possession Change Detection
        elif (
            self.last_game_state
            and current_state.possession_team != self.last_game_state.possession_team
        ):
            timing_info["is_valid"] = True
            timing_info["start_frame"] = max(
                0, frame_number - int(self.clip_config["pre_play_buffer"] * 30)
            )
            timing_info["end_frame"] = frame_number + int(self.clip_config["post_play_buffer"] * 30)
            timing_info["validation_methods"].append("possession_change")
            timing_info["confidence"] = 0.7
            timing_info["reason"] = (
                f"Possession changed from {self.last_game_state.possession_team} to {current_state.possession_team}"
            )

        # Method 4: Time-based fallback for continuous play
        else:
            # Use dynamic duration based on game situation
            if current_state.quarter and current_state.quarter >= 4:
                # Fourth quarter - longer clips for critical moments
                duration = self.clip_config["max_play_duration"]
            elif current_state.distance and current_state.distance <= 3:
                # Short yardage - shorter clips
                duration = self.clip_config["min_play_duration"] + 3
            else:
                # Normal play - medium duration
                duration = (
                    self.clip_config["min_play_duration"] + self.clip_config["max_play_duration"]
                ) / 2

            timing_info["is_valid"] = True
            timing_info["start_frame"] = max(
                0, frame_number - int(self.clip_config["pre_play_buffer"] * 30)
            )
            timing_info["end_frame"] = frame_number + int(duration * 30)
            timing_info["validation_methods"].append("dynamic_duration")
            timing_info["confidence"] = 0.5
            timing_info["reason"] = f"Dynamic duration: {duration}s based on game situation"

        return timing_info

    def create_clip(self, start_frame: int, end_frame: int) -> dict[str, Any]:
        """Create a clip with the validated frame range."""
        if start_frame is None or end_frame is None:
            return None

        # Ensure we have enough buffer
        start_frame = max(0, start_frame - int(self.clip_config["pre_play_buffer"] * 30))
        end_frame = min(
            len(self.frame_buffer) - 1, end_frame + int(self.clip_config["post_play_buffer"] * 30)
        )

        # Validate clip duration
        duration = self.frame_timestamps[end_frame] - self.frame_timestamps[start_frame]

        if (
            duration < self.clip_config["min_play_duration"]
            or duration > self.clip_config["max_play_duration"]
        ):
            return None

        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": duration,
            "frames": self.frame_buffer[start_frame : end_frame + 1],
            "timestamps": self.frame_timestamps[start_frame : end_frame + 1],
        }

    def update_detection_state(self, current_state: GameState, frame_number: int) -> GameState:
        """Update and persist detection states to handle intermittent failures.

        Enhanced to handle facecam occlusion and partial HUD visibility.
        """
        fps = 30  # Assuming 30fps, could be made configurable

        # Update detection history for each UI element
        for element in self.detection_history.keys():
            current_detected = current_state.state_indicators.get(element, False)
            history = self.detection_history[element]
            frames_since_visible = frame_number - history["last_seen"]

            if current_detected:
                # Element was detected in current frame
                history["last_seen"] = frame_number
                history["state_frames"] += 1
                history["confidence"] = min(1.0, history["confidence"] + 0.2)
                history["persisted_state"] = True

                # Clear any game interruption state if HUD returns
                if element == "hud" and self.hud_state["in_game_interruption"]:
                    self.hud_state["in_game_interruption"] = False
                    self.hud_state["interruption_type"] = None
                    self.hud_state["frames_since_visible"] = 0
            else:
                # Element not detected - apply gap logic
                if frames_since_visible <= self.state_persistence["max_frames_without_hud"]:
                    # Short gap (≤0.3s) - maintain state with penalty
                    history["confidence"] = max(0.0, history["confidence"] * 0.9)  # 10% penalty
                    history["persisted_state"] = True

                elif frames_since_visible <= self.hud_state["game_interruption_frames"]:
                    # Medium gap (0.3s-2.5s) - suspend some features
                    history["confidence"] = max(0.0, history["confidence"] * 0.8)  # 20% penalty
                    history["persisted_state"] = True

                    # Suspend statistical tracking and clip generation
                    if element == "hud":
                        current_state.can_track_stats = False
                        current_state.can_generate_clips = False

                else:
                    # Game interruption (≥2.5s) - determine type
                    history["confidence"] = 0.0
                    history["persisted_state"] = False
                    history["state_frames"] = 0

                    if element == "hud":
                        self.hud_state["in_game_interruption"] = True
                        self.hud_state["frames_since_visible"] = frames_since_visible

                        # Determine interruption type
                        if current_state.score_visible and not current_state.play_elements_visible:
                            if current_state.quarter_end_detected:
                                self.hud_state["interruption_type"] = "quarter_end"
                            else:
                                self.hud_state["interruption_type"] = "game_end"
                        elif not any(current_state.state_indicators.values()):
                            self.hud_state["interruption_type"] = "menu"
                        else:
                            self.hud_state["interruption_type"] = "unknown"

                        current_state.is_valid_gameplay = False
                        current_state.can_track_stats = False
                        current_state.can_generate_clips = False

            # Update state validation counters
            if history["state_frames"] >= self.state_persistence["min_frames_for_state"]:
                if history["confidence"] >= self.state_persistence["detection_threshold"]:
                    current_state.state_indicators[element] = True
                else:
                    current_state.state_indicators[element] = False

            # Store last valid state for recovery
            if history["confidence"] >= self.state_persistence["detection_threshold"]:
                if element == "hud":
                    current_state.last_valid_down = current_state.down
                    current_state.last_valid_possession = current_state.possession_team
                    current_state.last_valid_territory = current_state.territory

        # Recovery validation
        if current_state.state_indicators["hud"]:
            # Validate state changes after gaps
            if current_state.down != current_state.last_valid_down:
                # Require additional confirmation for down changes after gaps
                if history["confidence"] < 0.8:  # Higher threshold for state changes
                    current_state.down = current_state.last_valid_down

            # Progressive confidence restoration
            if not self.hud_state["in_game_interruption"]:
                for element in self.detection_history.keys():
                    history = self.detection_history[element]
                    if history["state_frames"] > self.state_persistence["min_frames_for_state"]:
                        history["confidence"] = min(1.0, history["confidence"] + 0.1)

        # Update region visibility
        frame_regions = {}
        for region_name, region in self.hud_occlusion["regions"].items():
            region_elements_visible = any(
                current_state.state_indicators.get(element, False) for element in region["elements"]
            )
            region["is_visible"] = region_elements_visible
            frame_regions[region_name] = region_elements_visible

            if region_elements_visible:
                # Store last known values for visible elements
                for element in region["elements"]:
                    if element in current_state.state_indicators:
                        self.hud_occlusion["last_known_values"][element] = getattr(
                            current_state, element, None
                        )

        # Detect and adapt to occlusion patterns
        self.detect_occlusion_pattern(frame_regions, frame_number)

        # Validate partial HUD state
        if self.validate_partial_hud(current_state):
            # We have enough visible elements to maintain state
            current_state.is_valid_gameplay = True
            # Use last known values for occluded elements
            for element, value in self.hud_occlusion["last_known_values"].items():
                if not current_state.state_indicators.get(element, False):
                    setattr(current_state, element, value)
        else:
            # Not enough visible elements - treat as interruption
            if not self.hud_state["in_game_interruption"]:
                self.hud_state["in_game_interruption"] = True
                self.hud_state["interruption_type"] = "partial_occlusion"

        return current_state

    def detect_occlusion_pattern(self, frame_regions: dict[str, bool], frame_number: int) -> None:
        """Detect consistent facecam occlusion patterns.

        Args:
            frame_regions: Dictionary of which HUD regions are visible
            frame_number: Current frame number for temporal analysis
        """
        if frame_number < 300:  # First 10 seconds at 30fps
            # Build initial occlusion pattern
            for region, is_visible in frame_regions.items():
                if not is_visible:
                    if not self.hud_occlusion["occlusion_pattern"]:
                        self.hud_occlusion["occlusion_pattern"] = set()
                    self.hud_occlusion["occlusion_pattern"].add(region)

        # Update confidence based on pattern consistency
        if self.hud_occlusion["occlusion_pattern"]:
            pattern_matches = all(
                not frame_regions[region] for region in self.hud_occlusion["occlusion_pattern"]
            )
            if pattern_matches:
                # Consistent occlusion pattern - adjust thresholds
                for region in self.hud_occlusion["occlusion_pattern"]:
                    self.hud_occlusion["regions"][region]["confidence"] = 0.0
            else:
                # Pattern broken - reset and readjust
                self.hud_occlusion["occlusion_pattern"] = None
                self._reset_region_confidences()

    def validate_partial_hud(self, current_state: GameState) -> bool:
        """Validate game state with partially visible HUD.

        Returns:
            bool: True if enough critical elements are visible for valid state
        """
        visible_regions = sum(
            1 for region in self.hud_occlusion["regions"].values() if region["is_visible"]
        )

        if visible_regions < self.hud_occlusion["min_visible_regions"]:
            return False

        # Check if we have at least one critical pair visible
        for pair in self.hud_occlusion["critical_pairs"]:
            pair_visible = all(
                any(
                    element in region["elements"] and region["is_visible"]
                    for region in self.hud_occlusion["regions"].values()
                )
                for element in pair
            )
            if pair_visible:
                return True

        return False

    def _reset_region_confidences(self) -> None:
        """Reset confidence values for all HUD regions."""
        for region in self.hud_occlusion["regions"].values():
            region["confidence"] = 1.0

    def update_hud_state(self, detections: dict[str, Any]) -> None:
        """
        Update HUD visibility state and its effects on analysis
        """
        # Check if HUD is detected
        hud_detected = any(d for d in detections if d["class"] == "hud" and d["confidence"] > 0.6)

        if hud_detected:
            # HUD is visible - reset counters and update state
            self.hud_state["is_visible"] = True
            self.hud_state["frames_since_visible"] = 0

    def detect_objects(self, frame: np.ndarray) -> list[dict]:
        """Detect objects in frame using YOLO model."""
        try:
            # Run YOLO detection
            results = self.model(frame)

            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        detection = {
                            "class": self.model.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": (
                                box.xyxy[0].tolist()
                                if hasattr(box, "xyxy")
                                else box.xywh[0].tolist()
                            ),
                        }
                        detections.append(detection)

            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

    def parse_detections(self, detections: list[dict]) -> GameState:
        """Parse detections into a GameState object."""
        game_state = self._get_pooled_game_state()

        # Initialize state indicators
        game_state.state_indicators = {}

        for detection in detections:
            class_name = detection["class"]
            confidence = detection["confidence"]

            # Update state indicators
            if class_name in self.ui_classes:
                game_state.state_indicators[class_name] = confidence > self.confidence_threshold

        return game_state

    def update_play_state(self, current_state: GameState) -> None:
        """Update play state based on current game state."""
        # This is a simplified version - the full implementation would track play boundaries
        if hasattr(current_state, "state_indicators"):
            preplay_visible = current_state.state_indicators.get("preplay_indicator", False)
            playcall_visible = current_state.state_indicators.get("play_call_screen", False)

            # Update play state tracking
            if preplay_visible:
                self.play_state["last_preplay_time"] = time.time()
            if playcall_visible:
                self.play_state["last_playcall_time"] = time.time()
            self.hud_state["in_game_interruption"] = False

            # Store last valid state
            self.hud_state["last_valid_state"] = {
                "down": self.get_current_down(detections),
                "possession": self.get_possession_state(detections),
                "territory": self.get_territory_state(detections),
            }

            # Enable gameplay tracking
            self.game_state["is_valid_gameplay"] = True
            self.game_state["can_track_stats"] = True
            self.game_state["can_generate_clips"] = True

        else:
            # HUD not detected - increment counter
            self.hud_state["frames_since_visible"] += 1
            self.hud_state["is_visible"] = False

            # Check for game interruption
            if self.hud_state["frames_since_visible"] >= self.hud_state["game_interruption_frames"]:
                self.hud_state["in_game_interruption"] = True
                self.game_state["is_valid_gameplay"] = False
                self.game_state["can_track_stats"] = False
                self.game_state["can_generate_clips"] = False

            # Check if within acceptable gap
            elif self.hud_state["frames_since_visible"] <= self.hud_state["max_frames_without_hud"]:
                # Use last valid state for continuity
                if self.hud_state["last_valid_state"]:
                    self.game_state["last_valid_down"] = self.hud_state["last_valid_state"]["down"]
                    self.game_state["last_valid_possession"] = self.hud_state["last_valid_state"][
                        "possession"
                    ]
                    self.game_state["last_valid_territory"] = self.hud_state["last_valid_state"][
                        "territory"
                    ]
            else:
                # Beyond acceptable gap but not game interruption
                self.game_state["can_track_stats"] = False
                self.game_state["can_generate_clips"] = False

    def should_analyze_frame(self) -> bool:
        """
        Determine if the current frame should be analyzed
        """
        # Skip analysis during game interruptions
        if self.hud_state["in_game_interruption"]:
            return False

        # Continue analysis if HUD visible or within acceptable gap
        return (
            self.hud_state["is_visible"]
            or self.hud_state["frames_since_visible"] <= self.hud_state["max_frames_without_hud"]
        )

    def can_generate_clip(self) -> bool:
        """
        Check if clip generation is currently valid
        """
        return self.game_state["can_generate_clips"]

    def can_track_stats(self) -> bool:
        """
        Check if statistical tracking is currently valid
        """
        return self.game_state["can_track_stats"]

    def track_persistence_metrics(self, current_state: GameState, frame_number: int) -> None:
        """Track performance metrics for state persistence system."""
        # ... existing code ...

    def get_persistence_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of state persistence performance metrics.

        Returns:
            Dictionary containing summarized metrics:
            - Recovery success rates for each gap type
            - Average recovery time
            - Average confidence
            - State change validation rate
            - Game interruption statistics
        """
        total_gaps = (
            self.persistence_metrics["short_gaps_recovered"]
            + self.persistence_metrics["medium_gaps_recovered"]
            + self.persistence_metrics["game_interruptions"]["unknown"]
        )

        # Calculate recovery success rates
        short_gap_rate = (
            self.persistence_metrics["short_gaps_recovered"] / total_gaps if total_gaps > 0 else 0.0
        )
        medium_gap_rate = (
            self.persistence_metrics["medium_gaps_recovered"] / total_gaps
            if total_gaps > 0
            else 0.0
        )
        game_interruption_rate = (
            self.persistence_metrics["game_interruptions"]["unknown"] / total_gaps
            if total_gaps > 0
            else 0.0
        )

        # Calculate average recovery time (in frames)
        avg_recovery_time = (
            sum(self.persistence_metrics["recovery_times"])
            / len(self.persistence_metrics["recovery_times"])
            if self.persistence_metrics["recovery_times"]
            else 0.0
        )

        # Calculate average confidence
        avg_confidence = (
            sum(self.persistence_metrics["confidence_history"])
            / len(self.persistence_metrics["confidence_history"])
            if self.persistence_metrics["confidence_history"]
            else 0.0
        )

        # Calculate state change validation rate
        state_changes = self.persistence_metrics["state_changes"]
        validation_rate = (
            state_changes["validated"] / state_changes["total"]
            if state_changes["total"] > 0
            else 0.0
        )

        return {
            "gap_recovery_rates": {
                "short_gaps": short_gap_rate,
                "medium_gaps": medium_gap_rate,
                "game_interruptions": game_interruption_rate,
            },
            "recovery_stats": {
                "average_frames_to_recover": avg_recovery_time,
                "average_confidence": avg_confidence,
            },
            "state_changes": {
                "validation_rate": validation_rate,
                "total_changes": state_changes["total"],
                "validated_changes": state_changes["validated"],
                "rejected_changes": state_changes["rejected"],
            },
            "gap_counts": {
                "short_gaps_recovered": self.persistence_metrics["short_gaps_recovered"],
                "medium_gaps_recovered": self.persistence_metrics["medium_gaps_recovered"],
                "game_interruptions_detected": self.persistence_metrics["game_interruptions"][
                    "unknown"
                ],
            },
        }

    def validate_triangle(
        self, contour: np.ndarray, validation: TriangleValidation
    ) -> tuple[bool, float]:
        """
        Enhanced triangle validation with geometric checks.

        Args:
            contour: Triangle contour points
            validation: Validation parameters

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        try:
            # Area validation
            area = cv2.contourArea(contour)
            if not (validation.min_area <= area <= validation.max_area):
                return False, 0.0

            # Aspect ratio validation
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h != 0 else 0
            if not (validation.min_aspect <= aspect <= validation.max_aspect):
                return False, 0.0

            # Angle validation
            if len(contour) >= 3:
                angles = []
                for i in range(3):
                    pt1 = contour[i][0]
                    pt2 = contour[(i + 1) % 3][0]
                    pt3 = contour[(i + 2) % 3][0]

                    v1 = pt1 - pt2
                    v2 = pt3 - pt2

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    angles.append(angle)

                # Check if angles are approximately 60° (equilateral)
                angle_diffs = [abs(angle - 60) for angle in angles]
                if max(angle_diffs) > validation.angle_tolerance:
                    return False, 0.0

                # Calculate confidence based on angle quality
                angle_confidence = 1.0 - (max(angle_diffs) / validation.angle_tolerance)

                # Combine with area confidence
                area_confidence = min(area / validation.max_area, 1.0)

                final_confidence = (angle_confidence + area_confidence) / 2
                return True, max(final_confidence, validation.min_confidence)

        except Exception as e:
            logger.error(f"Error in triangle validation: {e}")
            return False, 0.0

        return False, 0.0

    def _handle_possession_change(self, old_direction: str, new_direction: str) -> None:
        """
        Handle possession triangle flip - indicates turnover or change of possession.

        Args:
            old_direction: Previous possession direction ('left' or 'right')
            new_direction: New possession direction ('left' or 'right')
        """
        logger.info(f"🔄 POSSESSION CHANGE: {old_direction} → {new_direction}")

        # Determine which team gained/lost possession
        if old_direction == "left" and new_direction == "right":
            event_type = "possession_change_to_right"
            description = "Right team gained possession"
        elif old_direction == "right" and new_direction == "left":
            event_type = "possession_change_to_left"
            description = "Left team gained possession"
        else:
            return  # No actual change

        # This is a key moment - should trigger clip generation
        self._trigger_key_moment(
            {
                "type": event_type,
                "description": description,
                "timestamp": time.time(),
                "old_possession": old_direction,
                "new_possession": new_direction,
                "clip_worthy": True,
                "priority": "high",
            }
        )

    def _handle_territory_change(self, old_direction: str, new_direction: str) -> None:
        """
        Handle territory triangle flip - indicates crossing midfield.

        Args:
            old_direction: Previous territory direction ('up' or 'down')
            new_direction: New territory direction ('up' or 'down')
        """
        logger.info(f"🗺️ TERRITORY CHANGE: {old_direction} → {new_direction}")

        # Determine field position change
        if old_direction == "down" and new_direction == "up":
            event_type = "entered_opponent_territory"
            description = "Crossed into opponent's territory"
        elif old_direction == "up" and new_direction == "down":
            event_type = "entered_own_territory"
            description = "Crossed back into own territory"
        else:
            return  # No actual change

        # Territory changes are important for field position analysis
        self._trigger_key_moment(
            {
                "type": event_type,
                "description": description,
                "timestamp": time.time(),
                "old_territory": old_direction,
                "new_territory": new_direction,
                "clip_worthy": False,  # Usually not clip-worthy by itself
                "priority": "medium",
            }
        )

    def _get_team_with_ball(self, possession_direction: str) -> str:
        """
        Determine which team has the ball based on possession triangle direction.

        Args:
            possession_direction: Direction of possession triangle ('left' or 'right')

        Returns:
            Team identifier ('away_team' or 'home_team')
        """
        # In Madden 25 HUD:
        # - Away team is on the LEFT side of the score
        # - Home team is on the RIGHT side of the score
        # - Possession triangle points TO the team that HAS the ball

        direction_lower = possession_direction.lower()
        if direction_lower == "left":
            return "away_team"  # Left team (away) has the ball
        elif direction_lower == "right":
            return "home_team"  # Right team (home) has the ball
        else:
            return "unknown"

    def _get_field_context(self, territory_direction: str) -> str:
        """
        Determine field position context based on territory triangle direction.

        Args:
            territory_direction: Direction of territory triangle ('up' or 'down')

        Returns:
            Field context description
        """
        # In Madden 25 HUD:
        # - ▲ (up) = In OPPONENT'S territory (good field position)
        # - ▼ (down) = In OWN territory (poor field position)

        direction_lower = territory_direction.lower()
        if direction_lower == "up":
            return "opponent_territory"  # Good field position
        elif direction_lower == "down":
            return "own_territory"  # Poor field position
        else:
            return "unknown"

    def _trigger_key_moment(self, moment_data: dict[str, Any]) -> None:
        """
        Trigger a key moment event for potential clip generation.

        Args:
            moment_data: Dictionary containing moment information
        """
        # Add to key moments queue for processing
        if not hasattr(self, "key_moments"):
            self.key_moments = []

        self.key_moments.append(moment_data)

        # Log the key moment
        logger.info(f"🎯 KEY MOMENT: {moment_data['description']}")

        # If this is clip-worthy, trigger clip detection
        if moment_data.get("clip_worthy", False):
            self._queue_clip_generation(moment_data)

    def _queue_clip_generation(self, moment_data: dict[str, Any]) -> None:
        """
        Queue a clip for generation based on a key moment.

        Args:
            moment_data: Key moment data containing clip information
        """
        # Add to clip generation queue
        if not hasattr(self, "clip_queue"):
            self.clip_queue = []

        clip_data = {
            "trigger_event": moment_data["type"],
            "description": moment_data["description"],
            "timestamp": moment_data["timestamp"],
            "priority": moment_data["priority"],
            "pre_buffer_seconds": 3.0,  # 3 seconds before event
            "post_buffer_seconds": 2.0,  # 2 seconds after event
            "metadata": {
                "possession_change": moment_data.get("old_possession")
                != moment_data.get("new_possession"),
                "territory_change": moment_data.get("old_territory")
                != moment_data.get("new_territory"),
            },
        }

        self.clip_queue.append(clip_data)
        logger.info(f"📹 QUEUED CLIP: {clip_data['description']}")

    def get_triangle_state_summary(self) -> dict[str, Any]:
        """
        Get a summary of current triangle states and their game meaning.

        Returns:
            Dictionary containing triangle state information
        """
        possession_info = self.game_state.get("possession", {})
        territory_info = self.game_state.get("territory", {})

        return {
            "possession": {
                "direction": possession_info.get("direction", "unknown"),
                "team_with_ball": possession_info.get("team_with_ball", "unknown"),
                "confidence": possession_info.get("confidence", 0.0),
                "meaning": self._get_possession_meaning(possession_info.get("direction")),
            },
            "territory": {
                "direction": territory_info.get("direction", "unknown"),
                "field_context": territory_info.get("field_context", "unknown"),
                "confidence": territory_info.get("confidence", 0.0),
                "meaning": self._get_territory_meaning(territory_info.get("direction")),
            },
            "game_situation": self._analyze_combined_triangle_state(
                possession_info, territory_info
            ),
        }

    def _get_possession_meaning(self, direction: str) -> str:
        """Get human-readable meaning of possession triangle direction."""
        if direction == "left":
            return "Away team (left side) has possession"
        elif direction == "right":
            return "Home team (right side) has possession"
        else:
            return "Possession unclear"

    def _get_territory_meaning(self, direction: str) -> str:
        """Get human-readable meaning of territory triangle direction."""
        if direction == "up":
            return "In opponent's territory (good field position)"
        elif direction == "down":
            return "In own territory (poor field position)"
        else:
            return "Field position unclear"

    def _analyze_combined_triangle_state(self, possession: dict, territory: dict) -> str:
        """
        Analyze the combined meaning of both triangle states.

        Args:
            possession: Possession triangle state
            territory: Territory triangle state

        Returns:
            Combined game situation description
        """
        poss_dir = possession.get("direction", "unknown")
        terr_dir = territory.get("direction", "unknown")

        if poss_dir == "unknown" or terr_dir == "unknown":
            return "Game situation unclear"

        # Analyze combinations
        if poss_dir == "left" and terr_dir == "up":
            return "Away team has ball in opponent territory (scoring opportunity)"
        elif poss_dir == "left" and terr_dir == "down":
            return "Away team has ball in own territory (defensive position)"
        elif poss_dir == "right" and terr_dir == "up":
            return "Home team has ball in opponent territory (scoring opportunity)"
        elif poss_dir == "right" and terr_dir == "down":
            return "Home team has ball in own territory (defensive position)"
        else:
            return "Unknown game situation"

    # ========================================================================
    # ADVANCED SITUATION TRACKING WITH POSSESSION/TERRITORY INTELLIGENCE
    # ========================================================================

    def analyze_advanced_situation(self, game_state: GameState) -> SituationContext:
        """
        Analyze advanced game situation using possession/territory intelligence.

        Args:
            game_state: Current game state with possession/territory data

        Returns:
            Enhanced situation context with strategic analysis
        """
        context = SituationContext()

        # Determine possession team relative to user
        if game_state.possession_team and self.user_team:
            if game_state.possession_team == self.user_team:
                context.possession_team = "user"
            else:
                context.possession_team = "opponent"

        # Territory context
        context.territory = game_state.territory or "unknown"

        # Analyze situation type
        context.situation_type = self._classify_situation_type(game_state, context)

        # Determine pressure level
        context.pressure_level = self._calculate_pressure_level(game_state, context)

        # Calculate leverage index (situational importance)
        context.leverage_index = self._calculate_leverage_index(game_state, context)

        # Track for hidden MMR analysis
        self._track_situation_for_mmr(game_state, context)

        # Detect special situations (penalties, turnovers, etc.)
        context.special_situations = self._detect_special_situations(game_state)

        return context

    def _detect_special_situations(self, game_state: GameState) -> list[str]:
        """Detect special situations like penalties, turnovers, PAT, etc."""
        special_situations = []

        # PAT detection - check down text for "PAT" or similar
        if hasattr(game_state, "down_text") and game_state.down_text:
            down_text_upper = str(game_state.down_text).upper()
            if any(
                pat_indicator in down_text_upper
                for pat_indicator in ["PAT", "P.A.T", "POINT AFTER", "EXTRA POINT"]
            ):
                special_situations.append("pat")
                logger.info("🏈 PAT detected from down text")

        # Penalty detection - check for FLAG text or yellow indicators
        if hasattr(game_state, "penalty_detected") and game_state.penalty_detected:
            special_situations.append("penalty")
            logger.info("🚩 Penalty detected")

        # Turnover detection - check possession changes
        if hasattr(self, "previous_possession") and hasattr(game_state, "possession_team"):
            if self.previous_possession and game_state.possession_team:
                if self.previous_possession != game_state.possession_team:
                    special_situations.append("turnover")
                    logger.info(
                        f"🔄 Turnover detected: {self.previous_possession} → {game_state.possession_team}"
                    )
            self.previous_possession = game_state.possession_team

        # Touchdown detection - check for score changes of 6 points
        if (
            hasattr(self, "previous_scores")
            and game_state.score_home is not None
            and game_state.score_away is not None
        ):
            if self.previous_scores:
                home_diff = game_state.score_home - self.previous_scores["home"]
                away_diff = game_state.score_away - self.previous_scores["away"]

                if home_diff == 6 or away_diff == 6:
                    special_situations.append("touchdown")
                    logger.info("🏈 Touchdown detected from score change")
                elif home_diff == 3 or away_diff == 3:
                    special_situations.append("field_goal")
                    logger.info("🏈 Field goal detected from score change")
                elif home_diff == 2 or away_diff == 2:
                    special_situations.append("safety")
                    logger.info("🏈 Safety detected from score change")

            self.previous_scores = {"home": game_state.score_home, "away": game_state.score_away}

        # Sack detection - would require play result analysis
        # This is more complex and would need play-by-play data

        return special_situations

    def _classify_situation_type(self, game_state: GameState, context: SituationContext) -> str:
        """
        FIXED: Classify situation type using hierarchical priority system.
        More specific situations take precedence over general ones.
        Priority: Critical downs > Field position + downs > Field position > Time > Normal
        """

        # PRIORITY 1: Critical down situations (highest priority - never override)
        if game_state.down == 4:
            # Fourth down is always critical regardless of field position
            if (
                game_state.yard_line
                and context.territory == "opponent"
                and game_state.yard_line <= 5
            ):
                return "fourth_down_goal_line"
            elif (
                game_state.yard_line
                and context.territory == "opponent"
                and game_state.yard_line <= 20
            ):
                return "fourth_down_red_zone"
            else:
                return "fourth_down"

        # PRIORITY 2: Third down situations (high priority)
        if game_state.down == 3:
            # Combine third down with field position for more specific classification
            if game_state.distance and game_state.distance >= 7:
                if (
                    game_state.yard_line
                    and context.territory == "opponent"
                    and game_state.yard_line <= 20
                ):
                    return "third_and_long_red_zone"  # Most specific
                else:
                    return "third_and_long"
            elif game_state.distance and game_state.distance <= 3:
                if (
                    game_state.yard_line
                    and context.territory == "opponent"
                    and game_state.yard_line <= 5
                ):
                    return "third_and_short_goal_line"  # Most specific
                elif (
                    game_state.yard_line
                    and context.territory == "opponent"
                    and game_state.yard_line <= 20
                ):
                    return "third_and_short_red_zone"  # Most specific
                else:
                    return "third_and_short"
            else:
                # Generic third down with field position context
                if (
                    game_state.yard_line
                    and context.territory == "opponent"
                    and game_state.yard_line <= 20
                ):
                    return "third_down_red_zone"
                else:
                    return "third_down"

        # PRIORITY 3: Critical field position situations (medium-high priority)
        if game_state.yard_line:
            if context.possession_team == "user":
                # Goal line is most critical
                if context.territory == "opponent" and game_state.yard_line <= 5:
                    return "goal_line_offense"
                # Red zone is next most critical
                elif context.territory == "opponent" and game_state.yard_line <= 20:
                    return "red_zone_offense"
                # Backed up is defensive critical
                elif context.territory == "own" and game_state.yard_line <= 15:
                    return "backed_up_offense"
                # General scoring position (least specific field position)
                elif context.territory == "opponent":
                    return "scoring_position"
            else:  # opponent possession
                # Goal line defense is most critical
                if context.territory == "own" and game_state.yard_line <= 5:
                    return "goal_line_defense"
                # Red zone defense is next most critical
                elif context.territory == "own" and game_state.yard_line <= 20:
                    return "red_zone_defense"
                # Pressure defense when opponent is backed up
                elif context.territory == "opponent" and game_state.yard_line <= 15:
                    return "pressure_defense"

        # PRIORITY 4: Time-based situations (medium priority)
        if game_state.quarter and game_state.quarter >= 4:
            if game_state.time and self._is_two_minute_drill(game_state.time):
                return "two_minute_drill"
            else:
                return "fourth_quarter"

        # PRIORITY 5: Default (lowest priority)
        return "normal_play"

    def _calculate_pressure_level(self, game_state: GameState, context: SituationContext) -> str:
        """Calculate pressure level based on situation."""

        pressure_factors = 0

        # Down and distance pressure
        if game_state.down == 4:
            pressure_factors += 3
        elif game_state.down == 3:
            pressure_factors += 2

        # Distance pressure
        if game_state.distance and game_state.distance >= 10:
            pressure_factors += 1
        elif game_state.distance and game_state.distance >= 15:
            pressure_factors += 2

        # Field position pressure
        if context.possession_team == "user" and context.territory == "own":
            if game_state.yard_line and game_state.yard_line <= 10:
                pressure_factors += 2

        # Time pressure
        if game_state.quarter and game_state.quarter >= 4:
            if game_state.time and self._is_two_minute_drill(game_state.time):
                pressure_factors += 2

        # Score differential pressure
        if game_state.score_home is not None and game_state.score_away is not None:
            score_diff = abs(game_state.score_home - game_state.score_away)
            if score_diff <= 3:
                pressure_factors += 1
            elif score_diff <= 7:
                pressure_factors += 1

        # Classify pressure level
        if pressure_factors >= 6:
            return "critical"
        elif pressure_factors >= 4:
            return "high"
        elif pressure_factors >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_leverage_index(self, game_state: GameState, context: SituationContext) -> float:
        """Calculate situational leverage (importance) index."""

        base_leverage = 0.5

        # FIXED: Situation type modifiers with hierarchical specificity
        situation_modifiers = {
            # Critical down situations (highest leverage)
            "fourth_down_goal_line": 0.5,  # Most critical
            "fourth_down_red_zone": 0.45,  # Very critical
            "fourth_down": 0.3,  # Critical
            # Third down + field position combinations (high leverage)
            "third_and_long_red_zone": 0.35,  # Very specific and critical
            "third_and_short_goal_line": 0.4,  # Very specific and critical
            "third_and_short_red_zone": 0.35,  # Very specific and critical
            "third_down_red_zone": 0.3,  # Specific and important
            # Standard third down situations (medium-high leverage)
            "third_and_long": 0.2,  # Important
            "third_and_short": 0.25,  # Important
            "third_down": 0.2,  # Important
            # Field position situations (medium leverage)
            "goal_line_offense": 0.4,  # Very important
            "goal_line_defense": 0.4,  # Very important
            "red_zone_offense": 0.3,  # Important
            "red_zone_defense": 0.3,  # Important
            "backed_up_offense": 0.2,  # Moderate
            "pressure_defense": 0.2,  # Moderate
            "scoring_position": 0.15,  # Moderate
            # Time-based situations (medium leverage)
            "two_minute_drill": 0.4,  # Very important
            "fourth_quarter": 0.1,  # Moderate
            # Default (low leverage)
            "normal_play": 0.0,  # Baseline
        }

        base_leverage += situation_modifiers.get(context.situation_type, 0.0)

        # Pressure level modifiers
        pressure_modifiers = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.0}

        base_leverage += pressure_modifiers.get(context.pressure_level, 0.0)

        # Quarter modifiers
        if game_state.quarter:
            if game_state.quarter >= 4:
                base_leverage += 0.2
            elif game_state.quarter >= 3:
                base_leverage += 0.1

        return min(base_leverage, 1.0)

    def _track_situation_for_mmr(self, game_state: GameState, context: SituationContext) -> None:
        """Track situation for hidden MMR analysis."""

        situation_data = {
            "timestamp": time.time(),
            "game_state": game_state,
            "context": context,
            "possession_team": context.possession_team,
            "situation_type": context.situation_type,
            "pressure_level": context.pressure_level,
            "leverage_index": context.leverage_index,
        }

        # Add to appropriate tracking category
        if context.possession_team == "user":
            self.performance_stats["offensive_situations"][context.situation_type].append(
                situation_data
            )
        elif context.possession_team == "opponent":
            self.performance_stats["defensive_situations"][context.situation_type].append(
                situation_data
            )

        # Add to general situation history
        self.situation_history.append(situation_data)

        # Keep only last 100 situations for memory management
        if len(self.situation_history) > 100:
            self.situation_history.pop(0)

    def track_possession_change(
        self, old_possession: str, new_possession: str, game_state: GameState
    ) -> None:
        """Track possession changes for turnover analysis."""

        change_data = {
            "timestamp": time.time(),
            "old_possession": old_possession,
            "new_possession": new_possession,
            "game_state": game_state,
            "field_position": game_state.yard_line,
            "territory": game_state.territory,
            "down": game_state.down,
            "distance": game_state.distance,
        }

        self.drive_tracking["possession_changes"].append(change_data)
        self.performance_stats["transition_moments"].append(change_data)

        # Update hidden MMR based on turnover context
        self._update_mmr_for_turnover(change_data)

    def _update_mmr_for_turnover(self, change_data: dict[str, Any]) -> None:
        """Update hidden MMR metrics based on turnover context."""

        # Determine if this was user gaining or losing possession
        user_gained_possession = (
            change_data["new_possession"] == self.user_team
            and change_data["old_possession"] != self.user_team
        )

        user_lost_possession = (
            change_data["old_possession"] == self.user_team
            and change_data["new_possession"] != self.user_team
        )

        if user_gained_possession:
            # User created turnover - positive for defensive performance
            self.hidden_mmr.momentum_management += 0.1
            self.hidden_mmr.pressure_performance += 0.05
        elif user_lost_possession:
            # User turned ball over - negative for turnover avoidance
            self.hidden_mmr.turnover_avoidance -= 0.1

            # Extra penalty if in good field position
            if change_data.get("territory") == "opponent":
                self.hidden_mmr.turnover_avoidance -= 0.05

    def get_hidden_performance_summary(self) -> dict[str, Any]:
        """Get hidden performance summary for internal analysis."""

        mmr_score = self.hidden_mmr.calculate_overall_mmr()
        performance_tier = self.hidden_mmr.get_performance_tier()

        return {
            "hidden_mmr_score": mmr_score,
            "performance_tier": performance_tier.value,
            "tier_name": performance_tier.name,
            "situational_breakdown": {
                "red_zone_efficiency": self.hidden_mmr.red_zone_efficiency,
                "third_down_conversion": self.hidden_mmr.third_down_conversion,
                "turnover_avoidance": self.hidden_mmr.turnover_avoidance,
                "pressure_performance": self.hidden_mmr.pressure_performance,
                "clutch_factor": self.hidden_mmr.clutch_factor,
            },
            "situation_counts": {
                "offensive_situations": len(self.performance_stats["offensive_situations"]),
                "defensive_situations": len(self.performance_stats["defensive_situations"]),
                "transition_moments": len(self.performance_stats["transition_moments"]),
            },
            "analysis_context": self.analysis_context,
        }

    def set_user_context(self, user_team: str, analysis_type: str = "self") -> None:
        """Set user context for proper possession/performance tracking."""
        self.user_team = user_team  # "home" or "away"
        self.analysis_context = analysis_type  # "self", "opponent", "pro_study"

        logger.info(f"User context set: team={user_team}, analysis={analysis_type}")

    def reset_performance_tracking(self) -> None:
        """Reset all performance tracking metrics."""
        self.hidden_mmr = HiddenMMRMetrics()
        self.performance_stats = {
            "offensive_situations": defaultdict(list),
            "defensive_situations": defaultdict(list),
            "transition_moments": [],
            "decision_quality": [],
        }
        self.situation_history.clear()
        self.drive_tracking = {"current_drive": None, "drive_history": [], "possession_changes": []}

        logger.info("🔄 Performance tracking metrics reset")

    # ==================== NEW 8-CLASS MODEL ENHANCEMENT METHODS ====================

    def _extract_down_distance_from_region(
        self, region_data: dict[str, Any], current_time: float = None
    ) -> Optional[dict[str, Any]]:
        """
        Extract down and distance using the NEW 8-class model's precise down_distance_area detection.
        This replaces the old unreliable coordinate-based approach.
        Uses temporal confidence voting for performance optimization.
        """
        try:
            # FIXED: Unified confidence system - always extract OCR, let appropriate system handle confidence
            is_burst_mode = current_time is None

            # FIXED: Make temporal manager advisory, not blocking - always allow fresh OCR as fallback
            temporal_suggests_skip = False
            if not is_burst_mode and hasattr(self, "temporal_manager"):
                temporal_suggests_skip = not self.temporal_manager.should_extract(
                    "down_distance", current_time
                )
                if temporal_suggests_skip:
                    # Try cached value first, but don't block fresh OCR if cache is insufficient
                    cached_result = self.temporal_manager.get_current_value("down_distance")
                    if cached_result and cached_result.get("value"):
                        cached_confidence = cached_result.get("value", {}).get("confidence", 0.0)
                        # Only use cache if confidence is high enough (>0.7) - otherwise do fresh OCR
                        if cached_confidence > 0.7:
                            logger.debug(
                                f"⏰ TEMPORAL CACHE: Using high-confidence cached down/distance result (conf={cached_confidence:.3f})"
                            )
                            return cached_result.get("value")
                        else:
                            logger.debug(
                                f"⏰ TEMPORAL OVERRIDE: Cached confidence too low ({cached_confidence:.3f}), performing fresh OCR"
                            )
                    else:
                        logger.debug(
                            f"⏰ TEMPORAL OVERRIDE: No cached result available, performing fresh OCR"
                        )

            roi = region_data["roi"]
            confidence = region_data["confidence"]

            # Check cache first for OCR result
            if self.cache_enabled and self.advanced_cache:
                cached_ocr = self.advanced_cache.get_ocr_result(
                    roi, "down_distance", "enhanced_multi"
                )
                if cached_ocr is not None:
                    logger.debug("⚡ Cache HIT: Using cached OCR result for down/distance")
                    return cached_ocr

            # FIXED: Scale up tiny regions for better OCR (burst sampling fix)
            if roi.shape[0] < 20 or roi.shape[1] < 60:
                # Scale up small regions by 5x for OCR with better interpolation
                scale_factor = 5
                new_height = roi.shape[0] * scale_factor
                new_width = roi.shape[1] * scale_factor
                roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # Apply additional preprocessing for scaled regions
                roi = self._enhance_scaled_region_for_ocr(roi)

                if current_time is None:  # Burst sampling mode
                    print(f"🔧 BURST: Scaled tiny region to {roi.shape} for better OCR")

            # Apply same preprocessing as triangle detection for consistency
            processed_roi = self._preprocess_region_for_ocr(roi)

            # Use the FIXED OCR engine extract_down_distance method (has character confusion fixes)
            down_distance_text = self.ocr.extract_down_distance(roi)

            # Performance logging when overriding temporal manager suggestion
            if temporal_suggests_skip:
                logger.debug(
                    f"⚡ PERFORMANCE: Temporal manager suggested skip, but performed fresh OCR anyway"
                )

            # Debug logging for burst sampling
            if current_time is None:
                print(f"🔍 BURST SAMPLING: OCR result = '{down_distance_text}'")

            if down_distance_text:
                # Parse the validated and corrected text
                parsed_result = self._parse_down_distance_text(down_distance_text)

                if parsed_result:
                    # Pure OCR mode - no predictive logic validation
                    # Use OCR result as-is without any game logic interference

                    # Add metadata
                    parsed_result["method"] = "hybrid_ocr_logic"
                    parsed_result["source"] = "8class_down_distance_area"
                    parsed_result["region_confidence"] = confidence
                    parsed_result["region_bbox"] = region_data["bbox"]

                    # Cache the OCR result
                    if self.cache_enabled and self.advanced_cache:
                        try:
                            self.advanced_cache.set_ocr_result(
                                roi, "down_distance", "enhanced_multi", parsed_result
                            )
                            logger.debug("💾 Cached down/distance OCR result")
                        except Exception as e:
                            logger.debug(f"OCR cache storage failed: {e}")

                    # FIXED: Unified result handling - add to appropriate confidence system
                    if not is_burst_mode and hasattr(self, "temporal_manager"):
                        # Normal mode: Add to temporal manager for time-based confidence voting
                        extraction_result = ExtractionResult(
                            value=parsed_result,
                            confidence=parsed_result["confidence"],
                            timestamp=current_time,
                            raw_text=down_distance_text,
                            method=parsed_result["method"],
                        )
                        self.temporal_manager.add_extraction_result(
                            "down_distance", extraction_result
                        )
                        logger.debug(f"⏰ TEMPORAL: Added down/distance result to temporal manager")
                    else:
                        # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                        logger.debug(
                            f"🎯 BURST: Down/distance result will be handled by burst consensus"
                        )

                    return parsed_result

            # Fallback to original method if robust extraction fails
            down_distance_results = []

            # Configuration 1: High-resolution processed region optimized
            config_1 = (
                r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndrdthgoalTHNDRDSTRD"
            )
            try:
                text_1 = pytesseract.image_to_string(processed_roi, config=config_1).strip()
                if text_1:
                    # Apply OCR corrections before parsing
                    corrected_text_1 = self._apply_down_distance_corrections(text_1)
                    parsed_1 = self._parse_down_distance_text(corrected_text_1)
                    if parsed_1:
                        parsed_1["method"] = "config_1_enhanced"
                        parsed_1["confidence"] = confidence * 0.85  # Lower confidence for fallback
                        down_distance_results.append(parsed_1)
            except Exception as e:
                logger.debug(f"OCR config 1 failed: {e}")

            # Select best result from fallback
            if down_distance_results:
                best_result = max(down_distance_results, key=lambda x: x["confidence"])

                # Add region information
                best_result["source"] = "8class_down_distance_area"
                best_result["region_confidence"] = confidence
                best_result["region_bbox"] = region_data["bbox"]

                # FIXED: Unified result handling - add to appropriate confidence system
                if not is_burst_mode and hasattr(self, "temporal_manager"):
                    # Normal mode: Add to temporal manager for time-based confidence voting
                    extraction_result = ExtractionResult(
                        value=best_result,
                        confidence=best_result["confidence"],
                        timestamp=current_time,
                        raw_text=best_result.get("raw_text", ""),
                        method=best_result.get("method", "tesseract"),
                    )
                    self.temporal_manager.add_extraction_result("down_distance", extraction_result)
                    logger.debug(
                        f"⏰ TEMPORAL: Added fallback down/distance result to temporal manager"
                    )
                else:
                    # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                    logger.debug(
                        f"🎯 BURST: Fallback down/distance result will be handled by burst consensus"
                    )

                return best_result

            # No logic-only fallback - pure OCR detection only

            return None

        except Exception as e:
            logger.error(f"🚨 EXCEPTION in _extract_down_distance_from_region: {e}")
            logger.error(f"🚨 Exception type: {type(e)}")
            import traceback

            logger.error(f"🚨 Traceback: {traceback.format_exc()}")
            logger.error(f"Error extracting down/distance from region: {e}")
            return None

    def _extract_game_clock_from_region(
        self, region_data: dict[str, Any], current_time: float = None
    ) -> Optional[dict[str, Any]]:
        """
        Extract game clock using the SAME successful pattern as down/distance.
        Uses robust multi-engine OCR with fallback and logic validation.
        """
        try:
            # FIXED: Unified confidence system - always extract OCR, let appropriate system handle confidence
            is_burst_mode = current_time is None

            # FIXED: Make temporal manager advisory, not blocking - always allow fresh OCR as fallback
            temporal_suggests_skip = False
            if not is_burst_mode and hasattr(self, "temporal_manager"):
                temporal_suggests_skip = not self.temporal_manager.should_extract(
                    "game_clock", current_time
                )
                if temporal_suggests_skip:
                    # Try cached value first, but don't block fresh OCR if cache is insufficient
                    cached_result = self.temporal_manager.get_current_value("game_clock")
                    if cached_result and cached_result.get("value"):
                        cached_confidence = cached_result.get("value", {}).get("confidence", 0.0)
                        # Only use cache if confidence is high enough (>0.7) - otherwise do fresh OCR
                        if cached_confidence > 0.7:
                            logger.debug(
                                f"⏰ TEMPORAL CACHE: Using high-confidence cached game clock result (conf={cached_confidence:.3f})"
                            )
                            return cached_result.get("value")
                        else:
                            logger.debug(
                                f"⏰ TEMPORAL OVERRIDE: Cached confidence too low ({cached_confidence:.3f}), performing fresh OCR"
                            )
                    else:
                        logger.debug(
                            f"⏰ TEMPORAL OVERRIDE: No cached result available, performing fresh OCR"
                        )

            roi = region_data["roi"]
            confidence = region_data["confidence"]

            # FIXED: Scale up tiny regions for better OCR (same as down/distance)
            if roi.shape[0] < 20 or roi.shape[1] < 60:
                # Scale up small regions by 5x for OCR with better interpolation
                scale_factor = 5
                new_height = roi.shape[0] * scale_factor
                new_width = roi.shape[1] * scale_factor
                roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # Apply additional preprocessing for scaled regions
                roi = self._enhance_scaled_region_for_ocr(roi)

                if current_time is None:  # Burst sampling mode
                    print(f"🔧 BURST: Scaled tiny game clock region to {roi.shape} for better OCR")

            # Apply same preprocessing as down/distance for consistency
            processed_roi = self._preprocess_region_for_ocr(roi)

            # Multi-engine OCR with the enhanced OCR system (same as down/distance)

            # Use robust multi-attempt extraction (same pattern as down/distance)
            game_clock_text = self._extract_game_clock_robust(roi)

            # Performance logging when overriding temporal manager suggestion
            if temporal_suggests_skip:
                logger.debug(
                    f"⚡ PERFORMANCE: Temporal manager suggested skip, but performed fresh OCR anyway"
                )

            # Debug logging for burst sampling
            if current_time is None:
                print(f"🔍 BURST SAMPLING: Game clock OCR result = '{game_clock_text}'")

            # ALWAYS log game clock extraction attempts for debugging
            print(
                f"🕐 GAME CLOCK EXTRACTION: Region shape={roi.shape}, Result='{game_clock_text}', Confidence={confidence:.3f}"
            )

            if game_clock_text:
                # Parse the validated and corrected text
                parsed_result = self._parse_game_clock_text(game_clock_text)

                if parsed_result:
                    # Add metadata (same as down/distance)
                    parsed_result["method"] = "robust_multi_engine"
                    parsed_result["source"] = "8class_game_clock_area"
                    parsed_result["region_confidence"] = confidence
                    parsed_result["region_bbox"] = region_data["bbox"]

                    # Log quarter extraction if found
                    if "quarter" in parsed_result:
                        print(
                            f"🏈 QUARTER EXTRACTED: Quarter {parsed_result['quarter']} from game clock region"
                        )

                    # FIXED: Unified result handling - add to appropriate confidence system
                    if not is_burst_mode and hasattr(self, "temporal_manager"):
                        # Normal mode: Add to temporal manager for time-based confidence voting
                        extraction_result = ExtractionResult(
                            value=parsed_result,
                            confidence=parsed_result.get("confidence", confidence),
                            timestamp=current_time,
                            raw_text=game_clock_text,
                            method=parsed_result["method"],
                        )
                        self.temporal_manager.add_extraction_result("game_clock", extraction_result)
                        logger.debug(f"⏰ TEMPORAL: Added game clock result to temporal manager")
                    else:
                        # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                        logger.debug(
                            f"🎯 BURST: Game clock result will be handled by burst consensus"
                        )

                    return parsed_result

            # Fallback to original method if robust extraction fails (same as down/distance)
            game_clock_results = []

            # Configuration 1: High-resolution processed region optimized
            config_1 = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:"
            try:
                text_1 = pytesseract.image_to_string(processed_roi, config=config_1).strip()
                if text_1:
                    # Apply OCR corrections before parsing
                    corrected_text_1 = self._apply_game_clock_corrections(text_1)
                    parsed_1 = self._parse_game_clock_text(corrected_text_1)
                    if parsed_1:
                        parsed_1["method"] = "config_1_enhanced"
                        parsed_1["confidence"] = confidence * 0.85  # Lower confidence for fallback
                        game_clock_results.append(parsed_1)
            except Exception as e:
                logger.debug(f"Game clock OCR config 1 failed: {e}")

            # Select best result from fallback
            if game_clock_results:
                best_result = max(game_clock_results, key=lambda x: x.get("confidence", 0))

                # Add region information
                best_result["source"] = "8class_game_clock_area"
                best_result["region_confidence"] = confidence
                best_result["region_bbox"] = region_data["bbox"]

                # FIXED: Unified result handling - add to appropriate confidence system
                if not is_burst_mode and hasattr(self, "temporal_manager"):
                    # Normal mode: Add to temporal manager for time-based confidence voting
                    extraction_result = ExtractionResult(
                        value=best_result,
                        confidence=best_result.get("confidence", confidence),
                        timestamp=current_time,
                        raw_text=best_result.get("raw_text", ""),
                        method=best_result.get("method", "tesseract"),
                    )
                    self.temporal_manager.add_extraction_result("game_clock", extraction_result)
                    logger.debug(
                        f"⏰ TEMPORAL: Added fallback game clock result to temporal manager"
                    )
                else:
                    # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                    logger.debug(
                        f"🎯 BURST: Fallback game clock result will be handled by burst consensus"
                    )

                return best_result

            # LOGIC-ONLY FALLBACK: When OCR completely fails, use game context
            # Game clock can be estimated from quarter and game flow
            # No logic-only fallback - pure OCR detection only

            return None

        except Exception as e:
            logger.error(f"Error extracting game clock from region: {e}")
            return None

    def _extract_play_clock_from_region(
        self, region_data: dict[str, Any], current_time: float = None
    ) -> Optional[dict[str, Any]]:
        """
        Extract play clock using the SAME successful pattern as down/distance.
        Uses robust multi-engine OCR with fallback and logic validation.
        """
        try:
            # FIXED: Unified confidence system - always extract OCR, let appropriate system handle confidence
            is_burst_mode = current_time is None

            # FIXED: Make temporal manager advisory, not blocking - always allow fresh OCR as fallback
            temporal_suggests_skip = False
            if not is_burst_mode and hasattr(self, "temporal_manager"):
                temporal_suggests_skip = not self.temporal_manager.should_extract(
                    "play_clock", current_time
                )
                if temporal_suggests_skip:
                    # Try cached value first, but don't block fresh OCR if cache is insufficient
                    cached_result = self.temporal_manager.get_current_value("play_clock")
                    if cached_result and cached_result.get("value"):
                        cached_confidence = cached_result.get("value", {}).get("confidence", 0.0)
                        # Only use cache if confidence is high enough (>0.7) - otherwise do fresh OCR
                        if cached_confidence > 0.7:
                            logger.debug(
                                f"⏰ TEMPORAL CACHE: Using high-confidence cached play clock result (conf={cached_confidence:.3f})"
                            )
                            return cached_result.get("value")
                        else:
                            logger.debug(
                                f"⏰ TEMPORAL OVERRIDE: Cached confidence too low ({cached_confidence:.3f}), performing fresh OCR"
                            )
                    else:
                        logger.debug(
                            f"⏰ TEMPORAL OVERRIDE: No cached result available, performing fresh OCR"
                        )

            roi = region_data["roi"]
            confidence = region_data["confidence"]

            # FIXED: Scale up tiny regions for better OCR (same as down/distance)
            if roi.shape[0] < 20 or roi.shape[1] < 40:
                # Scale up small regions by 5x for OCR with better interpolation
                scale_factor = 5
                new_height = roi.shape[0] * scale_factor
                new_width = roi.shape[1] * scale_factor
                roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # Apply additional preprocessing for scaled regions
                roi = self._enhance_scaled_region_for_ocr(roi)

                if current_time is None:  # Burst sampling mode
                    print(f"🔧 BURST: Scaled tiny play clock region to {roi.shape} for better OCR")

            # Apply same preprocessing as down/distance for consistency
            processed_roi = self._preprocess_region_for_ocr(roi)

            # Multi-engine OCR with the enhanced OCR system (same as down/distance)

            # Use robust multi-attempt extraction (same pattern as down/distance)
            play_clock_text = self._extract_play_clock_robust(roi)

            # Performance logging when overriding temporal manager suggestion
            if temporal_suggests_skip:
                logger.debug(
                    f"⚡ PERFORMANCE: Temporal manager suggested skip, but performed fresh OCR anyway"
                )

            # Debug logging for burst sampling
            if current_time is None:
                print(f"🔍 BURST SAMPLING: Play clock OCR result = '{play_clock_text}'")

            if play_clock_text:
                # Parse the validated and corrected text
                parsed_result = self._parse_play_clock_text(play_clock_text)

                if parsed_result:
                    # Add metadata (same as down/distance)
                    parsed_result["method"] = "robust_multi_engine"
                    parsed_result["source"] = "8class_play_clock_area"
                    parsed_result["region_confidence"] = confidence
                    parsed_result["region_bbox"] = region_data["bbox"]

                    # FIXED: Unified result handling - add to appropriate confidence system
                    if not is_burst_mode and hasattr(self, "temporal_manager"):
                        # Normal mode: Add to temporal manager for time-based confidence voting
                        extraction_result = ExtractionResult(
                            value=parsed_result,
                            confidence=parsed_result.get("confidence", confidence),
                            timestamp=current_time,
                            raw_text=play_clock_text,
                            method=parsed_result["method"],
                        )
                        self.temporal_manager.add_extraction_result("play_clock", extraction_result)
                        logger.debug(f"⏰ TEMPORAL: Added play clock result to temporal manager")
                    else:
                        # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                        logger.debug(
                            f"🎯 BURST: Play clock result will be handled by burst consensus"
                        )

                    return parsed_result

            # Fallback to original method if robust extraction fails (same as down/distance)
            play_clock_results = []

            # Configuration 1: High-resolution processed region optimized
            config_1 = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
            try:
                text_1 = pytesseract.image_to_string(processed_roi, config=config_1).strip()
                if text_1:
                    # Apply OCR corrections before parsing
                    corrected_text_1 = self._apply_play_clock_corrections(text_1)
                    parsed_1 = self._parse_play_clock_text(corrected_text_1)
                    if parsed_1:
                        parsed_1["method"] = "config_1_enhanced"
                        parsed_1["confidence"] = confidence * 0.85  # Lower confidence for fallback
                        play_clock_results.append(parsed_1)
            except Exception as e:
                logger.debug(f"Play clock OCR config 1 failed: {e}")

            # Select best result from fallback
            if play_clock_results:
                best_result = max(play_clock_results, key=lambda x: x.get("confidence", 0))

                # Add region information
                best_result["source"] = "8class_play_clock_area"
                best_result["region_confidence"] = confidence
                best_result["region_bbox"] = region_data["bbox"]

                # FIXED: Unified result handling - add to appropriate confidence system
                if not is_burst_mode and hasattr(self, "temporal_manager"):
                    # Normal mode: Add to temporal manager for time-based confidence voting
                    extraction_result = ExtractionResult(
                        value=best_result,
                        confidence=best_result.get("confidence", confidence),
                        timestamp=current_time,
                        raw_text=best_result.get("raw_text", ""),
                        method=best_result.get("method", "tesseract"),
                    )
                    self.temporal_manager.add_extraction_result("play_clock", extraction_result)
                    logger.debug(
                        f"⏰ TEMPORAL: Added fallback play clock result to temporal manager"
                    )
                else:
                    # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                    logger.debug(
                        f"🎯 BURST: Fallback play clock result will be handled by burst consensus"
                    )

                return best_result

            # No logic-only fallback - pure OCR detection only

            return None

        except Exception as e:
            logger.error(f"Error extracting play clock from region: {e}")
            return None

    def _preprocess_region_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Use MAIN OCR preprocessing with optimal 0.939 score parameters from 20K sweep.
        This delegates to our proven EnhancedOCR.preprocess_image() method.
        """
        try:
            # Use the MAIN OCR preprocessing pipeline with optimal parameters
            # This includes: Scale=3.5x, CLAHE clip=1.0 grid=(4,4), Blur=(3,3),
            # Threshold=adaptive_mean block=13 C=3, Morphological=(3,3), Gamma=0.8
            return self.ocr.preprocess_image(roi)

        except Exception as e:
            logger.error(f"Error using main OCR preprocessing: {e}")
            # Fallback: minimal processing only if main OCR fails
            try:
                if len(roi.shape) == 3:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray = roi
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            except:
                return roi

    def _parse_down_distance_text(self, text: str) -> Optional[dict[str, Any]]:
        """Parse down and distance from OCR text with enhanced pattern matching including PAT."""
        import re

        # Apply corrections first to handle OCR mistakes
        corrected_text = self._apply_down_distance_corrections(text)

        # Special case: PAT (Point After Touchdown) - check both original and corrected
        if re.search(r"\bPAT\b", corrected_text, re.IGNORECASE):
            return {
                "down": None,  # PAT doesn't have a down
                "distance": None,  # PAT doesn't have distance
                "distance_type": "pat",
                "raw_text": text,
                "corrected_text": corrected_text,
                "is_pat": True,
            }

        # FIXED: Case-insensitive patterns for down and distance
        patterns = [
            r"(\d+)(?:st|nd|rd|th|ST|ND|RD|TH)?\s*&\s*(\d+)",  # "1st & 10", "1ST & 10", "3rd & 8"
            r"(\d+)(?:st|nd|rd|th|ST|ND|RD|TH)?\s*&\s*(?:Goal|GOAL)",  # "1st & Goal", "1ST & GOAL"
            r"(\d+)(?:st|nd|rd|th|ST|ND|RD|TH)?\s*&\s*(?:G|g)",  # "4th & G", "4TH & G"
            r"(\d+)\s*&\s*(\d+)",  # "3 & 8"
            r"(\d+)(?:ND|RD|TH|ST|nd|rd|th|st)\s*&\s*(\d+)",  # All OCR variations
        ]

        for pattern in patterns:
            match = re.search(pattern, corrected_text, re.IGNORECASE)
            if match:
                try:
                    down = int(match.group(1))
                    if 1 <= down <= 4:  # Valid down
                        # Check for goal line situation
                        if "goal" in corrected_text.lower() or "g" in match.group(2).lower():
                            result = {
                                "down": down,
                                "distance": 0,
                                "distance_type": "goal",
                                "raw_text": text,
                                "corrected_text": corrected_text,
                                "is_pat": False,
                            }
                        else:
                            try:
                                distance = int(match.group(2))
                                result = {
                                    "down": down,
                                    "distance": distance,
                                    "distance_type": "yards",
                                    "raw_text": text,
                                    "corrected_text": corrected_text,
                                    "is_pat": False,
                                }
                            except ValueError:
                                continue

                        return result

                except (ValueError, IndexError):
                    continue

        return None

    def _parse_game_clock_text(self, text: str) -> Optional[dict[str, Any]]:
        """Parse game clock text (MM:SS format) and quarter information."""
        import re

        # Parse time (MM:SS format)
        time_pattern = r"(\d{1,2}):(\d{2})"
        time_match = re.search(time_pattern, text)

        # Parse quarter information
        quarter_patterns = [
            r"\b(1ST|1st|1)\b",  # 1st quarter
            r"\b(2ND|2nd|2)\b",  # 2nd quarter
            r"\b(3RD|3rd|3)\b",  # 3rd quarter
            r"\b(4TH|4th|4)\b",  # 4th quarter
            r"\b(OT|OVERTIME)\b",  # Overtime
        ]

        quarter = None
        for i, pattern in enumerate(quarter_patterns, 1):
            if re.search(pattern, text.upper()):
                quarter = i if i <= 4 else 5  # OT = 5
                break

        if time_match:
            try:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))

                if 0 <= minutes <= 15 and 0 <= seconds <= 59:
                    result = {
                        "game_clock": f"{minutes:02d}:{seconds:02d}",
                        "minutes": minutes,
                        "seconds": seconds,
                        "total_seconds": minutes * 60 + seconds,
                        "raw_text": text,
                        "confidence": 0.9,
                    }

                    # Add quarter if found
                    if quarter:
                        result["quarter"] = quarter
                        result["confidence"] = 0.95  # Higher confidence with quarter info
                        print(f"🏈 QUARTER DETECTED: Quarter {quarter} from text '{text}'")

                    return result
            except ValueError:
                pass

        # If no time found but quarter detected, still return quarter info
        if quarter:
            print(f"🏈 QUARTER-ONLY DETECTED: Quarter {quarter} from text '{text}'")
            return {"quarter": quarter, "raw_text": text, "confidence": 0.7}

        return None

    def _parse_play_clock_text(self, text: str) -> Optional[dict[str, Any]]:
        """Parse play clock text (usually just seconds)."""
        import re

        # Play clock is typically 0-40 seconds
        number_pattern = r"(\d{1,2})"
        match = re.search(number_pattern, text)

        if match:
            try:
                seconds = int(match.group(1))

                if 0 <= seconds <= 40:  # Valid play clock range
                    return {"play_clock": seconds, "raw_text": text, "confidence": 0.85}
            except ValueError:
                pass

        return None

    def _extract_scores_from_possession_region(
        self, region_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Extract team scores from the possession_triangle_area region.
        This region contains: [AWAY_TEAM] [AWAY_SCORE] [→] [HOME_SCORE] [HOME_TEAM]
        """
        try:
            roi = region_data["roi"]
            confidence = region_data["confidence"]

            # Preprocess for score text
            processed_roi = self._preprocess_region_for_ocr(roi)

            # Score-specific OCR configuration
            score_config = (
                r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            )

            try:
                score_text = pytesseract.image_to_string(processed_roi, config=score_config).strip()
                if score_text:
                    parsed_scores = self._parse_score_text(score_text)
                    if parsed_scores:
                        parsed_scores["source"] = "8class_possession_triangle_area"
                        parsed_scores["region_confidence"] = confidence
                        parsed_scores["region_bbox"] = region_data["bbox"]
                        return parsed_scores
            except Exception:
                pass

            return None

        except Exception as e:
            logger.error(f"Error extracting scores from possession region: {e}")
            return None

    def _parse_score_text(self, text: str) -> Optional[dict[str, Any]]:
        """
        Parse team scores and abbreviations from possession region text.
        Expected format: "AWAY_TEAM AWAY_SCORE HOME_SCORE HOME_TEAM" or similar variations
        """
        import re

        # Clean up the text
        cleaned_text = re.sub(r"[^\w\s]", " ", text).strip()
        parts = cleaned_text.split()

        if len(parts) >= 4:
            # Try to identify scores (numbers) and team names (letters)
            scores = []
            teams = []

            for part in parts:
                if part.isdigit() and 0 <= int(part) <= 99:  # Valid score range
                    scores.append(int(part))
                elif part.isalpha() and 2 <= len(part) <= 4:  # Valid team abbreviation
                    teams.append(part.upper())

            # We need exactly 2 scores and 2 teams
            if len(scores) == 2 and len(teams) == 2:
                return {
                    "away_team": teams[0],
                    "away_score": scores[0],
                    "home_team": teams[1],
                    "home_score": scores[1],
                    "raw_text": text,
                    "confidence": 0.8,
                }

        # Alternative pattern matching for more complex cases
        score_patterns = [
            r"([A-Z]{2,4})\s*(\d{1,2})\s*(\d{1,2})\s*([A-Z]{2,4})",  # "TEAM1 21 14 TEAM2"
            r"([A-Z]{2,4})\s*(\d{1,2})\s*-\s*(\d{1,2})\s*([A-Z]{2,4})",  # "TEAM1 21 - 14 TEAM2"
        ]

        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    away_team = match.group(1)
                    away_score = int(match.group(2))
                    home_score = int(match.group(3))
                    home_team = match.group(4)

                    if 0 <= away_score <= 99 and 0 <= home_score <= 99:
                        return {
                            "away_team": away_team,
                            "away_score": away_score,
                            "home_team": home_team,
                            "home_score": home_score,
                            "raw_text": text,
                            "confidence": 0.75,
                        }
                except (ValueError, IndexError):
                    continue

        return None

    def _apply_down_distance_corrections(self, text: str) -> str:
        """Apply OCR corrections specific to down/distance text patterns including PAT."""
        import re

        # Convert to uppercase for consistent processing
        corrected = text.upper().strip()

        # Special case: PAT corrections
        pat_corrections = {
            "P4T": "PAT",
            "P8T": "PAT",
            "PRT": "PAT",
            "P@T": "PAT",
            "PAI": "PAT",
            "P4I": "PAT",
            "P8I": "PAT",
        }

        for mistake, correction in pat_corrections.items():
            if mistake in corrected:
                return correction

        # Check for partial PAT patterns
        if re.search(r"\bP[A4@8][TI]\b", corrected):
            return "PAT"

        # CONSERVATIVE APPROACH: Only apply corrections if confidence is low
        # or if we have clear indicators that correction is needed

        # First, check if the text already looks reasonable
        if re.search(r"[1-4](ST|ND|RD|TH)\s*&\s*\d+", corrected):
            # Text already looks good, minimal correction
            return self._minimal_corrections(corrected)

        # SMART CORRECTION: Look for the actual down number first
        # Extract any digit that appears before common OCR mistakes
        down_number = None
        down_match = re.search(r"(\d)", corrected)
        if down_match:
            down_number = down_match.group(1)

        # If we found a down number, use it to guide corrections
        if down_number and down_number in ["1", "2", "3", "4"]:
            # Build the correct ordinal based on the detected number
            ordinal_map = {"1": "1ST", "2": "2ND", "3": "3RD", "4": "4TH"}
            correct_ordinal = ordinal_map[down_number]

            # LESS AGGRESSIVE: Only replace obvious OCR mistakes, not everything
            mistake_patterns = [
                (
                    r"\b[A-Z]*[TAEL][AELT][TAEL]?\b",
                    correct_ordinal,
                ),  # Only if it looks like ordinal mistake
                (r"\b[A-Z]*[SZ][NR][DT]\b", correct_ordinal),  # Only if it looks like 2ND mistake
                (r"\b[A-Z]*[SB][RD][DT]\b", correct_ordinal),  # Only if it looks like 3RD mistake
                (r"\b[A-Z]*[AT][TH][TH]?\b", correct_ordinal),  # Only if it looks like 4TH mistake
            ]

            for pattern, replacement in mistake_patterns:
                # Only apply if the pattern matches something that looks like an ordinal
                if (
                    re.search(pattern, corrected)
                    and len(re.search(pattern, corrected).group()) <= 4
                ):
                    corrected = re.sub(pattern, replacement, corrected)
                    break  # Only apply one correction to avoid over-correction
        else:
            # Fallback: VERY conservative corrections only for clear mistakes
            conservative_corrections = {
                "TET": "1ST",  # Only if no digit found and looks like 1ST
                "TAT": "1ST",  # Only if no digit found and looks like 1ST
            }

            # Only apply if the text is very short and looks like a clear mistake
            if len(corrected.replace(" ", "").replace("&", "")) <= 6:
                for mistake, correction in conservative_corrections.items():
                    if mistake in corrected:
                        corrected = corrected.replace(mistake, correction)
                        break  # Only one correction

        # Apply minimal structural fixes
        corrected = self._minimal_corrections(corrected)

        return corrected

    def _minimal_corrections(self, text: str) -> str:
        """Apply only essential structural corrections."""
        import re

        # Fix spacing around &
        text = re.sub(r"(\w+)\s*&\s*(\d+)", r"\1 & \2", text)

        # Fix separated ordinals (1 ST → 1ST)
        text = re.sub(r"(\d)\s+(ST|ND|RD|TH)", r"\1\2", text)

        # Fix common & symbol mistakes only when clearly between ordinal and number
        text = re.sub(r"(1ST|2ND|3RD|4TH)\s*[A&S8B]\s*(\d+)", r"\1 & \2", text)

        # Fix GOAL patterns
        text = re.sub(r"(\d+)(ST|ND|RD|TH)?\s*&\s*(GOAL|G)", r"\1\2 & GOAL", text)

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_down_distance_robust(self, region: np.ndarray) -> Optional[str]:
        """
        SPEED-OPTIMIZED robust down/distance extraction.
        Reduced attempts for faster processing while maintaining accuracy.
        """
        import re

        logger.debug(f"🔍 Starting robust OCR extraction on region shape: {region.shape}")

        # OPTIMAL PREPROCESSING: Use 0.939 score parameters for ALL attempts
        attempts = [
            # Attempt 1: OPTIMAL preprocessing + Tesseract PSM 7
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 7 -c tessedit_char_whitelist=0123456789STNDRDTHGOALP&AT ",
                "description": "OPTIMAL + PSM7",
            },
            # Attempt 2: OPTIMAL preprocessing + Tesseract PSM 8
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 8 -c tessedit_char_whitelist=0123456789STNDRDTHGOALP&AT ",
                "description": "OPTIMAL + PSM8",
            },
            # Attempt 3: OPTIMAL preprocessing + EasyOCR
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "easyocr",
                "description": "OPTIMAL + EasyOCR",
            },
        ]

        results = []

        logger.debug(f"🔍 Attempting {len(attempts)} OCR methods")

        for attempt in attempts:
            try:
                logger.debug(f"🔍 Trying OCR attempt: {attempt['description']}")

                # Apply preprocessing
                processed = attempt["preprocess"](region)

                # Extract text
                if attempt["config"] == "easyocr":
                    # Use EasyOCR - FIXED: Use self.ocr which is the actual OCR engine
                    if hasattr(self, "ocr") and hasattr(self.ocr, "reader") and self.ocr.reader:
                        logger.debug(f"🔍 Using self.ocr for EasyOCR")
                        ocr_results = self.ocr.reader.readtext(processed)
                        logger.debug(f"🔍 EasyOCR found {len(ocr_results)} results")
                        for bbox, text, conf in ocr_results:
                            logger.debug(f"🔍 EasyOCR result: '{text}' (conf: {conf:.3f})")
                            # FIXED: Much lower threshold for scaled regions
                            min_conf = (
                                0.1 if processed.shape[0] > 50 else 0.05
                            )  # Very low for scaled regions
                            if conf > min_conf:
                                results.append((text.strip(), conf, attempt["description"]))
                                logger.debug(
                                    f"✅ Added EasyOCR result: '{text.strip()}' (conf: {conf:.3f})"
                                )
                            else:
                                logger.debug(
                                    f"❌ EasyOCR result below threshold: '{text}' (conf: {conf:.3f} < {min_conf})"
                                )
                    else:
                        logger.debug(
                            f"❌ No EasyOCR engine available (self.ocr not found or no reader)"
                        )
                else:
                    # Use Tesseract
                    import pytesseract

                    logger.debug(f"🔍 Using Tesseract with config: {attempt['config']}")
                    text = pytesseract.image_to_string(processed, config=attempt["config"]).strip()
                    logger.debug(f"🔍 Tesseract result: '{text}'")
                    if text:
                        # Estimate confidence based on text quality
                        conf = self._estimate_text_confidence(text)
                        results.append((text, conf, attempt["description"]))
                        logger.debug(f"✅ Added Tesseract result: '{text}' (conf: {conf:.3f})")

                        # SPEED OPTIMIZATION: Early exit if we get a high-confidence result
                        if conf > 0.8:
                            logger.debug(f"⚡ Early exit with high confidence: {conf:.2f}")
                            break
                    else:
                        logger.debug(f"❌ Tesseract returned empty text")

            except Exception as e:
                logger.debug(f"❌ OCR attempt {attempt['description']} failed: {e}")
                continue

        # Process results and find the best match
        logger.debug(f"🔍 Processing {len(results)} OCR results")
        best_result = None
        best_score = 0

        for text, conf, source in results:
            logger.debug(f"🔍 Processing result: '{text}' from {source}")
            # Apply corrections
            corrected = self._apply_down_distance_corrections(text)
            logger.debug(f"🔍 After corrections: '{corrected}'")

            # Validate the result
            is_valid = self._validate_down_distance(corrected)
            logger.debug(f"🔍 Validation result: {is_valid}")

            if is_valid:
                # Score based on confidence and text quality
                quality_score = self._calculate_text_quality_score(corrected)
                score = conf * quality_score
                logger.debug(f"🔍 Quality score: {quality_score:.3f}, Combined score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_result = corrected
                    logger.debug(
                        f"✅ New best down/distance from {source}: '{corrected}' (score: {score:.2f})"
                    )
            else:
                logger.debug(f"❌ Validation failed for '{corrected}'")

        if best_result:
            logger.debug(f"🎯 Final robust OCR result: '{best_result}'")
        else:
            logger.debug(f"❌ No valid OCR results found")

        return best_result

    # REMOVED: Obsolete custom preprocessing methods
    # All OCR operations now use self.ocr.preprocess_image() with optimal 0.939 score parameters
    # - _preprocess_standard() - REPLACED with optimal preprocessing
    # - _preprocess_high_contrast() - REPLACED with optimal preprocessing
    # - _preprocess_enlarged() - REPLACED with optimal preprocessing
    # - _preprocess_binary() - REPLACED with optimal preprocessing

    def _estimate_text_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics."""
        import re

        # Base confidence
        confidence = 0.5

        # Bonus for containing expected patterns
        if re.search(r"[1-4]", text):
            confidence += 0.2
        if re.search(r"(ST|ND|RD|TH)", text):
            confidence += 0.2
        if re.search(r"&", text):
            confidence += 0.1
        if re.search(r"\d+", text):
            confidence += 0.1

        # Penalty for too much noise
        if len(text) > 20:
            confidence -= 0.2
        if re.search(r"[^A-Z0-9\s&]", text):
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate quality score for down/distance text."""
        import re

        score = 0.0

        # Perfect pattern match gets highest score
        if re.match(r"^[1-4](ST|ND|RD|TH) & \d+$", text):
            score = 1.0
        elif re.match(r"^[1-4](ST|ND|RD|TH) & GOAL$", text):
            score = 1.0
        elif re.search(r"[1-4](ST|ND|RD|TH)", text) and "&" in text:
            score = 0.8
        elif re.search(r"[1-4]", text) and "&" in text:
            score = 0.6
        elif re.search(r"[1-4]", text):
            score = 0.4
        else:
            score = 0.1

        return score

    def _enhance_scaled_region_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Apply specialized enhancement for scaled-up regions to improve OCR accuracy.

        Args:
            roi: Scaled region image

        Returns:
            Enhanced region optimized for OCR
        """
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)

            # Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Apply sharpening kernel
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # Use adaptive threshold for better binarization
            binary = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Clean up small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Convert back to BGR for OCR engines
            enhanced_roi = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

            return enhanced_roi

        except Exception as e:
            logger.error(f"Error enhancing scaled region: {e}")
            return roi

    def _validate_down_distance(self, text: str) -> bool:
        """FIXED: More flexible validation for down/distance text including PAT."""
        import re

        if not text or len(text.strip()) == 0:
            return False

        # Special case: PAT (Point After Touchdown) - exact match
        if re.search(r"\bPAT\b", text, re.IGNORECASE):
            return True

        # Special case: PAT-like patterns that could be OCR mistakes
        pat_like_patterns = [
            r"\bP[A4@8][TI]\b",  # P4T, P8T, PAI, etc.
            r"\bPRT\b",  # PRT
        ]

        for pattern in pat_like_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        # Must contain a down number (1-4)
        if not re.search(r"[1-4]", text):
            return False

        # Must contain & or AND
        if not re.search(r"(&|AND)", text):
            return False

        # FIXED: More flexible patterns with optional spaces and case insensitive
        valid_patterns = [
            r"[1-4]\s*(ST|ND|RD|TH)?\s*&\s*\d+",  # 1ST & 10, 1 & 10, 1ST& 10, etc.
            r"[1-4]\s*(ST|ND|RD|TH)?\s*&\s*GOAL",  # 1ST & GOAL, 1 & GOAL, etc.
            r"[1-4]\s*(ST|ND|RD|TH)?\s*&\s*G",  # 1ST & G, 1 & G, etc.
        ]

        for pattern in valid_patterns:
            if re.search(pattern, text, re.IGNORECASE):  # Case insensitive matching
                return True

        return False

    def _extract_game_clock_robust(self, region: np.ndarray) -> Optional[str]:
        """
        Robust game clock extraction using the same pattern as down/distance.
        Multi-engine OCR with corrections and validation.
        """
        import re

        logger.debug(f"🔍 Starting robust game clock extraction on region shape: {region.shape}")

        # OPTIMAL PREPROCESSING: Use 0.939 score parameters for game clock
        attempts = [
            # Attempt 1: OPTIMAL preprocessing + Tesseract PSM 8
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 8 -c tessedit_char_whitelist=0123456789: ",
                "description": "OPTIMAL + PSM8 clock",
            },
            # Attempt 2: OPTIMAL preprocessing + Tesseract PSM 7
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 7 -c tessedit_char_whitelist=0123456789: ",
                "description": "OPTIMAL + PSM7 clock",
            },
            # Attempt 3: OPTIMAL preprocessing + EasyOCR
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "easyocr",
                "description": "OPTIMAL + EasyOCR clock",
            },
        ]

        results = []

        logger.debug(f"🔍 Attempting {len(attempts)} game clock OCR methods")

        for attempt in attempts:
            try:
                logger.debug(f"🔍 Trying game clock attempt: {attempt['description']}")

                # Apply preprocessing
                processed = attempt["preprocess"](region)

                # Extract text
                if attempt["config"] == "easyocr":
                    # Use EasyOCR - FIXED: Use self.ocr which is the actual OCR engine
                    if hasattr(self, "ocr") and hasattr(self.ocr, "reader") and self.ocr.reader:
                        logger.debug(f"🔍 Using self.ocr for game clock EasyOCR")
                        ocr_results = self.ocr.reader.readtext(processed)
                        logger.debug(f"🔍 Game clock EasyOCR found {len(ocr_results)} results")
                        for bbox, text, conf in ocr_results:
                            logger.debug(
                                f"🔍 Game clock EasyOCR result: '{text}' (conf: {conf:.3f})"
                            )
                            min_conf = 0.1 if processed.shape[0] > 50 else 0.05
                            if conf > min_conf:
                                results.append((text.strip(), conf, attempt["description"]))
                                logger.debug(
                                    f"✅ Added game clock EasyOCR result: '{text.strip()}' (conf: {conf:.3f})"
                                )
                    else:
                        logger.debug(f"❌ No EasyOCR engine available for game clock")
                else:
                    # Use Tesseract
                    import pytesseract

                    logger.debug(
                        f"🔍 Using Tesseract for game clock with config: {attempt['config']}"
                    )
                    text = pytesseract.image_to_string(processed, config=attempt["config"]).strip()
                    logger.debug(f"🔍 Game clock Tesseract result: '{text}'")
                    if text:
                        # Estimate confidence based on text quality
                        conf = self._estimate_game_clock_confidence(text)
                        results.append((text, conf, attempt["description"]))
                        logger.debug(
                            f"✅ Added game clock Tesseract result: '{text}' (conf: {conf:.3f})"
                        )

                        # Early exit if we get a high-confidence result
                        if conf > 0.8:
                            logger.debug(
                                f"⚡ Early exit with high game clock confidence: {conf:.2f}"
                            )
                            break
                    else:
                        logger.debug(f"❌ Game clock Tesseract returned empty text")

            except Exception as e:
                logger.debug(f"❌ Game clock OCR attempt {attempt['description']} failed: {e}")
                continue

        # Process results and find the best match
        logger.debug(f"🔍 Processing {len(results)} game clock OCR results")
        best_result = None
        best_score = 0
        temporal_candidates = []  # Store candidates for temporal validation

        for text, conf, source in results:
            logger.debug(f"🔍 Processing game clock result: '{text}' from {source}")
            # Apply corrections
            corrected = self._apply_game_clock_corrections(text)
            logger.debug(f"🔍 After game clock corrections: '{corrected}'")

            # Validate the result
            is_valid = self._validate_game_clock(corrected)
            logger.debug(f"🔍 Game clock validation result: {is_valid}")

            if is_valid:
                # Score based on confidence and text quality
                quality_score = self._calculate_game_clock_quality_score(corrected)
                score = conf * quality_score
                logger.debug(
                    f"🔍 Game clock quality score: {quality_score:.3f}, Combined score: {score:.3f}"
                )

                # Add temporal validation
                is_temporal_valid, temporal_reason = self._validate_game_clock_temporal(corrected)
                logger.debug(f"⏰ Temporal validation: {is_temporal_valid} - {temporal_reason}")

                if is_temporal_valid:
                    # Temporally valid candidate
                    if score > best_score:
                        best_score = score
                        best_result = corrected
                        logger.debug(
                            f"✅ New best temporally valid game clock from {source}: '{corrected}' (score: {score:.2f})"
                        )
                else:
                    # Store temporally invalid candidates for fallback
                    temporal_candidates.append((corrected, score, source, temporal_reason))
                    logger.debug(
                        f"⚠️ Temporally invalid but OCR-valid: '{corrected}' from {source} - {temporal_reason}"
                    )
            else:
                logger.debug(f"❌ Game clock validation failed for '{corrected}'")

        # If no temporally valid result found, use the best OCR result but log the issue
        if not best_result and temporal_candidates:
            # Sort by score and take the best OCR result despite temporal issues
            temporal_candidates.sort(key=lambda x: x[1], reverse=True)
            best_result, best_score, source, temporal_reason = temporal_candidates[0]
            logger.warning(
                f"⚠️ Using temporally invalid game clock '{best_result}' from {source} - {temporal_reason}"
            )
            logger.warning(f"⚠️ This suggests OCR error in current or previous frame")

        if best_result:
            # Update history with the accepted result
            self._update_game_clock_history(best_result)
            logger.debug(f"🎯 Final robust game clock result: '{best_result}'")
        else:
            logger.debug(f"❌ No valid game clock results found")

        return best_result

    def _extract_play_clock_robust(self, region: np.ndarray) -> Optional[str]:
        """
        Robust play clock extraction using the same pattern as game clock.
        Multi-engine OCR with corrections and validation.
        """
        import re

        logger.debug(f"🔍 Starting robust play clock extraction on region shape: {region.shape}")

        # OPTIMAL PREPROCESSING: Use 0.939 score parameters for play clock
        attempts = [
            # Attempt 1: OPTIMAL preprocessing + Tesseract PSM 8
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 8 -c tessedit_char_whitelist=0123456789: ",
                "description": "OPTIMAL + PSM8 play clock",
            },
            # Attempt 2: OPTIMAL preprocessing + Tesseract PSM 7
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 7 -c tessedit_char_whitelist=0123456789: ",
                "description": "OPTIMAL + PSM7 play clock",
            },
            # Attempt 3: OPTIMAL preprocessing + EasyOCR
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "easyocr",
                "description": "OPTIMAL + EasyOCR play clock",
            },
        ]

        results = []

        logger.debug(f"🔍 Attempting {len(attempts)} play clock OCR methods")

        for attempt in attempts:
            try:
                logger.debug(f"🔍 Trying play clock attempt: {attempt['description']}")

                # Apply preprocessing
                processed = attempt["preprocess"](region)

                # Extract text
                if attempt["config"] == "easyocr":
                    # Use EasyOCR - FIXED: Use self.ocr which is the actual OCR engine
                    if hasattr(self, "ocr") and hasattr(self.ocr, "reader") and self.ocr.reader:
                        logger.debug(f"🔍 Using self.ocr for play clock EasyOCR")
                        ocr_results = self.ocr.reader.readtext(processed)
                        logger.debug(f"🔍 Play clock EasyOCR found {len(ocr_results)} results")
                        for bbox, text, conf in ocr_results:
                            logger.debug(
                                f"🔍 Play clock EasyOCR result: '{text}' (conf: {conf:.3f})"
                            )
                            min_conf = 0.1 if processed.shape[0] > 50 else 0.05
                            if conf > min_conf:
                                results.append((text.strip(), conf, attempt["description"]))
                                logger.debug(
                                    f"✅ Added play clock EasyOCR result: '{text.strip()}' (conf: {conf:.3f})"
                                )
                    else:
                        logger.debug(f"❌ No EasyOCR engine available for play clock")
                else:
                    # Use Tesseract
                    import pytesseract

                    logger.debug(
                        f"🔍 Using Tesseract for play clock with config: {attempt['config']}"
                    )
                    text = pytesseract.image_to_string(processed, config=attempt["config"]).strip()
                    logger.debug(f"🔍 Play clock Tesseract result: '{text}'")
                    if text:
                        # Estimate confidence based on text quality
                        conf = self._estimate_play_clock_confidence(text)
                        results.append((text, conf, attempt["description"]))
                        logger.debug(
                            f"✅ Added play clock Tesseract result: '{text}' (conf: {conf:.3f})"
                        )

                        # Early exit if we get a high-confidence result
                        if conf > 0.8:
                            logger.debug(
                                f"⚡ Early exit with high play clock confidence: {conf:.2f}"
                            )
                            break
                    else:
                        logger.debug(f"❌ Play clock Tesseract returned empty text")

            except Exception as e:
                logger.debug(f"❌ Play clock OCR attempt {attempt['description']} failed: {e}")
                continue

        # Process results and find the best match
        logger.debug(f"🔍 Processing {len(results)} play clock OCR results")
        best_result = None
        best_score = 0

        for text, conf, source in results:
            logger.debug(f"🔍 Processing play clock result: '{text}' from {source}")
            # Apply corrections
            corrected = self._apply_play_clock_corrections(text)
            logger.debug(f"🔍 After play clock corrections: '{corrected}'")

            # Validate the result
            is_valid = self._validate_play_clock(corrected)
            logger.debug(f"🔍 Play clock validation result: {is_valid}")

            if is_valid:
                # Score based on confidence and text quality
                quality_score = self._calculate_play_clock_quality_score(corrected)
                score = conf * quality_score
                logger.debug(
                    f"🔍 Play clock quality score: {quality_score:.3f}, Combined score: {score:.3f}"
                )

                if score > best_score:
                    best_score = score
                    best_result = corrected
                    logger.debug(
                        f"✅ New best play clock from {source}: '{corrected}' (score: {score:.2f})"
                    )
            else:
                logger.debug(f"❌ Play clock validation failed for '{corrected}'")

        if best_result:
            logger.debug(f"🎯 Final robust play clock result: '{best_result}'")
        else:
            logger.debug(f"❌ No valid play clock results found")

        return best_result

    def _apply_play_clock_corrections(self, text: str) -> str:
        """Apply OCR corrections specific to play clock text patterns."""
        import re

        # Convert to uppercase for consistent processing
        corrected = text.upper().strip()

        # Remove common OCR artifacts
        corrected = re.sub(r"[^\d:]", "", corrected)  # Keep only digits and colons

        # Fix common OCR mistakes for clock digits
        corrections = {
            "O": "0",  # O -> 0
            "I": "1",  # I -> 1
            "S": "5",  # S -> 5
            "B": "8",  # B -> 8
            "G": "6",  # G -> 6
        }

        for mistake, correction in corrections.items():
            corrected = corrected.replace(mistake, correction)

        # Play clock is typically just seconds (0-40)
        if ":" in corrected:
            # Remove colon if present (play clock doesn't use MM:SS format)
            corrected = corrected.replace(":", "")

        # Ensure it's a valid number
        try:
            seconds = int(corrected)
            if 0 <= seconds <= 40:
                corrected = str(seconds)
            else:
                corrected = ""  # Invalid range
        except ValueError:
            corrected = ""  # Not a valid number

        return corrected

    def _validate_play_clock(self, text: str) -> bool:
        """Validate play clock text format and values."""
        import re

        # Must be a number between 0 and 40
        if not re.match(r"^\d{1,2}$", text):
            return False

        try:
            seconds = int(text)
            return 0 <= seconds <= 40
        except ValueError:
            return False

    def _estimate_play_clock_confidence(self, text: str) -> float:
        """Estimate confidence for play clock text."""
        if not text:
            return 0.0

        # Base confidence
        confidence = 0.5

        # Check if it's a valid number
        try:
            seconds = int(text.strip())
            if 0 <= seconds <= 40:
                confidence += 0.3
            else:
                confidence -= 0.2
        except ValueError:
            confidence -= 0.3

        # Length check (1-2 digits expected)
        if 1 <= len(text.strip()) <= 2:
            confidence += 0.2
        else:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _calculate_play_clock_quality_score(self, text: str) -> float:
        """Calculate quality score for play clock text."""
        if not text:
            return 0.0

        score = 0.5

        # Valid number check
        try:
            seconds = int(text)
            if 0 <= seconds <= 40:
                score += 0.4
            else:
                score -= 0.3
        except ValueError:
            score -= 0.4

        # Length appropriateness
        if 1 <= len(text) <= 2:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _apply_game_clock_corrections(self, text: str) -> str:
        """Apply OCR corrections specific to game clock text patterns."""
        import re

        # Convert to uppercase for consistent processing
        corrected = text.upper().strip()

        # Remove common OCR artifacts
        corrected = re.sub(r"[^\d:]", "", corrected)  # Keep only digits and colons

        # Fix common OCR mistakes for clock digits
        corrections = {
            "O": "0",  # O -> 0
            "I": "1",  # I -> 1
            "S": "5",  # S -> 5
            "B": "8",  # B -> 8
            "G": "6",  # G -> 6
        }

        for mistake, correction in corrections.items():
            corrected = corrected.replace(mistake, correction)

        # Ensure proper MM:SS format
        if ":" not in corrected and len(corrected) >= 3:
            # Try to insert colon in the right place
            if len(corrected) == 3:  # e.g., "145" -> "1:45"
                corrected = corrected[0] + ":" + corrected[1:]
            elif len(corrected) == 4:  # e.g., "1245" -> "12:45"
                corrected = corrected[:2] + ":" + corrected[2:]

        # Fix multiple colons
        corrected = re.sub(r":+", ":", corrected)

        # Ensure format is MM:SS
        if ":" in corrected:
            parts = corrected.split(":")
            if len(parts) == 2:
                try:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    # Validate ranges
                    if 0 <= minutes <= 15 and 0 <= seconds <= 59:
                        corrected = f"{minutes:02d}:{seconds:02d}"
                except ValueError:
                    pass

        return corrected

    def _validate_game_clock(self, text: str) -> bool:
        """Validate game clock text format and values."""
        import re

        # Must match MM:SS pattern
        if not re.match(r"^\d{1,2}:\d{2}$", text):
            return False

        try:
            parts = text.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1])

            # Game clock validation: 0-15 minutes, 0-59 seconds
            if not (0 <= minutes <= 15 and 0 <= seconds <= 59):
                return False

            return True
        except (ValueError, IndexError):
            return False

    def _estimate_game_clock_confidence(self, text: str) -> float:
        """Estimate confidence for game clock text."""
        import re

        confidence = 0.5

        # Bonus for containing expected patterns
        if re.search(r"\d+:\d+", text):
            confidence += 0.3
        if re.search(r"^\d{1,2}:\d{2}$", text):
            confidence += 0.2

        # Penalty for unexpected characters
        if re.search(r"[^0-9:]", text):
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _calculate_game_clock_quality_score(self, text: str) -> float:
        """Calculate quality score for game clock text."""
        import re

        score = 0.5

        # Perfect format bonus
        if re.match(r"^\d{1,2}:\d{2}$", text):
            score += 0.4

        # Reasonable time values
        try:
            parts = text.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1])

            if 0 <= minutes <= 15 and 0 <= seconds <= 59:
                score += 0.3

            # Typical game clock values
            if 0 <= minutes <= 15:
                score += 0.2

        except (ValueError, IndexError):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _validate_game_clock_temporal(self, new_clock: str) -> tuple[bool, str]:
        """
        Validate game clock against temporal consistency rules.
        Game clocks can only decrease or stay the same (never increase).

        Returns:
            tuple: (is_valid, reason)
        """
        if not self.game_clock_history:
            # First reading, always valid
            return True, "First reading"

        try:
            # Parse new clock time
            new_parts = new_clock.split(":")
            new_minutes = int(new_parts[0])
            new_seconds = int(new_parts[1])
            new_total_seconds = new_minutes * 60 + new_seconds

            # Get the most recent valid clock reading
            last_clock = self.game_clock_history[-1]
            last_parts = last_clock.split(":")
            last_minutes = int(last_parts[0])
            last_seconds = int(last_parts[1])
            last_total_seconds = last_minutes * 60 + last_seconds

            # Game clock rules:
            # 1. Can decrease (normal game flow)
            # 2. Can stay the same (brief pause, timeout)
            # 3. CANNOT increase (unless quarter change, but that's rare)

            if new_total_seconds <= last_total_seconds:
                # Valid: clock decreased or stayed same
                return True, f"Valid decrease: {last_clock} → {new_clock}"
            else:
                # Invalid: clock increased
                time_increase = new_total_seconds - last_total_seconds
                return False, f"Invalid increase: {last_clock} → {new_clock} (+{time_increase}s)"

        except (ValueError, IndexError) as e:
            return False, f"Parse error: {e}"

    def _update_game_clock_history(self, clock: str) -> None:
        """Update game clock history for temporal validation."""
        self.game_clock_history.append(clock)

        # Keep only the last N readings
        if len(self.game_clock_history) > self.max_clock_history:
            self.game_clock_history.pop(0)

        logger.debug(f"⏰ Updated game clock history: {self.game_clock_history}")

    def _extract_yard_line_from_region(
        self, region_data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Extract yard line information from territory region using robust OCR.

        Args:
            region_data: Dictionary containing ROI, bbox, and confidence

        Returns:
            Dictionary with yard line information or None if extraction fails
        """
        try:
            roi = region_data.get("roi")
            if roi is None or roi.size == 0:
                return None

            # Use robust yard line extraction
            yard_line_text = self._extract_yard_line_robust(roi)

            if yard_line_text:
                # Parse the yard line text (e.g., "A35", "H22", "50")
                parsed_result = self._parse_yard_line_text(yard_line_text)

                if parsed_result:
                    logger.info(
                        f"🏈 YARD LINE: Extracted '{yard_line_text}' -> {parsed_result['yard_line']} yard line"
                    )
                    return {
                        "yard_line": parsed_result["yard_line"],
                        "territory_side": parsed_result.get("territory_side"),
                        "raw_text": yard_line_text,
                        "confidence": region_data.get("confidence", 0.5),
                    }

            return None

        except Exception as e:
            logger.error(f"🚨 EXCEPTION in _extract_yard_line_from_region: {e}")
            return None

    def _extract_yard_line_robust(self, region: np.ndarray) -> Optional[str]:
        """
        Extract yard line text using multi-engine OCR with robust preprocessing.

        Args:
            region: Image region containing yard line text

        Returns:
            Extracted yard line text or None if extraction fails
        """
        if region is None or region.size == 0:
            return None

        # OPTIMAL PREPROCESSING: Use 0.939 score parameters for yard line
        ocr_attempts = [
            # Attempt 1: OPTIMAL preprocessing + Tesseract PSM 7
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 7 -c tessedit_char_whitelist=0123456789AH ",
                "description": "OPTIMAL + PSM7",
            },
            # Attempt 2: OPTIMAL preprocessing + Tesseract PSM 8
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 8 -c tessedit_char_whitelist=0123456789AH ",
                "description": "OPTIMAL + PSM8",
            },
            # Attempt 3: OPTIMAL preprocessing + Tesseract PSM 6
            {
                "preprocess": self.ocr.preprocess_image,  # Use optimal 0.939 score preprocessing
                "config": "--psm 6 -c tessedit_char_whitelist=0123456789AH ",
                "description": "OPTIMAL + PSM6",
            },
        ]

        results = []

        # Try Tesseract with different preprocessing
        for attempt in ocr_attempts:
            try:
                processed_region = attempt["preprocess"](region)

                # Tesseract OCR
                import pytesseract

                tesseract_text = pytesseract.image_to_string(
                    processed_region, config=attempt["config"]
                ).strip()

                if tesseract_text:
                    confidence = self._estimate_yard_line_confidence(tesseract_text)
                    results.append(
                        (tesseract_text, confidence, f"Tesseract-{attempt['description']}")
                    )

            except Exception as e:
                logger.debug(f"Tesseract {attempt['description']} failed: {e}")
                continue

        # Try EasyOCR with OPTIMAL preprocessing as fallback
        try:
            if hasattr(self, "ocr") and hasattr(self.ocr, "reader") and self.ocr.reader:
                # Apply optimal preprocessing FIRST, then use EasyOCR
                optimally_preprocessed_region = self.ocr.preprocess_image(region)
                easyocr_results = self.ocr.reader.readtext(optimally_preprocessed_region, detail=1)
                for bbox, text, conf in easyocr_results:
                    if text.strip():
                        results.append((text.strip(), conf, "OPTIMAL + EasyOCR"))
        except Exception as e:
            logger.debug(f"EasyOCR yard line extraction failed: {e}")

        # Process results and find the best match
        logger.debug(f"🔍 Processing {len(results)} yard line OCR results")
        best_result = None
        best_score = 0

        for text, conf, source in results:
            logger.debug(f"🔍 Processing yard line result: '{text}' from {source}")
            # Apply corrections
            corrected = self._apply_yard_line_corrections(text)
            logger.debug(f"🔍 After yard line corrections: '{corrected}'")

            # Validate the result
            is_valid = self._validate_yard_line(corrected)
            logger.debug(f"🔍 Yard line validation result: {is_valid}")

            if is_valid:
                # Score based on confidence and text quality
                quality_score = self._calculate_yard_line_quality_score(corrected)
                score = conf * quality_score

                logger.debug(f"🔍 Yard line quality score: {quality_score}, final score: {score}")

                if score > best_score:
                    best_result = corrected
                    best_score = score
                    logger.debug(
                        f"🔍 New best yard line result: '{best_result}' (score: {best_score})"
                    )

        if best_result:
            logger.info(f"🏈 YARD LINE OCR SUCCESS: '{best_result}' (confidence: {best_score:.2f})")
            return best_result
        else:
            logger.debug("🔍 No valid yard line OCR results found")
            return None

    def _parse_yard_line_text(self, text: str) -> Optional[dict[str, Any]]:
        """Parse yard line text into structured data."""
        import re

        # Apply corrections first
        corrected_text = self._apply_yard_line_corrections(text)

        # Patterns for yard line detection
        patterns = [
            r"([AH])(\d+)",  # A35, H22 (Away/Home + number)
            r"(\d+)",  # 50 (midfield)
        ]

        for pattern in patterns:
            match = re.search(pattern, corrected_text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:
                        # A35, H22 format
                        territory_side = match.group(1).upper()
                        yard_number = int(match.group(2))

                        # Validate yard number (0-50)
                        if 0 <= yard_number <= 50:
                            return {
                                "yard_line": yard_number,
                                "territory_side": territory_side,
                                "raw_text": text,
                                "corrected_text": corrected_text,
                            }
                    else:
                        # Just number (50 yard line)
                        yard_number = int(match.group(1))

                        # Validate yard number (0-50)
                        if 0 <= yard_number <= 50:
                            return {
                                "yard_line": yard_number,
                                "territory_side": None,  # Midfield
                                "raw_text": text,
                                "corrected_text": corrected_text,
                            }

                except ValueError:
                    continue

        return None

    def _apply_yard_line_corrections(self, text: str) -> str:
        """Apply OCR corrections specific to yard line text patterns."""
        import re

        # Convert to uppercase for consistent processing
        corrected = text.upper().strip()

        # Common OCR mistakes for yard line - CONSERVATIVE corrections only
        corrections = {
            # Letter corrections for territory indicators only
            "B": "8",  # B often misread as 8
            "S": "5",  # S often misread as 5
            "I": "1",  # I often misread as 1
            "Z": "2",  # Z often misread as 2
            "G": "6",  # G often misread as 6
            # Territory side corrections (only when followed by numbers)
            "4": "A",  # 4 sometimes misread as A
            "8": "A",  # 8 sometimes misread as A (only in territory context)
            "R": "A",  # R sometimes misread as A
            "N": "H",  # N sometimes misread as H
            "M": "H",  # M sometimes misread as H
        }

        # Apply conservative corrections only if pattern suggests yard line
        if re.search(r"[AH4R8NM]\d+", corrected) or re.search(r"\d+", corrected):
            for mistake, correction in corrections.items():
                # Only apply territory corrections if they're at the start and followed by digits
                if mistake in ["4", "8", "R", "N", "M"]:
                    if (
                        corrected.startswith(mistake)
                        and len(corrected) > 1
                        and corrected[1:].isdigit()
                    ):
                        corrected = correction + corrected[1:]
                else:
                    corrected = corrected.replace(mistake, correction)

        # Remove extra spaces and non-alphanumeric characters (but keep A, H, and digits)
        corrected = re.sub(r"[^AH0-9]", "", corrected)

        return corrected

    def _validate_yard_line(self, text: str) -> bool:
        """Validate yard line text format."""
        import re

        if not text or len(text.strip()) == 0:
            return False

        # Valid patterns for yard line
        valid_patterns = [
            r"^[AH]\d+$",  # A35, H22
            r"^\d+$",  # 50 (midfield)
        ]

        for pattern in valid_patterns:
            if re.match(pattern, text):
                # Extract number and validate range
                numbers = re.findall(r"\d+", text)
                if numbers:
                    yard_num = int(numbers[0])
                    return 0 <= yard_num <= 50

        return False

    def _estimate_yard_line_confidence(self, text: str) -> float:
        """Estimate confidence of yard line OCR result."""
        if not text:
            return 0.0

        # Base confidence for having text
        confidence = 0.3

        # Length-based confidence
        if 2 <= len(text) <= 4:
            confidence += 0.3
        elif len(text) == 1:
            confidence += 0.1

        # Pattern-based confidence
        import re

        if re.match(r"^[AH]\d+$", text):  # A35, H22 format
            confidence += 0.4
        elif re.match(r"^\d+$", text):  # 50 format
            confidence += 0.3

        # Penalty for unusual characters
        if re.search(r"[^AH0-9]", text):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _calculate_yard_line_quality_score(self, text: str) -> float:
        """Calculate quality score for yard line text based on format and content."""
        if not text:
            return 0.0

        score = 0.0

        # Base score for having text
        score += 0.3

        # Score for valid format patterns
        import re

        if re.match(r"^[AH]\d+$", text):  # A35, H22 format
            score += 0.4
        elif re.match(r"^\d+$", text):  # 50 format (midfield)
            score += 0.3

        # Score for reasonable yard line numbers
        numbers = re.findall(r"\d+", text)
        if numbers:
            yard_num = int(numbers[0])
            if 0 <= yard_num <= 50:
                score += 0.3
                # Bonus for common yard lines
                if yard_num in [20, 25, 30, 35, 40, 45, 50]:
                    score += 0.1

        # Penalty for unusual characters
        if re.search(r"[^AH0-9]", text):
            score -= 0.2

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

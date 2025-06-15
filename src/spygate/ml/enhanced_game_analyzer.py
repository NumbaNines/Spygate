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
from .enhanced_ocr import EnhancedOCR
from .situational_predictor import GameSituation, SituationalPredictor
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
    visualization_layers: dict[str, np.ndarray] = None


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

        # Initialize MAIN OCR engine with optimal preprocessing parameters (0.939 score from 20K sweep)
        self.ocr = EnhancedOCR(hardware=self.hardware.detect_tier())  # PRIMARY OCR system

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

        # Initialize situational predictor for hybrid OCR+logic approach
        self.situational_predictor = SituationalPredictor()
        logger.info("🎯 Situational predictor initialized for hybrid OCR+logic validation")

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

    def analyze_frame(
        self, frame: np.ndarray, current_time: float = None, frame_number: int = None
    ) -> GameState:
        """Analyze a single frame and return game state with visualization layers."""
        # Initialize frame cache if not exists (CRITICAL FIX: Don't reset every call!)
        if not hasattr(self, "frame_cache"):
            self.frame_cache = {}
            self.frame_cache_ttl = 30  # seconds
            self.max_frame_cache_size = 50  # frames

        # Create copy for visualization
        vis_frame = frame.copy()

        # Use current time for temporal optimization
        # Note: For burst sampling, current_time=None should bypass temporal manager
        # Don't automatically generate a timestamp in this case
        if current_time is None:
            # Only generate timestamp if we don't explicitly want to bypass temporal manager
            # For burst sampling, we want to keep current_time=None to disable temporal optimization
            pass  # Keep current_time=None for burst sampling

        # Run YOLOv8 detection
        detections = self.model.detect(frame)

        # Initialize visualization layers
        layers = {
            "original_frame": frame.copy(),  # Add original frame for OCR extraction
            "hud_detection": frame.copy(),
            "triangle_detection": frame.copy(),
            "ocr_results": frame.copy(),
        }

        # Process triangle orientations
        possession_results = []
        territory_results = []

        for detection in detections:
            # Handle both old and new detection formats
            if hasattr(detection, "cls"):
                # Old format (direct detection object)
                class_id = detection.cls
                bbox = detection.xyxy[0] if hasattr(detection, "xyxy") else detection.bbox
            else:
                # New format (dictionary)
                class_name = detection.get("class", "")
                class_id = None
                for name, id in self.class_map.items():
                    if name == class_name:
                        class_id = id
                        break
                bbox = detection.get("bbox", [])

            if class_id == self.class_map.get("possession_triangle_area"):
                # Extract ROI from YOLO detection
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    # Use proven template matching (97.6% accuracy) to find possession triangles
                    triangles = self.triangle_detector.detect_triangles_in_roi(roi, "possession")
                    if triangles:
                        # Select the best single triangle
                        best_triangle = self.triangle_detector.select_best_single_triangles(
                            triangles, "possession"
                        )
                        if best_triangle:
                            possession_results.append(best_triangle)
                            self.last_possession_direction = best_triangle["direction"]

            elif class_id == self.class_map.get("territory_triangle_area"):
                # Extract ROI from YOLO detection
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    # Use proven template matching (97.6% accuracy) to find territory triangles
                    triangles = self.triangle_detector.detect_triangles_in_roi(roi, "territory")
                    if triangles:
                        # Select the best single triangle
                        best_triangle = self.triangle_detector.select_best_single_triangles(
                            triangles, "territory"
                        )
                        if best_triangle:
                            territory_results.append(best_triangle)
                            self.last_territory_direction = best_triangle["direction"]

        # Update game state with triangle information and detect flips
        if possession_results:
            # Use highest confidence result
            best_possession = max(possession_results, key=lambda x: x["confidence"])
            new_possession_direction = best_possession["direction"]

            # Detect possession change (triangle flip)
            if (
                self.last_possession_direction
                and self.last_possession_direction != new_possession_direction
            ):
                self._handle_possession_change(
                    self.last_possession_direction, new_possession_direction
                )

            self.game_state["possession"] = {
                "direction": new_possession_direction,
                "confidence": best_possession["confidence"],
                "team_with_ball": self._get_team_with_ball(new_possession_direction),
            }
            self.last_possession_direction = new_possession_direction
        elif self.last_possession_direction:
            # Use last known direction with reduced confidence
            self.game_state["possession"] = {
                "direction": self.last_possession_direction,
                "confidence": self.direction_confidence_threshold / 2,
                "team_with_ball": self._get_team_with_ball(self.last_possession_direction),
            }

        if territory_results:
            # Use highest confidence result
            best_territory = max(territory_results, key=lambda x: x["confidence"])
            new_territory_direction = best_territory["direction"]

            # Detect territory change (triangle flip)
            if (
                self.last_territory_direction
                and self.last_territory_direction != new_territory_direction
            ):
                self._handle_territory_change(
                    self.last_territory_direction, new_territory_direction
                )

            self.game_state["territory"] = {
                "direction": new_territory_direction,
                "confidence": best_territory["confidence"],
                "field_context": self._get_field_context(new_territory_direction),
            }
            self.last_territory_direction = new_territory_direction
        elif self.last_territory_direction:
            # Use last known direction with reduced confidence
            self.game_state["territory"] = {
                "direction": self.last_territory_direction,
                "confidence": self.direction_confidence_threshold / 2,
                "field_context": self._get_field_context(self.last_territory_direction),
            }

        # Process NEW 8-class model detections for enhanced HUD analysis
        down_distance_regions = []
        game_clock_regions = []
        play_clock_regions = []

        for detection in detections:
            # Handle both old and new detection formats for 8-class processing
            if hasattr(detection, "cls"):
                # Old format (direct detection object)
                class_id = detection.cls
                bbox = detection.xyxy[0] if hasattr(detection, "xyxy") else detection.bbox
                confidence = float(detection.conf) if hasattr(detection, "conf") else 0.5
            else:
                # New format (dictionary)
                class_name = detection.get("class", "")
                class_id = None
                for name, id in self.class_map.items():
                    if name == class_name:
                        class_id = id
                        break
                bbox = detection.get("bbox", [])
                confidence = float(detection.get("confidence", 0.5))

            # Process specific down_distance_area detections
            if class_id == self.class_map.get("down_distance_area"):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    down_distance_regions.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "roi": frame[int(y1) : int(y2), int(x1) : int(x2)],
                        }
                    )

            # Process game_clock_area detections
            elif class_id == self.class_map.get("game_clock_area"):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    game_clock_regions.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "roi": frame[int(y1) : int(y2), int(x1) : int(x2)],
                        }
                    )

            # Process play_clock_area detections
            elif class_id == self.class_map.get("play_clock_area"):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    play_clock_regions.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "roi": frame[int(y1) : int(y2), int(x1) : int(x2)],
                        }
                    )

        # Process detections and update visualization layers
        logger.debug(f"🎯 Processing {len(detections)} detections through unified OCR pipeline")
        game_state = self._process_detections(detections, layers, current_time)
        game_state.visualization_layers = layers

        # INTELLIGENT STATE PERSISTENCE: Replace naive persistence with smarter logic
        if not hasattr(self, "last_game_state"):
            self.last_game_state = GameState()
        if not hasattr(self, "persistence_counters"):
            self.persistence_counters = {"down": 0, "distance": 0, "quarter": 0, "time": 0}
        if not hasattr(self, "last_quarter"):
            self.last_quarter = None

        # Reset persistence when quarter changes (new quarter = potential reset)
        if game_state.quarter is not None and self.last_quarter != game_state.quarter:
            print(
                f"🔄 QUARTER CHANGE: {self.last_quarter} → {game_state.quarter}, resetting down/distance persistence"
            )
            self.persistence_counters["down"] = 0
            self.persistence_counters["distance"] = 0
            self.last_quarter = game_state.quarter

        # Smart persistence with frame limits (max 10 frames = ~0.33 seconds at 30fps)
        MAX_PERSISTENCE_FRAMES = 10

        # Down persistence - FIXED: Only apply persistence when OCR fails (down is None)
        # CRITICAL FIX: Don't overwrite fresh OCR results with stale persistence data
        if game_state.down is not None:
            # Fresh OCR result - reset persistence counter and update last known value
            self.persistence_counters["down"] = 0
            print(f"✅ FRESH OCR: down={game_state.down} (resetting persistence)")
        elif self.last_game_state.down is not None:
            # OCR failed - use persistence only if within frame limit
            if self.persistence_counters["down"] < MAX_PERSISTENCE_FRAMES:
                game_state.down = self.last_game_state.down
                self.persistence_counters["down"] += 1
                print(
                    f"🔄 STATE PERSISTENCE: Using down={game_state.down} from previous frame ({self.persistence_counters['down']}/{MAX_PERSISTENCE_FRAMES})"
                )
            else:
                print(
                    f"⏰ PERSISTENCE EXPIRED: Allowing down to become None after {MAX_PERSISTENCE_FRAMES} frames"
                )
                self.persistence_counters["down"] = 0

        # Distance persistence - FIXED: Only apply persistence when OCR fails (distance is None)
        # CRITICAL FIX: Don't overwrite fresh OCR results with stale persistence data
        if game_state.distance is not None:
            # Fresh OCR result - reset persistence counter and update last known value
            self.persistence_counters["distance"] = 0
            print(f"✅ FRESH OCR: distance={game_state.distance} (resetting persistence)")
        elif self.last_game_state.distance is not None:
            # OCR failed - use persistence only if within frame limit
            if self.persistence_counters["distance"] < MAX_PERSISTENCE_FRAMES:
                game_state.distance = self.last_game_state.distance
                self.persistence_counters["distance"] += 1
                print(
                    f"🔄 STATE PERSISTENCE: Using distance={game_state.distance} from previous frame ({self.persistence_counters['distance']}/{MAX_PERSISTENCE_FRAMES})"
                )
            else:
                print(
                    f"⏰ PERSISTENCE EXPIRED: Allowing distance to become None after {MAX_PERSISTENCE_FRAMES} frames"
                )
                self.persistence_counters["distance"] = 0

        # Quarter persistence - FIXED: Only apply persistence when OCR fails (quarter is None)
        # CRITICAL FIX: Don't overwrite fresh OCR results with stale persistence data
        if game_state.quarter is not None:
            # Fresh OCR result - reset persistence counter and update last known value
            self.persistence_counters["quarter"] = 0
            print(f"✅ FRESH OCR: quarter={game_state.quarter} (resetting persistence)")
        elif self.last_game_state.quarter is not None:
            # OCR failed - use persistence only if within frame limit (longer for quarter)
            if self.persistence_counters["quarter"] < 60:  # 2 seconds
                game_state.quarter = self.last_game_state.quarter
                self.persistence_counters["quarter"] += 1
                print(
                    f"🔄 STATE PERSISTENCE: Using quarter={game_state.quarter} from previous frame ({self.persistence_counters['quarter']}/60)"
                )
            else:
                print(f"⏰ PERSISTENCE EXPIRED: Allowing quarter to become None after 60 frames")
                self.persistence_counters["quarter"] = 0

        # Time persistence - FIXED: Only apply persistence when OCR fails (time is None)
        # CRITICAL FIX: Don't overwrite fresh OCR results with stale persistence data
        if game_state.time is not None:
            # Fresh OCR result - reset persistence counter and update last known value
            self.persistence_counters["time"] = 0
            print(f"✅ FRESH OCR: time={game_state.time} (resetting persistence)")
        elif self.last_game_state.time is not None:
            # OCR failed - use persistence only if within frame limit (moderate duration)
            if self.persistence_counters["time"] < 30:  # 1 second
                game_state.time = self.last_game_state.time
                self.persistence_counters["time"] += 1
                print(
                    f"🔄 STATE PERSISTENCE: Using time={game_state.time} from previous frame ({self.persistence_counters['time']}/30)"
                )
            else:
                print(f"⏰ PERSISTENCE EXPIRED: Allowing time to become None after 30 frames")
                self.persistence_counters["time"] = 0

        # Update last_game_state for next frame (only update non-None values)
        if game_state.down is not None:
            self.last_game_state.down = game_state.down
        if game_state.distance is not None:
            self.last_game_state.distance = game_state.distance
        if game_state.quarter is not None:
            self.last_game_state.quarter = game_state.quarter
        if game_state.time is not None:
            self.last_game_state.time = game_state.time

        # Debug: Log what _process_detections extracted
        logger.debug(
            f"🔍 _process_detections results: down={game_state.down}, distance={game_state.distance}, yard_line={game_state.yard_line}, time={game_state.time}"
        )

        # 🔥 CRITICAL FIX: Enhanced OCR data merging from both extraction paths
        # Ensure all OCR results from both systems are properly integrated
        if hasattr(self, "game_state") and self.game_state:
            logger.debug(f"🔄 Merging OCR results: self.game_state = {self.game_state}")
            logger.debug(
                f"🔄 Current GameState: down={game_state.down}, distance={game_state.distance}, yard_line={game_state.yard_line}"
            )

            # Prioritize _process_detections results, but fill gaps with pre-processed results
            # This ensures we get the BEST results from either system

            # Down/Distance: Use _process_detections if available, otherwise use pre-processed
            if game_state.down is None and "down" in self.game_state:
                game_state.down = self.game_state["down"]
                logger.debug(f"📝 Using pre-processed down: {game_state.down}")
            if game_state.distance is None and "distance" in self.game_state:
                game_state.distance = self.game_state["distance"]
                logger.debug(f"📝 Using pre-processed distance: {game_state.distance}")

            # Yard Line: Merge results intelligently
            if game_state.yard_line is None and "yard_line" in self.game_state:
                game_state.yard_line = self.game_state["yard_line"]
                logger.debug(f"📝 Using pre-processed yard_line: {game_state.yard_line}")

            # Time/Quarter: Merge clock information
            if game_state.time is None and "time" in self.game_state:
                game_state.time = self.game_state["time"]
                logger.debug(f"📝 Using pre-processed time: {game_state.time}")
            if game_state.quarter is None and "quarter" in self.game_state:
                game_state.quarter = self.game_state["quarter"]
                logger.debug(f"📝 Using pre-processed quarter: {game_state.quarter}")

            # Scores: Merge score information
            if game_state.score_away is None and "away_score" in self.game_state:
                game_state.score_away = self.game_state["away_score"]
                logger.debug(f"📝 Using pre-processed away_score: {game_state.score_away}")
            if game_state.score_home is None and "home_score" in self.game_state:
                game_state.score_home = self.game_state["home_score"]
                logger.debug(f"📝 Using pre-processed home_score: {game_state.score_home}")

            # Team names
            if game_state.away_team is None and "away_team" in self.game_state:
                game_state.away_team = self.game_state["away_team"]
                logger.debug(f"📝 Using pre-processed away_team: {game_state.away_team}")
            if game_state.home_team is None and "home_team" in self.game_state:
                game_state.home_team = self.game_state["home_team"]
                logger.debug(f"📝 Using pre-processed home_team: {game_state.home_team}")

            logger.debug(
                f"✅ Final merged GameState: down={game_state.down}, distance={game_state.distance}, yard_line={game_state.yard_line}, time={game_state.time}"
            )

            # Clear the temporary game_state dictionary to prevent stale data
            self.game_state.clear()

        # Detect burst sampling mode and handle accordingly
        is_burst_mode = current_time is None
        if is_burst_mode:
            logger.debug(
                f"🎯 BURST MODE: Adding frame {frame_number or 'unknown'} to consensus system"
            )

            # Create frame result for burst consensus
            frame_result = {
                "timestamp": frame_number / 30.0 if frame_number else 0.0,  # Estimate timestamp
                "down": game_state.down,
                "distance": game_state.distance,
                "yard_line": game_state.yard_line,
                "game_clock": game_state.time,
                "play_clock": None,  # Add if available
                "possession_team": game_state.possession_team,
                "territory": game_state.territory,
                "confidence": self._calculate_frame_confidence(game_state),
                "method": "yolo_ocr_hybrid",
            }

            # Add to burst consensus system
            self.add_burst_result(frame_result, frame_number)

        # Cache the complete analysis result
        if self.cache_enabled and self.advanced_cache:
            try:
                serialized_result = self._serialize_game_state(game_state)
                self.advanced_cache.set_frame_analysis(frame, serialized_result, "v2_optimized")
                logger.debug("💾 Cached frame analysis result")
            except Exception as e:
                logger.debug(f"Cache storage failed: {e}")

        # Save debug data
        self._save_debug_frame_data(frame_number, frame, game_state, detections, {})

        return game_state

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

        # Update overall confidence
        game_state.confidence = total_confidence / num_detections if num_detections > 0 else 0.0

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
        game_state = super().extract_game_state(frame)

        # Get current field zone if yard line available
        if game_state.get("yard_line") and game_state.get("territory"):
            current_territory, current_zone = self._get_field_zone(
                game_state["yard_line"], game_state["territory"]
            )

            # Track zone change
            current_zone_full = f"{current_territory}_{current_zone}"
            if self.last_zone != current_zone_full:
                self.tracking_metrics["zone_changes"].append(
                    {
                        "timestamp": time.time(),
                        "from_zone": self.last_zone,
                        "to_zone": current_zone_full,
                        "yard_line": game_state["yard_line"],
                    }
                )
                self.last_zone = current_zone_full

            # Update zone statistics
            self._update_zone_stats(game_state)

            # Add zone info to game state
            game_state["field_zone"] = {"territory": current_territory, "zone_name": current_zone}

        # Track formation matches internally
        if game_state.get("formation"):
            formation_matched = self._detect_formation_match(game_state["formation"])
            if formation_matched:
                self.tracking_metrics["formation_sequences"].append(
                    {
                        "timestamp": time.time(),
                        "formation": game_state["formation"],
                        "previous_formation": self.last_formation,
                        "field_zone": self.last_zone,  # Include zone context
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
                    # HYBRID APPROACH: Validate OCR with situational logic
                    ocr_down = parsed_result.get("down")
                    ocr_distance = parsed_result.get("distance")
                    ocr_confidence = parsed_result.get("confidence", 0.0)

                    if ocr_down and ocr_distance:
                        # Create current game situation for logic validation
                        current_situation = GameSituation(
                            down=ocr_down,
                            distance=ocr_distance,
                            yard_line=self.game_state.get("yard_line"),
                            territory=self.game_state.get("territory", {}).get("field_context"),
                            possession_team=self.game_state.get("possession", {}).get(
                                "team_with_ball"
                            ),
                            quarter=self.game_state.get("quarter"),
                            time_remaining=self.game_state.get("time"),
                        )

                        # Validate OCR result with game logic
                        validation_result = self.situational_predictor.validate_ocr_with_logic(
                            ocr_down, ocr_distance, ocr_confidence, current_situation
                        )

                        # Use hybrid result
                        if validation_result["correction_applied"]:
                            logger.info(
                                f"🎯 HYBRID: Logic corrected OCR {ocr_down}&{ocr_distance} → {validation_result['recommended_down']}&{validation_result['recommended_distance']} ({validation_result['reasoning']})"
                            )
                            parsed_result["down"] = validation_result["recommended_down"]
                            parsed_result["distance"] = validation_result["recommended_distance"]
                            parsed_result["confidence"] = validation_result["final_confidence"]
                            parsed_result["hybrid_correction"] = True
                            parsed_result["original_ocr"] = validation_result["original_ocr"]
                            parsed_result["logic_reasoning"] = validation_result["reasoning"]
                        else:
                            # OCR was validated by logic
                            parsed_result["confidence"] = validation_result["final_confidence"]
                            parsed_result["hybrid_validation"] = True
                            parsed_result["logic_reasoning"] = validation_result["reasoning"]

                        # Update situational predictor with current state
                        self.situational_predictor.update_game_state(current_situation)

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

            # LOGIC-ONLY FALLBACK: When OCR completely fails, try pure game logic
            if hasattr(self, "situational_predictor"):
                try:
                    # Create current game situation from available context
                    current_situation = GameSituation(
                        down=None,  # OCR failed
                        distance=None,  # OCR failed
                        yard_line=self.game_state.get("yard_line"),
                        territory=self.game_state.get("territory", {}).get("field_context"),
                        possession_team=self.game_state.get("possession", {}).get("team_with_ball"),
                        quarter=self.game_state.get("quarter"),
                        time_remaining=self.game_state.get("time"),
                    )

                    # Try pure logic prediction
                    logic_prediction = self.situational_predictor._predict_from_game_logic(
                        current_situation
                    )

                    if (
                        logic_prediction.predicted_down
                        and logic_prediction.predicted_distance
                        and logic_prediction.confidence > 0.4
                    ):  # Minimum confidence threshold

                        # Create result from logic prediction
                        logic_result = {
                            "down": logic_prediction.predicted_down,
                            "distance": logic_prediction.predicted_distance,
                            "confidence": logic_prediction.confidence,
                            "method": "logic_only_fallback",
                            "source": "8class_down_distance_area",
                            "region_confidence": confidence,
                            "region_bbox": region_data["bbox"],
                            "logic_only": True,
                            "logic_reasoning": logic_prediction.reasoning,
                            "ocr_failed": True,
                        }

                        # Update situational predictor with current state
                        self.situational_predictor.update_game_state(current_situation)

                        logger.info(
                            f"🧠 LOGIC-ONLY: OCR failed, using pure logic → {logic_prediction.predicted_down}&{logic_prediction.predicted_distance} ({logic_prediction.reasoning})"
                        )

                        # FIXED: Unified result handling - add to appropriate confidence system
                        if not is_burst_mode and hasattr(self, "temporal_manager"):
                            # Normal mode: Add to temporal manager for time-based confidence voting
                            extraction_result = ExtractionResult(
                                value=logic_result,
                                confidence=logic_result["confidence"],
                                timestamp=current_time,
                                raw_text="OCR_FAILED",
                                method="logic_only",
                            )
                            self.temporal_manager.add_extraction_result(
                                "down_distance", extraction_result
                            )
                            logger.debug(
                                f"⏰ TEMPORAL: Added logic-only down/distance result to temporal manager"
                            )
                        else:
                            # Burst mode: Results will be handled by burst consensus system in analyze_frame()
                            logger.debug(
                                f"🎯 BURST: Logic-only down/distance result will be handled by burst consensus"
                            )

                        return logic_result
                    else:
                        if current_time is None:  # Burst sampling debug
                            print(
                                f"🧠 LOGIC-ONLY: Insufficient confidence ({logic_prediction.confidence:.3f}) or missing prediction"
                            )
                            print(
                                f"🧠 Available context: yard_line={current_situation.yard_line}, territory={current_situation.territory}, possession={current_situation.possession_team}"
                            )

                except Exception as e:
                    logger.error(f"Logic-only fallback failed: {e}")
                    if current_time is None:  # Burst sampling debug
                        logger.debug(f"🚨 LOGIC-ONLY EXCEPTION: {e}")

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
            if hasattr(self, "game_state"):
                try:
                    quarter = self.game_state.get("quarter")
                    if quarter and 1 <= quarter <= 4:
                        # Estimate typical game clock based on quarter
                        estimated_minutes = 15 - ((quarter - 1) * 3)  # Rough estimate
                        if estimated_minutes > 0:
                            logic_result = {
                                "minutes": estimated_minutes,
                                "seconds": 0,
                                "time_string": f"{estimated_minutes:02d}:00",
                                "confidence": 0.3,  # Low confidence for logic estimate
                                "method": "logic_only_fallback",
                                "source": "8class_game_clock_area",
                                "region_confidence": confidence,
                                "region_bbox": region_data["bbox"],
                                "logic_only": True,
                                "logic_reasoning": f"Estimated from quarter {quarter}",
                            }

                            if current_time is None:  # Burst sampling mode
                                print(
                                    f"🧠 LOGIC-ONLY: Game clock OCR failed, using quarter-based estimate → {logic_result['time_string']} (Quarter {quarter})"
                                )

                            return logic_result
                except Exception as e:
                    logger.debug(f"Game clock logic fallback failed: {e}")

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

            # LOGIC-ONLY FALLBACK: When OCR completely fails, use game context
            # Play clock can be estimated based on game flow and typical patterns
            if hasattr(self, "game_state"):
                try:
                    # Estimate play clock based on game situation
                    # Typical play clock is 40 seconds, but varies by situation
                    estimated_seconds = 25  # Default mid-range estimate

                    # Adjust based on game context if available
                    down = self.game_state.get("down")
                    if down:
                        if down == 1:
                            estimated_seconds = 30  # More time on 1st down
                        elif down >= 3:
                            estimated_seconds = 20  # Less time on 3rd/4th down

                    logic_result = {
                        "seconds": estimated_seconds,
                        "confidence": 0.25,  # Low confidence for logic estimate
                        "method": "logic_only_fallback",
                        "source": "8class_play_clock_area",
                        "region_confidence": confidence,
                        "region_bbox": region_data["bbox"],
                        "logic_only": True,
                        "logic_reasoning": f"Estimated {estimated_seconds}s based on down {down if down else 'unknown'}",
                    }

                    if current_time is None:  # Burst sampling mode
                        print(
                            f"🧠 LOGIC-ONLY: Play clock OCR failed, using situation-based estimate → {logic_result['seconds']}s (Down {down if down else 'unknown'})"
                        )

                    return logic_result
                except Exception as e:
                    logger.debug(f"Play clock logic fallback failed: {e}")

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

        confidence = 0.5  # Base confidence

        # Boost confidence for valid patterns
        import re

        if re.match(r"^[AH]\d+$", text):
            confidence += 0.3  # Strong pattern match
        elif re.match(r"^\d+$", text):
            confidence += 0.2  # Midfield pattern

        # Boost for reasonable yard numbers
        numbers = re.findall(r"\d+", text)
        if numbers:
            yard_num = int(numbers[0])
            if 0 <= yard_num <= 50:
                confidence += 0.2

        return min(1.0, confidence)

    def _calculate_yard_line_quality_score(self, text: str) -> float:
        """Calculate quality score for yard line text."""
        if not text:
            return 0.0

        score = 0.5  # Base score

        # Pattern quality
        import re

        if re.match(r"^[AH]\d+$", text):
            score += 0.3  # Perfect pattern
        elif re.match(r"^\d+$", text):
            score += 0.2  # Midfield pattern

        # Length appropriateness (2-3 characters)
        if 2 <= len(text) <= 3:
            score += 0.2

        return min(1.0, score)

    def add_burst_result(self, frame_result: dict[str, Any], frame_number: int = None) -> None:
        """
        Add a frame analysis result to the burst consensus system.

        Args:
            frame_result: Analysis result from a single frame
            frame_number: Optional frame number for tracking
        """
        burst_entry = {
            "frame_number": frame_number,
            "timestamp": frame_result.get("timestamp"),
            "down": frame_result.get("down"),
            "distance": frame_result.get("distance"),
            "yard_line": frame_result.get("yard_line"),
            "game_clock": frame_result.get("game_clock"),
            "play_clock": frame_result.get("play_clock"),
            "possession_team": frame_result.get("possession_team"),
            "territory": frame_result.get("territory"),
            "confidence": frame_result.get("confidence", 0.0),
            "method": frame_result.get("method", "unknown"),
        }

        self.burst_results.append(burst_entry)

        # Keep only the most recent frames
        if len(self.burst_results) > self.max_burst_frames:
            self.burst_results.pop(0)

        logger.debug(f"🎯 Added burst result: {burst_entry}")

    def get_burst_consensus(self) -> dict[str, Any]:
        """
        Analyze all burst results and return consensus decision.

        Returns:
            Dictionary with consensus results and confidence scores
        """
        if not self.burst_results:
            return {"error": "No burst results available"}

        logger.info(f"🎯 BURST CONSENSUS: Analyzing {len(self.burst_results)} frame results")

        # Separate analysis for each field
        consensus = {
            "down_distance": self._get_down_distance_consensus(),
            "game_clock": self._get_game_clock_consensus(),
            "play_clock": self._get_play_clock_consensus(),
            "yard_line": self._get_yard_line_consensus(),
            "possession": self._get_possession_consensus(),
            "territory": self._get_territory_consensus(),
            "summary": {
                "total_frames": len(self.burst_results),
                "consensus_method": "confidence_weighted_voting",
            },
        }

        logger.info(f"🎯 BURST CONSENSUS COMPLETE: {consensus['summary']}")
        return consensus

    def _get_down_distance_consensus(self) -> dict[str, Any]:
        """Get consensus for down and distance from burst results."""
        down_votes = {}
        distance_votes = {}

        for result in self.burst_results:
            down = result.get("down")
            distance = result.get("distance")
            confidence = result.get("confidence", 0.0)

            if down is not None:
                key = f"{down}"
                if key not in down_votes:
                    down_votes[key] = {"votes": 0, "total_confidence": 0.0, "frames": []}
                down_votes[key]["votes"] += 1
                down_votes[key]["total_confidence"] += confidence
                down_votes[key]["frames"].append(result.get("frame_number", "unknown"))

            if distance is not None:
                key = f"{distance}"
                if key not in distance_votes:
                    distance_votes[key] = {"votes": 0, "total_confidence": 0.0, "frames": []}
                distance_votes[key]["votes"] += 1
                distance_votes[key]["total_confidence"] += confidence
                distance_votes[key]["frames"].append(result.get("frame_number", "unknown"))

        # Find best down
        best_down = None
        best_down_score = 0
        if down_votes:
            for down_str, data in down_votes.items():
                # Score = votes * average_confidence
                avg_confidence = data["total_confidence"] / data["votes"]
                score = data["votes"] * avg_confidence
                if score > best_down_score:
                    best_down_score = score
                    best_down = int(down_str)

        # Find best distance
        best_distance = None
        best_distance_score = 0
        if distance_votes:
            for distance_str, data in distance_votes.items():
                avg_confidence = data["total_confidence"] / data["votes"]
                score = data["votes"] * avg_confidence
                if score > best_distance_score:
                    best_distance_score = score
                    best_distance = int(distance_str)

        # Temporal validation check
        temporal_validation = None
        if best_down and best_distance:
            # Check if this progression makes sense
            temporal_validation = self._validate_down_distance_progression(best_down, best_distance)

        result = {
            "down": best_down,
            "distance": best_distance,
            "down_confidence": best_down_score / len(self.burst_results) if best_down else 0.0,
            "distance_confidence": (
                best_distance_score / len(self.burst_results) if best_distance else 0.0
            ),
            "down_votes": down_votes,
            "distance_votes": distance_votes,
            "temporal_validation": temporal_validation,
        }

        logger.info(
            f"🎯 DOWN/DISTANCE CONSENSUS: {best_down} & {best_distance} (conf: {result['down_confidence']:.2f}, {result['distance_confidence']:.2f})"
        )
        return result

    def _get_game_clock_consensus(self) -> dict[str, Any]:
        """Get consensus for game clock from burst results."""
        clock_votes = {}

        for result in self.burst_results:
            clock = result.get("game_clock")
            confidence = result.get("confidence", 0.0)

            if clock:
                if clock not in clock_votes:
                    clock_votes[clock] = {"votes": 0, "total_confidence": 0.0, "frames": []}
                clock_votes[clock]["votes"] += 1
                clock_votes[clock]["total_confidence"] += confidence
                clock_votes[clock]["frames"].append(result.get("frame_number", "unknown"))

        # Find best clock with temporal validation
        best_clock = None
        best_score = 0
        temporal_issues = []

        for clock, data in clock_votes.items():
            avg_confidence = data["total_confidence"] / data["votes"]
            score = data["votes"] * avg_confidence

            # Apply temporal validation penalty
            is_valid, reason = self._validate_game_clock_temporal(clock)
            if not is_valid:
                temporal_issues.append(f"{clock}: {reason}")
                score *= 0.5  # Penalty for temporal issues

            if score > best_score:
                best_score = score
                best_clock = clock

        result = {
            "game_clock": best_clock,
            "confidence": best_score / len(self.burst_results) if best_clock else 0.0,
            "votes": clock_votes,
            "temporal_issues": temporal_issues,
        }

        logger.info(f"🎯 GAME CLOCK CONSENSUS: {best_clock} (conf: {result['confidence']:.2f})")
        if temporal_issues:
            logger.warning(f"⚠️ Temporal issues detected: {temporal_issues}")

        return result

    def _get_play_clock_consensus(self) -> dict[str, Any]:
        """Get consensus for play clock from burst results."""
        clock_votes = {}

        for result in self.burst_results:
            clock = result.get("play_clock")
            confidence = result.get("confidence", 0.0)

            if clock:
                if clock not in clock_votes:
                    clock_votes[clock] = {"votes": 0, "total_confidence": 0.0}
                clock_votes[clock]["votes"] += 1
                clock_votes[clock]["total_confidence"] += confidence

        # Find best play clock
        best_clock = None
        best_score = 0

        for clock, data in clock_votes.items():
            avg_confidence = data["total_confidence"] / data["votes"]
            score = data["votes"] * avg_confidence

            if score > best_score:
                best_score = score
                best_clock = clock

        result = {
            "play_clock": best_clock,
            "confidence": best_score / len(self.burst_results) if best_clock else 0.0,
            "votes": clock_votes,
        }

        logger.info(f"🎯 PLAY CLOCK CONSENSUS: {best_clock} (conf: {result['confidence']:.2f})")
        return result

    def _get_yard_line_consensus(self) -> dict[str, Any]:
        """Get consensus for yard line from burst results."""
        yard_votes = {}

        for result in self.burst_results:
            yard_line = result.get("yard_line")
            confidence = result.get("confidence", 0.0)

            if yard_line is not None:
                key = str(yard_line)
                if key not in yard_votes:
                    yard_votes[key] = {"votes": 0, "total_confidence": 0.0}
                yard_votes[key]["votes"] += 1
                yard_votes[key]["total_confidence"] += confidence

        # Find best yard line
        best_yard_line = None
        best_score = 0

        for yard_str, data in yard_votes.items():
            avg_confidence = data["total_confidence"] / data["votes"]
            score = data["votes"] * avg_confidence

            if score > best_score:
                best_score = score
                best_yard_line = int(yard_str)

        result = {
            "yard_line": best_yard_line,
            "confidence": best_score / len(self.burst_results) if best_yard_line else 0.0,
            "votes": yard_votes,
        }

        logger.info(f"🎯 YARD LINE CONSENSUS: {best_yard_line} (conf: {result['confidence']:.2f})")
        return result

    def _get_possession_consensus(self) -> dict[str, Any]:
        """Get consensus for possession from burst results."""
        possession_votes = {}

        for result in self.burst_results:
            possession = result.get("possession_team")
            confidence = result.get("confidence", 0.0)

            if possession:
                if possession not in possession_votes:
                    possession_votes[possession] = {"votes": 0, "total_confidence": 0.0}
                possession_votes[possession]["votes"] += 1
                possession_votes[possession]["total_confidence"] += confidence

        # Find best possession
        best_possession = None
        best_score = 0

        for possession, data in possession_votes.items():
            avg_confidence = data["total_confidence"] / data["votes"]
            score = data["votes"] * avg_confidence

            if score > best_score:
                best_score = score
                best_possession = possession

        result = {
            "possession_team": best_possession,
            "confidence": best_score / len(self.burst_results) if best_possession else 0.0,
            "votes": possession_votes,
        }

        logger.info(
            f"🎯 POSSESSION CONSENSUS: {best_possession} (conf: {result['confidence']:.2f})"
        )
        return result

    def _get_territory_consensus(self) -> dict[str, Any]:
        """Get consensus for territory from burst results."""
        territory_votes = {}

        for result in self.burst_results:
            territory = result.get("territory")
            confidence = result.get("confidence", 0.0)

            if territory:
                if territory not in territory_votes:
                    territory_votes[territory] = {"votes": 0, "total_confidence": 0.0}
                territory_votes[territory]["votes"] += 1
                territory_votes[territory]["total_confidence"] += confidence

        # Find best territory
        best_territory = None
        best_score = 0

        for territory, data in territory_votes.items():
            avg_confidence = data["total_confidence"] / data["votes"]
            score = data["votes"] * avg_confidence

            if score > best_score:
                best_score = score
                best_territory = territory

        result = {
            "territory": best_territory,
            "confidence": best_score / len(self.burst_results) if best_territory else 0.0,
            "votes": territory_votes,
        }

        logger.info(f"🎯 TERRITORY CONSENSUS: {best_territory} (conf: {result['confidence']:.2f})")
        return result

    def _validate_down_distance_progression(self, down: int, distance: int) -> dict[str, Any]:
        """Validate if down/distance makes sense in context of previous plays."""
        if not self.burst_results or len(self.burst_results) < 2:
            return {"valid": True, "reason": "Insufficient history for validation"}

        # Look for patterns that suggest this is reasonable
        down_counts = {}
        for result in self.burst_results:
            result_down = result.get("down")
            if result_down:
                down_counts[result_down] = down_counts.get(result_down, 0) + 1

        # Check for impossible progressions
        if down == 4 and distance > 25:
            return {"valid": False, "reason": f"Unlikely 4th & {distance} situation"}

        if down == 1 and distance > 30:
            return {"valid": False, "reason": f"Impossible 1st & {distance} situation"}

        return {"valid": True, "reason": "Progression appears reasonable"}

    def clear_burst_results(self) -> None:
        """Clear burst consensus results."""
        self.burst_results = []

    def _should_process_region(
        self, current_roi: np.ndarray, region_type: str, similarity_threshold: float = 0.95
    ) -> bool:
        """
        Determine if region needs processing based on visual changes.

        Args:
            current_roi: Current region of interest
            region_type: Type of region (down_distance, game_clock, etc.)
            similarity_threshold: Threshold for considering regions similar

        Returns:
            True if region should be processed, False if cached result can be used
        """
        try:
            if hasattr(self, "previous_regions") and region_type in self.previous_regions:
                previous_roi = self.previous_regions[region_type]

                # Ensure both ROIs have the same dimensions
                if current_roi.shape != previous_roi.shape:
                    self.previous_regions[region_type] = current_roi.copy()
                    return True

                # Calculate structural similarity using template matching
                if current_roi.size > 0 and previous_roi.size > 0:
                    # Convert to grayscale if needed
                    if len(current_roi.shape) == 3:
                        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        current_gray = current_roi

                    if len(previous_roi.shape) == 3:
                        previous_gray = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        previous_gray = previous_roi

                    # Use template matching for similarity
                    result = cv2.matchTemplate(current_gray, previous_gray, cv2.TM_CCOEFF_NORMED)
                    similarity = result[0][0] if result.size > 0 else 0.0

                    if similarity > similarity_threshold:
                        return False  # Skip processing, use cached result

            # Store current region for next comparison
            self.previous_regions[region_type] = current_roi.copy()
            return True

        except Exception as e:
            # If comparison fails, always process the region
            logger.debug(f"Region similarity check failed for {region_type}: {e}")
            self.previous_regions[region_type] = current_roi.copy()
            return True

    def _get_pooled_game_state(self) -> GameState:
        """Get a GameState object from the pool for memory efficiency."""
        try:
            if self.object_pools["game_state_pool"]:
                game_state = self.object_pools["game_state_pool"].pop()
                # Reset the game state
                game_state.possession_team = None
                game_state.territory = None
                game_state.down = None
                game_state.distance = None
                game_state.yard_line = None
                game_state.score_home = None
                game_state.score_away = None
                game_state.home_team = None
                game_state.away_team = None
                game_state.quarter = None
                game_state.time = None
                game_state.confidence = 0.0
                game_state.visualization_layers = None
                return game_state
            else:
                # Pool is empty, create new one
                return GameState()
        except Exception:
            return GameState()

    def _return_pooled_game_state(self, game_state: GameState) -> None:
        """Return a GameState object to the pool."""
        try:
            if len(self.object_pools["game_state_pool"]) < 10:
                self.object_pools["game_state_pool"].append(game_state)
        except Exception:
            pass  # If return fails, just let it be garbage collected

    def _get_cached_preprocessing(
        self, roi: np.ndarray, preprocess_type: str
    ) -> Optional[np.ndarray]:
        """Get cached preprocessing result if available."""
        try:
            # Create a simple hash of the ROI for caching
            roi_hash = hash(roi.tobytes()) if roi.size > 0 else 0
            cache_key = f"{preprocess_type}_{roi_hash}_{roi.shape}"

            if cache_key in self.preprocessing_cache:
                return self.preprocessing_cache[cache_key].copy()
            return None
        except Exception:
            return None

    def _cache_preprocessing_result(
        self, roi: np.ndarray, preprocess_type: str, result: np.ndarray
    ) -> None:
        """Cache preprocessing result for future use."""
        try:
            # Limit cache size to prevent memory bloat
            if len(self.preprocessing_cache) > 50:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.preprocessing_cache))
                del self.preprocessing_cache[oldest_key]

            roi_hash = hash(roi.tobytes()) if roi.size > 0 else 0
            cache_key = f"{preprocess_type}_{roi_hash}_{roi.shape}"
            self.preprocessing_cache[cache_key] = result.copy()
        except Exception:
            pass  # If caching fails, continue without cache

    def _get_cached_kernel(self, kernel_type: str, size: tuple) -> Optional[np.ndarray]:
        """Get cached morphological kernel."""
        try:
            cache_key = f"{kernel_type}_{size}"
            if cache_key not in self.object_pools["cv2_kernels"]:
                if kernel_type == "ellipse":
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
                elif kernel_type == "rect":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
                self.object_pools["cv2_kernels"][cache_key] = kernel
            return self.object_pools["cv2_kernels"][cache_key]
        except Exception:
            return None

    def _analyze_triangle_direction(
        self, region_roi: np.ndarray, triangle_type: str
    ) -> Optional[str]:
        """Analyze triangle direction using template matching (proven 97.6% accuracy)."""
        try:
            if region_roi is None or region_roi.size == 0:
                return None

            # Convert to grayscale if needed
            if len(region_roi.shape) == 3:
                gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_roi.copy()

            # Apply preprocessing for better template matching
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Define triangle templates (simple geometric shapes)
            h, w = gray.shape
            template_size = min(h, w) // 4

            if template_size < 10:  # Too small to analyze
                return None

            # Create triangle templates for different directions
            templates = {}

            # Left-pointing triangle (for possession)
            left_template = np.zeros((template_size, template_size), dtype=np.uint8)
            pts = np.array(
                [[template_size - 1, template_size // 2], [0, 0], [0, template_size - 1]], np.int32
            )
            cv2.fillPoly(left_template, [pts], 255)
            templates["left"] = left_template

            # Right-pointing triangle (for possession)
            right_template = np.zeros((template_size, template_size), dtype=np.uint8)
            pts = np.array(
                [
                    [0, template_size // 2],
                    [template_size - 1, 0],
                    [template_size - 1, template_size - 1],
                ],
                np.int32,
            )
            cv2.fillPoly(right_template, [pts], 255)
            templates["right"] = right_template

            # Up-pointing triangle (for territory)
            up_template = np.zeros((template_size, template_size), dtype=np.uint8)
            pts = np.array(
                [
                    [template_size // 2, 0],
                    [0, template_size - 1],
                    [template_size - 1, template_size - 1],
                ],
                np.int32,
            )
            cv2.fillPoly(up_template, [pts], 255)
            templates["up"] = up_template

            # Down-pointing triangle (for territory)
            down_template = np.zeros((template_size, template_size), dtype=np.uint8)
            pts = np.array(
                [[template_size // 2, template_size - 1], [0, 0], [template_size - 1, 0]], np.int32
            )
            cv2.fillPoly(down_template, [pts], 255)
            templates["down"] = down_template

            # Perform template matching
            best_match = None
            best_score = 0.0

            for direction, template in templates.items():
                # Skip irrelevant directions based on triangle type
                if triangle_type == "possession" and direction in ["up", "down"]:
                    continue
                if triangle_type == "territory" and direction in ["left", "right"]:
                    continue

                # Template matching
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_match = direction

            # Return result if confidence is high enough
            if best_score > 0.3:  # Threshold for triangle detection
                logger.debug(
                    f"Triangle direction detected: {best_match} (confidence: {best_score:.3f})"
                )
                return best_match

            return None

        except Exception as e:
            logger.error(f"Error in triangle direction analysis: {e}")
            return None

    def export_debug_data(self, output_dir="debug_output"):
        """Export all collected debug data for analysis"""
        if not self.debug_mode:
            print("Debug mode not enabled - no data to export")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)

            # Export clips data
            clips_file = os.path.join(output_dir, "debug_clips_data.json")
            with open(clips_file, "w") as f:
                json.dump(self.debug_data["clips"], f, indent=2, default=str)

            # Export logs
            logs_file = os.path.join(output_dir, "analysis_logs.txt")
            with open(logs_file, "w") as f:
                f.write("\n".join(self.debug_data["logs"]))

            # Export frame analysis data
            frame_data_file = os.path.join(output_dir, "frame_analysis_data.pkl")
            with open(frame_data_file, "wb") as f:
                pickle.dump(self.debug_data["frame_analysis"], f)

            # Export OCR and YOLO data separately
            ocr_file = os.path.join(output_dir, "ocr_results.json")
            with open(ocr_file, "w") as f:
                json.dump(self.debug_data["ocr_results"], f, indent=2, default=str)

            yolo_file = os.path.join(output_dir, "yolo_detections.json")
            with open(yolo_file, "w") as f:
                json.dump(self.debug_data["yolo_detections"], f, indent=2, default=str)

            # Create summary report
            summary = {
                "analysis_summary": {
                    "total_frames_analyzed": len(self.debug_data["frame_analysis"]),
                    "total_clips_detected": len(self.debug_data["clips"]),
                    "total_log_entries": len(self.debug_data["logs"]),
                    "export_timestamp": datetime.now().isoformat(),
                },
                "clip_situations": [clip["situation"] for clip in self.debug_data["clips"]],
                "common_issues": self._analyze_common_issues(),
            }

            summary_file = os.path.join(output_dir, "debug_summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"🔍 Debug data exported to {output_dir}/")
            print(f"   - {len(self.debug_data['clips'])} clips detected")
            print(f"   - {len(self.debug_data['frame_analysis'])} frames analyzed")
            print(f"   - {len(self.debug_data['logs'])} log entries")

            return output_dir

        except Exception as e:
            print(f"Error exporting debug data: {str(e)}")
            return None

    def _analyze_common_issues(self):
        """Analyze debug data for common issues"""
        issues = []

        # Check for repeated "1st & 10" clips
        first_down_clips = [
            c for c in self.debug_data["clips"] if "1st & 10" in c.get("situation", "")
        ]
        if len(first_down_clips) > len(self.debug_data["clips"]) * 0.7:  # More than 70%
            issues.append(
                {
                    "type": "excessive_first_downs",
                    "description": f'{len(first_down_clips)} out of {len(self.debug_data["clips"])} clips labeled as "1st & 10"',
                    "severity": "high",
                }
            )

        # Check for low OCR confidence
        low_confidence_ocr = []
        for frame_data in self.debug_data["frame_analysis"].values():
            ocr_results = frame_data.get("ocr_results", {})
            for region, result in ocr_results.items():
                if result.get("confidence", 1.0) < 0.5:
                    low_confidence_ocr.append((region, result.get("confidence", 0)))

        if len(low_confidence_ocr) > 10:
            issues.append(
                {
                    "type": "low_ocr_confidence",
                    "description": f"{len(low_confidence_ocr)} OCR results with confidence < 0.5",
                    "severity": "medium",
                }
            )

        # Check for state persistence patterns
        persistence_logs = [log for log in self.debug_data["logs"] if "persistence" in log.lower()]
        if len(persistence_logs) > 50:
            issues.append(
                {
                    "type": "excessive_state_persistence",
                    "description": f"{len(persistence_logs)} state persistence events detected",
                    "severity": "medium",
                }
            )

        return issues

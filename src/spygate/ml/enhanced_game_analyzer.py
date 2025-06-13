"""
Enhanced game analyzer module for SpygateAI.
Handles game state detection and analysis using YOLOv8 and OpenCV.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..core.hardware import HardwareDetector, HardwareTier
from ..core.optimizer import TierOptimizer
from .yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig, MODEL_CONFIGS
from .enhanced_ocr import EnhancedOCR
from .template_triangle_detector import TemplateTriangleDetector

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
        "target_fps": 0.2
    },
    "LOW": {
        "model_size": "n", 
        "img_size": 416, 
        "batch_size": 2,
        "device": "auto", 
        "conf": 0.3, 
        "target_fps": 0.5
    },
    "MEDIUM": {
        "model_size": "s", 
        "img_size": 640, 
        "batch_size": 4,
        "half": True, 
        "quantize": True, 
        "target_fps": 1.0
    },
    "HIGH": {
        "model_size": "m", 
        "img_size": 832, 
        "batch_size": 8,
        "compile": True, 
        "target_fps": 2.0
    },
    "ULTRA": {
        "model_size": "l", 
        "img_size": 1280, 
        "batch_size": 16,
        "optimize": True, 
        "target_fps": 2.5
    }
}

# UI element classes for detection
UI_CLASSES = [
    "hud",                      # Main HUD bar
    "possession_triangle_area",  # Left triangle (possession)
    "territory_triangle_area",   # Right triangle (territory)
    "preplay_indicator",        # Pre-play state
    "play_call_screen"          # Play selection overlay
]

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
    visualization_layers: Dict[str, np.ndarray] = None

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
    
    def get_average_metrics(self) -> Dict[str, float]:
        return {
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0.0,
            "avg_accuracy": np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
            }

@dataclass
class DetectionResult:
    """Detection result from YOLOv8."""
    class_id: int
    confidence: float
    bbox: List[int]  # [x, y, w, h]
    
@dataclass
class SituationContext:
    """Enhanced situation context with possession/territory awareness."""
    possession_team: str = "unknown"  # "user" or "opponent"
    territory: str = "unknown"  # "own" or "opponent"
    situation_type: str = "normal"
    pressure_level: str = "low"  # "low", "medium", "high", "critical"
    leverage_index: float = 0.5  # 0.0-1.0 situational importance
    
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
            self.red_zone_efficiency + self.third_down_conversion + 
            self.turnover_avoidance + self.clock_management + 
            self.field_position_awareness
        ) / 5
        
        execution_quality = (
            self.pressure_performance + self.clutch_factor + 
            self.consistency + self.adaptability + 
            self.momentum_management
        ) / 5
        
        strategic_depth = (
            self.formation_diversity + self.situational_play_calling + 
            self.opponent_exploitation + self.game_flow_reading + 
            self.risk_reward_balance
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
        debug_output_dir: Optional[Path] = None
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
        if hardware is not None and hasattr(hardware, '__class__') and 'HardwareTier' in str(type(hardware)):
            raise TypeError(
                f"Expected HardwareDetector instance, got HardwareTier: {hardware}. "
                f"Pass HardwareDetector() instance instead."
            )
        
        self.hardware = hardware or HardwareDetector()
        self.optimization_config = optimization_config
        
        # Use custom trained HUD model if no specific path provided
        if model_path is None:
            model_path = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"

        # Initialize YOLO model with hardware optimization
        self.model = EnhancedYOLOv8(
            model_path=model_path,
            hardware_tier=self.hardware.detect_tier()  # Use detect_tier() method
        )
        # Set confidence threshold if supported
        if hasattr(self.model, 'conf'):
            self.model.conf = 0.25  # Default confidence threshold

        # Initialize OCR engine
        self.ocr = EnhancedOCR(hardware=self.hardware.detect_tier())  # Use detect_tier() method
        
        # Initialize game state tracking
        self.current_state = None
        self.state_history = []
        self.confidence_threshold = 0.25
        self.detection_interval = 30
        
        # Add colors for visualization
        self.colors = {
            "hud": (0, 255, 0),  # Green
            "possession_triangle_area": (255, 0, 0),  # Blue  
            "territory_triangle_area": (0, 0, 255),  # Red
            "preplay_indicator": (255, 255, 0),  # Cyan
            "play_call_screen": (255, 0, 255),  # Magenta
        }

        # UI Classes for our custom model
        self.ui_classes = [
            "hud",
            "possession_triangle_area", 
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen"
        ]
        
        logger.info(
            f"Enhanced Game Analyzer initialized for {self.hardware.detect_tier().name} hardware"
        )
        
        self.triangle_detector = TemplateTriangleDetector(debug_output_dir)
        
        # Advanced situation tracking
        self.hidden_mmr = HiddenMMRMetrics()
        self.situation_history = []
        self.drive_tracking = {
            "current_drive": None,
            "drive_history": [],
            "possession_changes": []
        }
        
        # Performance tracking (hidden from user)
        self.performance_stats = {
            "offensive_situations": defaultdict(list),
            "defensive_situations": defaultdict(list),
            "transition_moments": [],
            "decision_quality": []
        }
        
        # User context (set by application)
        self.user_team = None  # "home" or "away"
        self.analysis_context = "self"  # "self", "opponent", "pro_study"
        self.last_possession_direction = None
        self.last_territory_direction = None
        self.direction_confidence_threshold = 0.4
        
        # Initialize game state dictionary
        self.game_state = {}
        
        # Initialize class mapping for YOLO detections
        self.class_map = {
            "hud": 0,
            "possession_triangle_area": 1,
            "territory_triangle_area": 2,
            "preplay_indicator": 3,
            "play_call_screen": 4
        }
        
        # Initialize key moments and clip queues
        self.key_moments = []
        self.clip_queue = []
        
        # Initialize HUD state tracking
        self.hud_state = {
            "is_visible": True,
            "frames_since_visible": 0,
            "in_game_interruption": False,
            "max_frames_without_hud": 75,  # 2.5 seconds at 30fps
            "game_interruption_frames": 75,
            "last_valid_state": {}
        }
        
        # Initialize performance metrics
        self.performance_metrics = {
            "inference_times": deque(maxlen=100),
            "confidence_scores": deque(maxlen=100)
        }
    
    def analyze_frame(self, frame: np.ndarray) -> GameState:
        """Analyze a single frame and return game state with visualization layers."""
        # Create copy for visualization
        vis_frame = frame.copy()
        
        # Run YOLOv8 detection
        detections = self.model.detect(frame)
        
        # Initialize visualization layers
        layers = {
            "hud_detection": frame.copy(),
            "triangle_detection": frame.copy(),
            "ocr_results": frame.copy()
        }
        
        # Process triangle orientations
        possession_results = []
        territory_results = []
        
        for detection in detections:
            if detection.cls == self.class_map["possession_triangle_area"]:
                # Extract ROI from YOLO detection
                x1, y1, x2, y2 = detection.xyxy[0]
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Use template matching to find possession triangles
                triangles = self.triangle_detector.detect_triangles_in_roi(roi, "possession")
                if triangles:
                    # Select the best single triangle
                    best_triangle = self.triangle_detector.select_best_single_triangles(triangles, "possession")
                    if best_triangle:
                        possession_results.append(best_triangle)
                        self.last_possession_direction = best_triangle['direction']
            
            elif detection.cls == self.class_map["territory_triangle_area"]:
                # Extract ROI from YOLO detection
                x1, y1, x2, y2 = detection.xyxy[0]
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Use template matching to find territory triangles
                triangles = self.triangle_detector.detect_triangles_in_roi(roi, "territory")
                if triangles:
                    # Select the best single triangle
                    best_triangle = self.triangle_detector.select_best_single_triangles(triangles, "territory")
                    if best_triangle:
                        territory_results.append(best_triangle)
                        self.last_territory_direction = best_triangle['direction']
        
        # Update game state with triangle information and detect flips
        if possession_results:
            # Use highest confidence result
            best_possession = max(possession_results, key=lambda x: x['confidence'])
            new_possession_direction = best_possession['direction']
            
            # Detect possession change (triangle flip)
            if (self.last_possession_direction and 
                self.last_possession_direction != new_possession_direction):
                self._handle_possession_change(self.last_possession_direction, new_possession_direction)
            
            self.game_state["possession"] = {
                "direction": new_possession_direction,
                "confidence": best_possession['confidence'],
                "team_with_ball": self._get_team_with_ball(new_possession_direction)
            }
            self.last_possession_direction = new_possession_direction
        elif self.last_possession_direction:
            # Use last known direction with reduced confidence
            self.game_state["possession"] = {
                "direction": self.last_possession_direction,
                "confidence": self.direction_confidence_threshold / 2,
                "team_with_ball": self._get_team_with_ball(self.last_possession_direction)
            }
        
        if territory_results:
            # Use highest confidence result
            best_territory = max(territory_results, key=lambda x: x['confidence'])
            new_territory_direction = best_territory['direction']
            
            # Detect territory change (triangle flip)
            if (self.last_territory_direction and 
                self.last_territory_direction != new_territory_direction):
                self._handle_territory_change(self.last_territory_direction, new_territory_direction)
            
            self.game_state["territory"] = {
                "direction": new_territory_direction,
                "confidence": best_territory['confidence'],
                "field_context": self._get_field_context(new_territory_direction)
            }
            self.last_territory_direction = new_territory_direction
        elif self.last_territory_direction:
            # Use last known direction with reduced confidence
            self.game_state["territory"] = {
                "direction": self.last_territory_direction,
                "confidence": self.direction_confidence_threshold / 2,
                "field_context": self._get_field_context(self.last_territory_direction)
        }
        
        # Process detections and update visualization layers
        game_state = self._process_detections(detections, layers)
        game_state.visualization_layers = layers
        
        return game_state
    
    def _process_detections(self, detections: List[Dict], layers: Dict[str, np.ndarray]) -> GameState:
        """Process detections and update visualization layers."""
        game_state = GameState()
        total_confidence = 0
        num_detections = 0
        
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Update total confidence
            total_confidence += conf
            num_detections += 1
            
            # Draw detection on appropriate layer
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[class_name]
            
            if class_name == "hud":
                cv2.rectangle(layers["hud_detection"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(layers["hud_detection"], f"{class_name} {conf:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Extract game state from HUD region
                self._extract_hud_info(layers["hud_detection"][y1:y2, x1:x2], game_state)
                
            elif "triangle" in class_name:
                cv2.rectangle(layers["triangle_detection"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(layers["triangle_detection"], f"{class_name} {conf:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update possession/territory based on triangle detection
                if class_name == "possession_triangle_area":
                    game_state.possession_team = "Left Team"  # Simplified for now
                elif class_name == "territory_triangle_area":
                    game_state.territory = "Red Zone"  # Simplified for now
            
            else:  # preplay_indicator or play_call_screen
                cv2.rectangle(layers["ocr_results"], (x1, y1), (x2, y2), color, 2)
                cv2.putText(layers["ocr_results"], f"{class_name} {conf:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update overall confidence
        game_state.confidence = total_confidence / num_detections if num_detections > 0 else 0.0
        
        return game_state
    
    def _extract_hud_info(self, hud_region: np.ndarray, game_state: GameState) -> None:
        """Extract game state information from HUD region using enhanced OCR."""
        try:
            # Use our enhanced OCR system for professional-grade extraction
            if not hasattr(self, 'enhanced_ocr'):
                self.enhanced_ocr = EnhancedOCR(hardware=self.hardware)
            
            # Process the HUD region with our proven OCR pipeline
            ocr_results = self.enhanced_ocr.process_hud_region(hud_region)
            
            # Extract down and distance with high accuracy
            if 'down' in ocr_results:
                down_data = ocr_results['down']
                if down_data['confidence'] >= 0.7:  # High confidence threshold
                    game_state.down = down_data['text']
                    
            if 'distance' in ocr_results:
                distance_data = ocr_results['distance']
                if distance_data['confidence'] >= 0.7:
                    game_state.distance = distance_data['text']
            
            # Extract team scores
            if 'left_score' in ocr_results:
                score_data = ocr_results['left_score']
                if score_data['confidence'] >= 0.6:  # Slightly lower threshold for scores
                    game_state.score_away = score_data['text']
                    
            if 'right_score' in ocr_results:
                score_data = ocr_results['right_score']
                if score_data['confidence'] >= 0.6:
                    game_state.score_home = score_data['text']
            
            # Extract team abbreviations
            if 'left_team' in ocr_results:
                team_data = ocr_results['left_team']
                if team_data['confidence'] >= 0.8:  # High threshold for team names
                    game_state.away_team = team_data['text']
                    
            if 'right_team' in ocr_results:
                team_data = ocr_results['right_team']
                if team_data['confidence'] >= 0.8:
                    game_state.home_team = team_data['text']
            
            # Extract yard line information
            if 'yard_line' in ocr_results:
                yard_data = ocr_results['yard_line']
                if yard_data['confidence'] >= 0.6:
                    game_state.yard_line = int(yard_data['text'])
            
            # Apply professional-grade down detection enhancement
            self._enhance_down_detection(hud_region, game_state, ocr_results)
            
            # Calculate overall confidence based on successful extractions
            successful_extractions = sum(1 for key in ['down', 'distance', 'left_score', 'right_score'] 
                                       if key in ocr_results and ocr_results[key]['confidence'] >= 0.6)
            game_state.confidence = min(0.95, successful_extractions / 4.0)
            
            # Log extraction results for debugging
            if successful_extractions >= 2:
                logger.debug(f"HUD extraction successful: {successful_extractions}/4 elements detected")
                if game_state.down and game_state.distance:
                    logger.debug(f"Down & Distance: {game_state.down} & {game_state.distance}")
            else:
                logger.debug(f"HUD extraction partial: only {successful_extractions}/4 elements detected")
                
        except Exception as e:
            logger.error(f"Error in HUD extraction: {e}")
            # Fallback to previous values or defaults
            game_state.confidence = 0.0

    def _enhance_down_detection(self, hud_region: np.ndarray, game_state: GameState, ocr_results: Dict[str, Any]) -> None:
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
            if not hasattr(self, 'down_detection_history'):
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
            if smoothed_results and smoothed_results['confidence'] >= self.down_confidence_threshold:
                if 'down' in smoothed_results and smoothed_results['down'] is not None:
                    game_state.down = smoothed_results['down']
                if 'distance' in smoothed_results and smoothed_results['distance'] is not None:
                    game_state.distance = smoothed_results['distance']
                    
                logger.debug(f"Enhanced down detection: {game_state.down} & {game_state.distance} (confidence: {smoothed_results['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Error in enhanced down detection: {e}")

    def _preprocess_down_region(self, down_region: np.ndarray) -> np.ndarray:
        """Apply professional-grade preprocessing to down/distance region."""
        # Scale up for better OCR (same approach as triangle detection)
        scale_factor = 5
        scaled_height, scaled_width = down_region.shape[0] * scale_factor, down_region.shape[1] * scale_factor
        scaled_region = cv2.resize(down_region, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(scaled_region.shape) == 3:
            gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = scaled_region
        
        # Apply high-contrast preprocessing for clean OCR
        _, thresh_region = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh_region

    def _multi_engine_down_detection(self, enhanced_region: np.ndarray) -> List[Dict[str, Any]]:
        """Multi-engine OCR approach with fallback (similar to triangle detection)."""
        results = []
        
        try:
            # Engine 1: EasyOCR (primary)
            if hasattr(self.enhanced_ocr, 'reader'):
                easyocr_results = self.enhanced_ocr.reader.readtext(enhanced_region)
                for bbox, text, conf in easyocr_results:
                    results.append({
                        'text': text.strip(),
                        'confidence': conf,
                        'source': 'easyocr',
                        'bbox': bbox
                    })
        except Exception as e:
            logger.debug(f"EasyOCR failed for down detection: {e}")
        
        try:
            # Engine 2: Tesseract (fallback)
            import pytesseract
            from PIL import Image
            
            # Optimal Tesseract config for Madden HUD text
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP '
            
            pil_image = Image.fromarray(enhanced_region)
            tesseract_text = pytesseract.image_to_string(pil_image, config=custom_config).strip()
            
            if tesseract_text:
                results.append({
                    'text': tesseract_text,
                    'confidence': 0.8,  # Default confidence for Tesseract
                    'source': 'tesseract',
                    'bbox': None
                })
                
        except Exception as e:
            logger.debug(f"Tesseract failed for down detection: {e}")
        
        return results

    def _validate_down_results(self, down_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced pattern matching and validation (similar to triangle selection)."""
        best_result = {'confidence': 0.0, 'down': None, 'distance': None}
        
        # Comprehensive down/distance patterns (from our previous implementations)
        down_patterns = [
            r'(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)',  # "1st & 10", "3rd & 8"
            r'(\d+)(?:st|nd|rd|th)?\s*&\s*Goal',   # "1st & Goal"
            r'(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)',  # Variations
            r'(\d+)\s*&\s*(\d+)',                  # Simple "3 & 8"
            r'(\d+)(?:nd|rd|th|st)\s*&\s*(\d+)',   # OCR variations
        ]
        
        for result in down_results:
            text = result['text']
            confidence = result['confidence']
            
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
                            if distance_text.lower() == 'goal':
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
                        if weighted_confidence > best_result['confidence']:
                            best_result = {
                                'confidence': weighted_confidence,
                                'down': down,
                                'distance': distance,
                                'source': result['source'],
                                'raw_text': text
                            }
                            
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing down/distance: {e}")
                        continue
        
        return best_result

    def _apply_temporal_smoothing(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing for consistency (similar to triangle detection)."""
        if not current_result or current_result['confidence'] < 0.5:
            return current_result
            
        # Add to history
        self.down_detection_history.append(current_result)
        
        # If we have enough history, apply smoothing
        if len(self.down_detection_history) >= 3:
            # Get recent results
            recent_results = list(self.down_detection_history)[-5:]
            
            # Find most common down value
            down_values = [r['down'] for r in recent_results if r['down'] is not None]
            if down_values:
                from collections import Counter
                most_common_down = Counter(down_values).most_common(1)[0][0]
                
                # If current result matches most common, boost confidence
                if current_result['down'] == most_common_down:
                    current_result['confidence'] = min(0.95, current_result['confidence'] + 0.1)
                    
                # If current result differs significantly, reduce confidence
                elif abs(current_result['down'] - most_common_down) > 1:
                    current_result['confidence'] *= 0.8
        
        return current_result

    def optimize_memory(self):
        """Optimize memory usage based on hardware tier."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        hardware_tier = self.hardware.detect_tier()
        if hardware_tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            # Aggressive optimization for low-end systems
            self.detection_interval = 45  # Reduce detection frequency
            torch.cuda.empty_cache()  # Clear CUDA cache
            
        elif hardware_tier == HardwareTier.MEDIUM:
            self.detection_interval = 30
            
        else:  # HIGH and ULTRA
            self.detection_interval = 15  # More frequent updates
            
    def log_performance_metrics(self):
        """Log current performance metrics."""
        if self.performance_metrics["inference_times"]:
            avg_inference = np.mean(self.performance_metrics["inference_times"][-100:])
            avg_fps = 1.0 / avg_inference if avg_inference > 0 else 0
            
            logger.info(f"Performance Metrics:")
            logger.info(f"- Average FPS: {avg_fps:.1f}")
            logger.info(f"- Average Inference Time: {avg_inference*1000:.1f}ms")
            logger.info(f"- Detection Rate: {self.detection_interval} frames")
            logger.info(f"- Average Confidence: {np.mean(self.performance_metrics['confidence_scores'][-100:]):.3f}")

    def set_user_team(self, team_abbrev: str, is_home: bool) -> None:
        """Set the user's team for analysis.
        
        Args:
            team_abbrev: User's team abbreviation (e.g., 'KC', 'SF')
            is_home: Whether the user is the home team (True) or away team (False)
        """
        self.user_team = team_abbrev.strip().upper()
        self.is_user_home = is_home

    def set_analysis_context(self, analysis_type: str, player_name: str = None, is_home: bool = None) -> None:
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
        if analysis_type not in ['self', 'opponent']:
            raise ValueError("analysis_type must be either 'self' or 'opponent'")
            
        self.analysis_type = analysis_type
        self.analyzed_player = player_name or "Unknown Player"
        
        if analysis_type == 'self':
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

    def detect_key_moment(self, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
                current_state["scores"]["analyzed_player"] != 
                self.last_game_state["scores"]["analyzed_player"]
            )
            opponent_score_changed = (
                current_state["scores"]["opponent"] != 
                self.last_game_state["scores"]["opponent"]
            )
            if analyzed_score_changed or opponent_score_changed:
                triggers.append("scoring_play")

        # Check possession changes
        if (self.key_moment_triggers["possession_change"] and
            current_state["possession"] != self.last_game_state["possession"]):
            triggers.append("possession_change")

        # Check for critical situations (only during active play)
        if (self.key_moment_triggers["critical_situation"] and 
            self.state_indicators["play_in_progress"]):
            situation = current_state.get("situation", {})
            if (situation.get("down") in [3, 4] or 
                situation.get("in_redzone") or
                self._is_two_minute_drill(situation.get("game_clock", ""))):
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
                    "is_home": self.is_user_home
                }
            }

        self.last_game_state = current_state
        return key_moment

    def _update_state_indicators(self, current_state: Dict[str, Any]) -> None:
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
        self.state_indicators["possession_triangle"] = current_state.get("possession_triangle_detected", False)
        self.state_indicators["territory_triangle"] = current_state.get("territory_triangle_detected", False)

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
            (not was_preplay and self.state_indicators["preplay_indicator"]) or
            (not was_playcall and self.state_indicators["play_call_screen"])
        ):
            self.play_state["is_play_active"] = False
            play_duration = current_time - self.play_state["play_start_time"]
            logger.info(f"Play {self.play_state['play_count']} ended after {play_duration:.2f} seconds")

    def is_play_active(self) -> bool:
        """Check if a play is currently active."""
        return self.play_state["is_play_active"]

    def get_current_play_info(self) -> Dict[str, Any]:
        """Get information about the current/last play."""
        return {
            "play_number": self.play_state["play_count"],
            "is_active": self.play_state["is_play_active"],
            "play_duration": time.time() - self.play_state["play_start_time"] if self.play_state["is_play_active"] else 0,
            "last_preplay_time": self.play_state["last_preplay_time"],
            "last_playcall_time": self.play_state["last_playcall_time"]
        }

    def _track_zone_change(self, current_state: Dict[str, Any]) -> None:
        """Track zone changes without generating clips."""
        if current_state.get("yard_line") and current_state.get("territory"):
            current_territory, current_zone = self._get_field_zone(
                current_state["yard_line"],
                current_state["territory"]
            )
            
            # Track zone change
            current_zone_full = f"{current_territory}_{current_zone}"
            if self.last_zone != current_zone_full:
                self.tracking_metrics["zone_changes"].append({
                    "timestamp": time.time(),
                    "from_zone": self.last_zone,
                    "to_zone": current_zone_full,
                    "yard_line": current_state["yard_line"]
                })
                self.last_zone = current_zone_full

    def _track_formation_sequence(self, current_state: Dict[str, Any]) -> None:
        """Track formation sequences without generating clips."""
        if current_state.get("formation"):
            formation_matched = self._detect_formation_match(current_state["formation"])
            if formation_matched:
                self.tracking_metrics["formation_sequences"].append({
                    "timestamp": time.time(),
                    "formation": current_state["formation"],
                    "previous_formation": self.last_formation,
                    "field_zone": self.last_zone  # Include zone context
                })
            self.last_formation = current_state["formation"]

    def _is_two_minute_drill(self, game_clock: str) -> bool:
        """Check if current time indicates two-minute drill situation."""
        try:
            minutes, seconds = map(int, game_clock.split(":"))
            return minutes < 2
        except (ValueError, AttributeError):
            return False

    def _get_field_zone(self, yard_line: int, territory: str) -> Tuple[str, str]:
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

    def _update_zone_stats(self, game_state: Dict[str, Any]) -> None:
        """Update statistics for the current field zone.
        
        Args:
            game_state: Current game state including yard line and play result
        """
        if not game_state.get('yard_line') or not game_state.get('territory'):
            return

        territory, zone = self._get_field_zone(
            game_state['yard_line'],
            game_state['territory']
        )
        
        # Update play count
        self.tracking_metrics["zone_stats"][territory][zone]["plays"] += 1
        
        # Update yards gained if available
        if game_state.get('yards_gained'):
            self.tracking_metrics["zone_stats"][territory][zone]["yards"] += game_state['yards_gained']
            
        # Update scores if touchdown or field goal
        if game_state.get('score_type'):
            self.tracking_metrics["zone_stats"][territory][zone]["scores"] += 1

    def extract_game_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract complete game state from a frame."""
        game_state = super().extract_game_state(frame)
        
        # Get current field zone if yard line available
        if game_state.get('yard_line') and game_state.get('territory'):
            current_territory, current_zone = self._get_field_zone(
                game_state['yard_line'],
                game_state['territory']
            )
            
            # Track zone change
            current_zone_full = f"{current_territory}_{current_zone}"
            if self.last_zone != current_zone_full:
                self.tracking_metrics["zone_changes"].append({
                    "timestamp": time.time(),
                    "from_zone": self.last_zone,
                    "to_zone": current_zone_full,
                    "yard_line": game_state['yard_line']
                })
                self.last_zone = current_zone_full
            
            # Update zone statistics
            self._update_zone_stats(game_state)
            
            # Add zone info to game state
            game_state['field_zone'] = {
                'territory': current_territory,
                'zone_name': current_zone
            }
        
        # Track formation matches internally
        if game_state.get('formation'):
            formation_matched = self._detect_formation_match(game_state['formation'])
            if formation_matched:
                self.tracking_metrics["formation_sequences"].append({
                    "timestamp": time.time(),
                    "formation": game_state['formation'],
                    "previous_formation": self.last_formation,
                    "field_zone": self.last_zone  # Include zone context
                })
            
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
        self.formation_history.append({
            'formation': current_formation,
            'timestamp': time.time(),
            'is_matching': False
        })
        
        # Keep only last 10 formations
        if len(self.formation_history) > 10:
            self.formation_history.pop(0)
            
        # Check if this formation is a known counter to the last one
        is_matching = self._check_formation_counter_database(
            previous=self.last_formation,
            current=current_formation
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

    def update_play_state(self, current_state: Dict[str, Any], timestamp: float) -> None:
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
                    self.play_state["current_play_duration"] = timestamp - self.play_state["play_start_time"]
                self.play_state["play_start_time"] = None

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
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
            }
        }
        
        return current_state

    def validate_clip_timing(self, current_state: GameState, frame_number: int) -> Dict[str, Any]:
        """Validate clip start/end timing using available detection methods."""
        # First update detection states with persistence logic
        current_state = self.update_detection_state(current_state, frame_number)
        
        timing_info = {
            "is_valid": False,
            "start_frame": None,
            "end_frame": None,
            "confidence": 0.0,
            "validation_methods": [],
            "persistence_used": False  # Track if we used persisted states
        }

        # Method 1: UI Element Sequence Validation with persistence
        ui_timing = self._validate_ui_sequence(current_state, frame_number)
        if ui_timing["is_valid"]:
            timing_info["validation_methods"].append("ui_sequence")
            if ui_timing.get("used_persistence", False):
                timing_info["persistence_used"] = True

        # Method 2: HUD State Changes with persistence
        hud_timing = self._validate_hud_state(current_state, frame_number)
        if hud_timing["is_valid"]:
            timing_info["validation_methods"].append("hud_state")
            if hud_timing.get("used_persistence", False):
                timing_info["persistence_used"] = True

        # Method 3: Possession Triangle State with persistence
        possession_timing = self._validate_possession_state(current_state, frame_number)
        if possession_timing["is_valid"]:
            timing_info["validation_methods"].append("possession_state")
            if possession_timing.get("used_persistence", False):
                timing_info["persistence_used"] = True

        # Adjust confidence based on persistence usage
        required_methods = 2  # Base requirement
        if timing_info["persistence_used"]:
            required_methods = 3  # Require more validation if using persisted states

        # Combine validations to determine final timing
        if len(timing_info["validation_methods"]) >= required_methods:
            timing_info["is_valid"] = True
            timing_info["start_frame"] = self._get_earliest_validated_start([
                ui_timing, hud_timing, possession_timing
            ])
            timing_info["end_frame"] = self._get_latest_validated_end([
                ui_timing, hud_timing, possession_timing
            ])
            
            # Adjust confidence based on persistence
            base_confidence = len(timing_info["validation_methods"]) / 3.0
            if timing_info["persistence_used"]:
                timing_info["confidence"] = base_confidence * 0.8  # 20% confidence penalty
            else:
                timing_info["confidence"] = base_confidence

        return timing_info

    def _validate_ui_sequence(self, current_state: GameState, frame_number: int) -> Dict[str, Any]:
        """Validate clip timing based on UI element sequence with persistence."""
        timing = {
            "is_valid": False, 
            "start_frame": None, 
            "end_frame": None,
            "used_persistence": False
        }

        # Check if we're using persisted states
        preplay_history = self.detection_history["preplay_indicator"]
        playcall_history = self.detection_history["play_call_screen"]
        
        if (preplay_history["persisted_state"] or 
            playcall_history["persisted_state"]):
            timing["used_persistence"] = True

        # Play start validation with persistence
        if (not current_state.state_indicators["preplay_indicator"] and 
            self.play_timing["last_preplay_frame"] is not None):
            # Pre-play disappeared after being visible
            timing["start_frame"] = max(0, frame_number - int(self.clip_config["pre_play_buffer"] * 30))
            timing["is_valid"] = True

        # Play end validation with persistence
        if (current_state.state_indicators["preplay_indicator"] or 
            current_state.state_indicators["play_call_screen"]):
            if current_state.play_active:
                timing["end_frame"] = frame_number + int(self.clip_config["post_play_buffer"] * 30)
                timing["is_valid"] = True

        # Update last seen frames
        if current_state.state_indicators["preplay_indicator"]:
            self.play_timing["last_preplay_frame"] = frame_number
        if current_state.state_indicators["play_call_screen"]:
            self.play_timing["last_playcall_frame"] = frame_number

        return timing

    def _validate_hud_state(self, current_state: GameState, frame_number: int) -> Dict[str, Any]:
        """Validate clip timing based on HUD state changes with persistence."""
        timing = {
            "is_valid": False, 
            "start_frame": None, 
            "end_frame": None,
            "used_persistence": False
        }

        if not self.last_game_state:
            return timing

        # Check if we're using persisted HUD state
        if self.detection_history["hud"]["persisted_state"]:
            timing["used_persistence"] = True

        # Track significant HUD changes with persistence
        if current_state.down is not None and current_state.distance is not None:
            if not self.play_timing["play_start_frame"]:
                timing["start_frame"] = frame_number
                timing["is_valid"] = True
            elif (current_state.down != self.last_game_state.down or 
                  current_state.distance != self.last_game_state.distance):
                timing["end_frame"] = frame_number
                timing["is_valid"] = True

        return timing

    def _validate_possession_state(self, current_state: GameState, frame_number: int) -> Dict[str, Any]:
        """Validate clip timing based on possession triangle state with persistence."""
        timing = {
            "is_valid": False, 
            "start_frame": None, 
            "end_frame": None,
            "used_persistence": False
        }

        if not self.last_game_state:
            return timing

        # Check if we're using persisted states
        if (self.detection_history["possession_triangle"]["persisted_state"] or
            self.detection_history["territory_triangle"]["persisted_state"]):
            timing["used_persistence"] = True

        # Check for possession changes with persistence
        if (current_state.state_indicators["possession_triangle"] and
            current_state.possession_team != self.last_game_state.possession_team):
            timing["end_frame"] = frame_number
            timing["is_valid"] = True

        # Check for territory changes with persistence
        if (current_state.state_indicators["territory_triangle"] and
            current_state.territory != self.last_game_state.territory):
            timing["end_frame"] = frame_number
            timing["is_valid"] = True

        return timing

    def _get_earliest_validated_start(self, timings: List[Dict[str, Any]]) -> int:
        """Get earliest validated start frame from multiple methods."""
        valid_starts = [t["start_frame"] for t in timings 
                       if t["is_valid"] and t["start_frame"] is not None]
        return min(valid_starts) if valid_starts else None

    def _get_latest_validated_end(self, timings: List[Dict[str, Any]]) -> int:
        """Get latest validated end frame from multiple methods."""
        valid_ends = [t["end_frame"] for t in timings 
                     if t["is_valid"] and t["end_frame"] is not None]
        return max(valid_ends) if valid_ends else None

    def create_clip(self, start_frame: int, end_frame: int) -> Dict[str, Any]:
        """Create a clip with the validated frame range."""
        if start_frame is None or end_frame is None:
            return None

        # Ensure we have enough buffer
        start_frame = max(0, start_frame - int(self.clip_config["pre_play_buffer"] * 30))
        end_frame = min(len(self.frame_buffer) - 1, 
                       end_frame + int(self.clip_config["post_play_buffer"] * 30))

        # Validate clip duration
        duration = (self.frame_timestamps[end_frame] - 
                   self.frame_timestamps[start_frame])
        
        if (duration < self.clip_config["min_play_duration"] or 
            duration > self.clip_config["max_play_duration"]):
            return None

        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": duration,
            "frames": self.frame_buffer[start_frame:end_frame + 1],
            "timestamps": self.frame_timestamps[start_frame:end_frame + 1]
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
                current_state.state_indicators.get(element, False)
                for element in region["elements"]
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

    def detect_occlusion_pattern(self, frame_regions: Dict[str, bool], frame_number: int) -> None:
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
                not frame_regions[region] 
                for region in self.hud_occlusion["occlusion_pattern"]
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
            1 for region in self.hud_occlusion["regions"].values()
            if region["is_visible"]
        )
        
        if visible_regions < self.hud_occlusion["min_visible_regions"]:
            return False
            
        # Check if we have at least one critical pair visible
        for pair in self.hud_occlusion["critical_pairs"]:
            pair_visible = all(
                any(element in region["elements"] and region["is_visible"]
                    for region in self.hud_occlusion["regions"].values())
                for element in pair
            )
            if pair_visible:
                return True
                
        return False

    def _reset_region_confidences(self) -> None:
        """Reset confidence values for all HUD regions."""
        for region in self.hud_occlusion["regions"].values():
            region["confidence"] = 1.0

    def update_hud_state(self, detections: Dict[str, Any]) -> None:
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
                "territory": self.get_territory_state(detections)
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
                    self.game_state["last_valid_possession"] = self.hud_state["last_valid_state"]["possession"]
                    self.game_state["last_valid_territory"] = self.hud_state["last_valid_state"]["territory"]
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
        return (self.hud_state["is_visible"] or 
                self.hud_state["frames_since_visible"] <= self.hud_state["max_frames_without_hud"])

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

    def get_persistence_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of state persistence performance metrics.
        
        Returns:
            Dictionary containing summarized metrics:
            - Recovery success rates for each gap type
            - Average recovery time
            - Average confidence
            - State change validation rate
            - Game interruption statistics
        """
        total_gaps = (self.persistence_metrics["short_gaps_recovered"] + 
                     self.persistence_metrics["medium_gaps_recovered"] + 
                     self.persistence_metrics["game_interruptions"]["unknown"])
        
        # Calculate recovery success rates
        short_gap_rate = (self.persistence_metrics["short_gaps_recovered"] / total_gaps 
                         if total_gaps > 0 else 0.0)
        medium_gap_rate = (self.persistence_metrics["medium_gaps_recovered"] / total_gaps 
                         if total_gaps > 0 else 0.0)
        game_interruption_rate = (self.persistence_metrics["game_interruptions"]["unknown"] / total_gaps 
                               if total_gaps > 0 else 0.0)
        
        # Calculate average recovery time (in frames)
        avg_recovery_time = (sum(self.persistence_metrics["recovery_times"]) / 
                           len(self.persistence_metrics["recovery_times"])
                           if self.persistence_metrics["recovery_times"] else 0.0)
        
        # Calculate average confidence
        avg_confidence = (sum(self.persistence_metrics["confidence_history"]) / 
                        len(self.persistence_metrics["confidence_history"])
                        if self.persistence_metrics["confidence_history"] else 0.0)
        
        # Calculate state change validation rate
        state_changes = self.persistence_metrics["state_changes"]
        validation_rate = (state_changes["validated"] / state_changes["total"] 
                         if state_changes["total"] > 0 else 0.0)
        
        return {
            "gap_recovery_rates": {
                "short_gaps": short_gap_rate,
                "medium_gaps": medium_gap_rate,
                "game_interruptions": game_interruption_rate
            },
            "recovery_stats": {
                "average_frames_to_recover": avg_recovery_time,
                "average_confidence": avg_confidence
            },
            "state_changes": {
                "validation_rate": validation_rate,
                "total_changes": state_changes["total"],
                "validated_changes": state_changes["validated"],
                "rejected_changes": state_changes["rejected"]
            },
            "gap_counts": {
                "short_gaps_recovered": self.persistence_metrics["short_gaps_recovered"],
                "medium_gaps_recovered": self.persistence_metrics["medium_gaps_recovered"],
                "game_interruptions_detected": self.persistence_metrics["game_interruptions"]["unknown"]
            }
        } 

    def validate_triangle(self, contour: np.ndarray, validation: TriangleValidation) -> Tuple[bool, float]:
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
                    pt2 = contour[(i+1)%3][0]
                    pt3 = contour[(i+2)%3][0]
                    
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
        if old_direction == 'left' and new_direction == 'right':
            event_type = "possession_change_to_right"
            description = "Right team gained possession"
        elif old_direction == 'right' and new_direction == 'left':
            event_type = "possession_change_to_left"
            description = "Left team gained possession"
        else:
            return  # No actual change
        
        # This is a key moment - should trigger clip generation
        self._trigger_key_moment({
            "type": event_type,
            "description": description,
            "timestamp": time.time(),
            "old_possession": old_direction,
            "new_possession": new_direction,
            "clip_worthy": True,
            "priority": "high"
        })

    def _handle_territory_change(self, old_direction: str, new_direction: str) -> None:
        """
        Handle territory triangle flip - indicates crossing midfield.
        
        Args:
            old_direction: Previous territory direction ('up' or 'down')
            new_direction: New territory direction ('up' or 'down')
        """
        logger.info(f"🗺️ TERRITORY CHANGE: {old_direction} → {new_direction}")
        
        # Determine field position change
        if old_direction == 'down' and new_direction == 'up':
            event_type = "entered_opponent_territory"
            description = "Crossed into opponent's territory"
        elif old_direction == 'up' and new_direction == 'down':
            event_type = "entered_own_territory"
            description = "Crossed back into own territory"
        else:
            return  # No actual change
        
        # Territory changes are important for field position analysis
        self._trigger_key_moment({
            "type": event_type,
            "description": description,
            "timestamp": time.time(),
            "old_territory": old_direction,
            "new_territory": new_direction,
            "clip_worthy": False,  # Usually not clip-worthy by itself
            "priority": "medium"
        })

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
        
        if possession_direction == 'left':
            return "away_team"  # Left team (away) has the ball
        elif possession_direction == 'right':
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
        
        if territory_direction == 'up':
            return "opponent_territory"  # Good field position
        elif territory_direction == 'down':
            return "own_territory"  # Poor field position
        else:
            return "unknown"

    def _trigger_key_moment(self, moment_data: Dict[str, Any]) -> None:
        """
        Trigger a key moment event for potential clip generation.
        
        Args:
            moment_data: Dictionary containing moment information
        """
        # Add to key moments queue for processing
        if not hasattr(self, 'key_moments'):
            self.key_moments = []
        
        self.key_moments.append(moment_data)
        
        # Log the key moment
        logger.info(f"🎯 KEY MOMENT: {moment_data['description']}")
        
        # If this is clip-worthy, trigger clip detection
        if moment_data.get('clip_worthy', False):
            self._queue_clip_generation(moment_data)

    def _queue_clip_generation(self, moment_data: Dict[str, Any]) -> None:
        """
        Queue a clip for generation based on a key moment.
        
        Args:
            moment_data: Key moment data containing clip information
        """
        # Add to clip generation queue
        if not hasattr(self, 'clip_queue'):
            self.clip_queue = []
        
        clip_data = {
            "trigger_event": moment_data['type'],
            "description": moment_data['description'],
            "timestamp": moment_data['timestamp'],
            "priority": moment_data['priority'],
            "pre_buffer_seconds": 3.0,  # 3 seconds before event
            "post_buffer_seconds": 2.0,  # 2 seconds after event
            "metadata": {
                "possession_change": moment_data.get('old_possession') != moment_data.get('new_possession'),
                "territory_change": moment_data.get('old_territory') != moment_data.get('new_territory')
            }
        }
        
        self.clip_queue.append(clip_data)
        logger.info(f"📹 QUEUED CLIP: {clip_data['description']}")

    def get_triangle_state_summary(self) -> Dict[str, Any]:
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
                "meaning": self._get_possession_meaning(possession_info.get("direction"))
            },
            "territory": {
                "direction": territory_info.get("direction", "unknown"),
                "field_context": territory_info.get("field_context", "unknown"),
                "confidence": territory_info.get("confidence", 0.0),
                "meaning": self._get_territory_meaning(territory_info.get("direction"))
            },
            "game_situation": self._analyze_combined_triangle_state(possession_info, territory_info)
        }

    def _get_possession_meaning(self, direction: str) -> str:
        """Get human-readable meaning of possession triangle direction."""
        if direction == 'left':
            return "Away team (left side) has possession"
        elif direction == 'right':
            return "Home team (right side) has possession"
        else:
            return "Possession unclear"

    def _get_territory_meaning(self, direction: str) -> str:
        """Get human-readable meaning of territory triangle direction."""
        if direction == 'up':
            return "In opponent's territory (good field position)"
        elif direction == 'down':
            return "In own territory (poor field position)"
        else:
            return "Field position unclear"

    def _analyze_combined_triangle_state(self, possession: Dict, territory: Dict) -> str:
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
        if poss_dir == 'left' and terr_dir == 'up':
            return "Away team has ball in opponent territory (scoring opportunity)"
        elif poss_dir == 'left' and terr_dir == 'down':
            return "Away team has ball in own territory (defensive position)"
        elif poss_dir == 'right' and terr_dir == 'up':
            return "Home team has ball in opponent territory (scoring opportunity)"
        elif poss_dir == 'right' and terr_dir == 'down':
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
        
        return context
    
    def _classify_situation_type(self, game_state: GameState, context: SituationContext) -> str:
        """Classify the type of situation based on game state."""
        
        # Field position based situations
        if game_state.yard_line:
            if context.possession_team == "user":
                if context.territory == "opponent" and game_state.yard_line <= 20:
                    return "red_zone_offense"
                elif context.territory == "opponent" and game_state.yard_line <= 5:
                    return "goal_line_offense"
                elif context.territory == "own" and game_state.yard_line <= 15:
                    return "backed_up_offense"
                elif context.territory == "opponent":
                    return "scoring_position"
            else:  # opponent possession
                if context.territory == "own" and game_state.yard_line <= 20:
                    return "red_zone_defense"
                elif context.territory == "own" and game_state.yard_line <= 5:
                    return "goal_line_defense"
                elif context.territory == "opponent" and game_state.yard_line <= 15:
                    return "pressure_defense"
        
        # Down and distance situations
        if game_state.down == 3:
            if game_state.distance and game_state.distance >= 7:
                return "third_and_long"
            elif game_state.distance and game_state.distance <= 3:
                return "third_and_short"
            else:
                return "third_down"
        elif game_state.down == 4:
            return "fourth_down"
        
        # Time-based situations
        if game_state.quarter and game_state.quarter >= 4:
            if game_state.time and self._is_two_minute_drill(game_state.time):
                return "two_minute_drill"
            else:
                return "fourth_quarter"
        
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
        
        # Situation type modifiers
        situation_modifiers = {
            "red_zone_offense": 0.3,
            "goal_line_offense": 0.4,
            "red_zone_defense": 0.3,
            "goal_line_defense": 0.4,
            "third_and_long": 0.2,
            "fourth_down": 0.3,
            "two_minute_drill": 0.4,
            "backed_up_offense": 0.2,
            "pressure_defense": 0.2
        }
        
        base_leverage += situation_modifiers.get(context.situation_type, 0.0)
        
        # Pressure level modifiers
        pressure_modifiers = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.0
        }
        
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
            "leverage_index": context.leverage_index
        }
        
        # Add to appropriate tracking category
        if context.possession_team == "user":
            self.performance_stats["offensive_situations"][context.situation_type].append(situation_data)
        elif context.possession_team == "opponent":
            self.performance_stats["defensive_situations"][context.situation_type].append(situation_data)
        
        # Add to general situation history
        self.situation_history.append(situation_data)
        
        # Keep only last 100 situations for memory management
        if len(self.situation_history) > 100:
            self.situation_history.pop(0)
    
    def track_possession_change(self, old_possession: str, new_possession: str, game_state: GameState) -> None:
        """Track possession changes for turnover analysis."""
        
        change_data = {
            "timestamp": time.time(),
            "old_possession": old_possession,
            "new_possession": new_possession,
            "game_state": game_state,
            "field_position": game_state.yard_line,
            "territory": game_state.territory,
            "down": game_state.down,
            "distance": game_state.distance
        }
        
        self.drive_tracking["possession_changes"].append(change_data)
        self.performance_stats["transition_moments"].append(change_data)
        
        # Update hidden MMR based on turnover context
        self._update_mmr_for_turnover(change_data)
    
    def _update_mmr_for_turnover(self, change_data: Dict[str, Any]) -> None:
        """Update hidden MMR metrics based on turnover context."""
        
        # Determine if this was user gaining or losing possession
        user_gained_possession = (
            change_data["new_possession"] == self.user_team and 
            change_data["old_possession"] != self.user_team
        )
        
        user_lost_possession = (
            change_data["old_possession"] == self.user_team and 
            change_data["new_possession"] != self.user_team
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
    
    def get_hidden_performance_summary(self) -> Dict[str, Any]:
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
                "clutch_factor": self.hidden_mmr.clutch_factor
            },
            "situation_counts": {
                "offensive_situations": len(self.performance_stats["offensive_situations"]),
                "defensive_situations": len(self.performance_stats["defensive_situations"]),
                "transition_moments": len(self.performance_stats["transition_moments"])
            },
            "analysis_context": self.analysis_context
        }
    
    def set_user_context(self, user_team: str, analysis_type: str = "self") -> None:
        """Set user context for proper possession/performance tracking."""
        self.user_team = user_team  # "home" or "away"
        self.analysis_context = analysis_type  # "self", "opponent", "pro_study"
        
        logger.info(f"User context set: team={user_team}, analysis={analysis_type}")
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking for new analysis session."""
        self.hidden_mmr = HiddenMMRMetrics()
        self.situation_history = []
        self.performance_stats = {
            "offensive_situations": defaultdict(list),
            "defensive_situations": defaultdict(list),
            "transition_moments": [],
            "decision_quality": []
        }
        
        logger.info("Performance tracking reset for new session") 
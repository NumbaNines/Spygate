"""
Context-Aware Triangle Detector for SpygateAI

This module combines super loose geometric detection with intelligent game context
validation to accurately detect possession and territory triangles while filtering
out false positives using scores, yard lines, and game state information.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path

from .enhanced_ocr import EnhancedOCR
from .triangle_orientation_detector import TriangleOrientationDetector, TriangleType, Direction


class GameContext(Enum):
    """Game context states that affect triangle validation."""
    POSSESSION_CHANGE = "possession_change"
    TERRITORY_CHANGE = "territory_change"
    SCORE_UPDATE = "score_update"
    DOWN_CHANGE = "down_change"
    NORMAL_PLAY = "normal_play"


@dataclass
class GameState:
    """Current game state information."""
    away_team: Optional[str] = None
    home_team: Optional[str] = None
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    down: Optional[int] = None
    distance: Optional[int] = None
    yard_line: Optional[str] = None  # e.g., "A35", "H22"
    quarter: Optional[int] = None
    time_remaining: Optional[str] = None
    possession_team: Optional[str] = None  # "away" or "home"
    field_territory: Optional[str] = None  # "own" or "opponent"


@dataclass
class TriangleCandidate:
    """A potential triangle with context information."""
    contour: np.ndarray
    area: float
    center: Tuple[int, int]
    bounding_rect: Tuple[int, int, int, int]
    triangle_type: TriangleType
    confidence: float = 0.0
    direction: Optional[Direction] = None
    context_score: float = 0.0
    validation_reasons: List[str] = None
    
    def __post_init__(self):
        if self.validation_reasons is None:
            self.validation_reasons = []
    
    def __eq__(self, other):
        """Compare candidates based on center and area."""
        if not isinstance(other, TriangleCandidate):
            return False
        return (self.center == other.center and 
                abs(self.area - other.area) < 1.0 and
                self.triangle_type == other.triangle_type)
    
    def __hash__(self):
        """Make candidates hashable for set operations."""
        return hash((self.center, int(self.area), self.triangle_type))


class ContextAwareTriangleDetector:
    """
    Advanced triangle detector that uses game context to validate detections.
    
    Uses a two-stage approach:
    1. Super loose geometric detection to catch all possible triangles
    2. Context-aware validation using OCR and game state logic
    """
    
    def __init__(self, debug_output_dir: Optional[Path] = None):
        self.debug_output_dir = debug_output_dir
        if self.debug_output_dir:
            self.debug_output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.ocr = EnhancedOCR()
        self.geometric_detector = TriangleOrientationDetector(debug_output_dir)
        
        # Super loose geometric thresholds
        self.SUPER_LOOSE_THRESHOLDS = {
            'MIN_AREA': 25.0,           # Very small triangles allowed
            'MAX_AREA': 2000.0,         # Large triangles allowed
            'MIN_CONVEXITY': 0.3,       # Very loose convexity
            'MAX_VERTICES': 15,         # Allow very complex shapes
            'MIN_ASPECT': 0.2,          # Very wide range
            'MAX_ASPECT': 5.0,          # Very wide range
            'SYMMETRY_TOLERANCE': 0.9,  # Very loose symmetry
        }
        
        # Game state tracking
        self.current_game_state = GameState()
        self.previous_game_state = GameState()
        self.state_history: List[GameState] = []
        
        # Context validation weights
        self.CONTEXT_WEIGHTS = {
            'ocr_consistency': 0.3,     # OCR data supports triangle presence
            'game_logic': 0.25,         # Game state logic supports triangle
            'position_logic': 0.2,      # Triangle position makes sense
            'temporal_consistency': 0.15, # Consistent with recent detections
            'geometric_quality': 0.1,   # Basic geometric validation
        }
    
    def detect_triangles_with_context(self, frame: np.ndarray, 
                                    hud_region: Tuple[int, int, int, int]) -> List[TriangleCandidate]:
        """
        Main detection method that combines loose geometric detection with context validation.
        
        Args:
            frame: Full video frame
            hud_region: HUD bounding box (x1, y1, x2, y2)
            
        Returns:
            List of validated triangle candidates
        """
        # Extract HUD region
        x1, y1, x2, y2 = hud_region
        hud_img = frame[y1:y2, x1:x2]
        
        # Update game state from OCR
        self.update_game_state(hud_img)
        
        # Get possession and territory ROIs
        possession_roi, territory_roi = self.extract_triangle_rois(hud_img)
        
        # Super loose geometric detection
        possession_candidates = self.super_loose_detection(
            possession_roi, TriangleType.POSSESSION, (x1, y1)
        )
        territory_candidates = self.super_loose_detection(
            territory_roi, TriangleType.TERRITORY, (x1, y1)
        )
        
        all_candidates = possession_candidates + territory_candidates
        
        # Context-aware validation
        validated_candidates = []
        for candidate in all_candidates:
            context_score = self.validate_with_context(candidate, hud_img)
            candidate.context_score = context_score
            
            # Accept candidates with high context scores
            if context_score > 0.4:  # Lower threshold for acceptance
                validated_candidates.append(candidate)
        
        # Debug visualization
        if self.debug_output_dir:
            self.create_context_debug_visualization(
                hud_img, all_candidates, validated_candidates
            )
        
        return validated_candidates
    
    def super_loose_detection(self, roi_img: np.ndarray, triangle_type: TriangleType,
                            offset: Tuple[int, int]) -> List[TriangleCandidate]:
        """
        Super loose geometric detection to catch all possible triangles.
        """
        if roi_img is None or roi_img.size == 0:
            return []
        
        candidates = []
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold attempts to catch different triangle types
        thresholds = [80, 120, 160, 200]
        all_contours = []
        
        for thresh_val in thresholds:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Remove duplicate contours
        unique_contours = []
        for contour in all_contours:
            is_duplicate = False
            for existing in unique_contours:
                if cv2.matchShapes(contour, existing, cv2.CONTOURS_MATCH_I1, 0) < 0.1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_contours.append(contour)
        
        for contour in unique_contours:
            area = cv2.contourArea(contour)
            
            # Super loose area check
            if not (self.SUPER_LOOSE_THRESHOLDS['MIN_AREA'] <= area <= self.SUPER_LOOSE_THRESHOLDS['MAX_AREA']):
                continue
            
            # Basic shape analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                continue
            
            convexity = area / hull_area
            
            # Super loose convexity check
            if convexity < self.SUPER_LOOSE_THRESHOLDS['MIN_CONVEXITY']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Super loose aspect ratio check
            if not (self.SUPER_LOOSE_THRESHOLDS['MIN_ASPECT'] <= aspect_ratio <= self.SUPER_LOOSE_THRESHOLDS['MAX_ASPECT']):
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx + offset[0], cy + offset[1])
            else:
                center = (x + w//2 + offset[0], y + h//2 + offset[1])
            
            # Create candidate
            candidate = TriangleCandidate(
                contour=contour,
                area=area,
                center=center,
                bounding_rect=(x + offset[0], y + offset[1], w, h),
                triangle_type=triangle_type,
                confidence=convexity,  # Initial confidence based on convexity
                validation_reasons=["Super loose geometric detection passed"]
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def update_game_state(self, hud_img: np.ndarray):
        """
        Update current game state using OCR analysis of HUD.
        """
        # Store previous state
        self.previous_game_state = GameState(
            away_team=self.current_game_state.away_team,
            home_team=self.current_game_state.home_team,
            away_score=self.current_game_state.away_score,
            home_score=self.current_game_state.home_score,
            down=self.current_game_state.down,
            distance=self.current_game_state.distance,
            yard_line=self.current_game_state.yard_line,
            possession_team=self.current_game_state.possession_team,
            field_territory=self.current_game_state.field_territory
        )
        
        # Extract text from HUD
        try:
            ocr_results = self.ocr.process_hud_region(hud_img)
            # Convert to list format expected by parsing methods
            ocr_list = []
            for key, value in ocr_results.items():
                if isinstance(value, dict) and 'text' in value:
                    ocr_list.append({
                        'text': str(value['text']),
                        'confidence': value.get('confidence', 0.5),
                        'bbox': [0, 0, 0, 0]  # Default bbox
                    })
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            ocr_list = []
        
        # Parse game information
        self.parse_scores(ocr_list)
        self.parse_down_distance(ocr_list)
        self.parse_yard_line(ocr_list)
        self.parse_team_names(ocr_list)
        
        # Add to history
        self.state_history.append(GameState(**self.current_game_state.__dict__))
        
        # Keep only recent history
        if len(self.state_history) > 10:
            self.state_history.pop(0)
    
    def parse_scores(self, ocr_results: List[Dict[str, Any]]):
        """Parse team scores from OCR results."""
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            # Look for score patterns (numbers)
            if text.isdigit() and len(text) <= 2:
                score = int(text)
                
                # Determine if this is away or home score based on position
                bbox = result.get('bbox', [0, 0, 0, 0])
                x = bbox[0] if len(bbox) > 0 else 0
                
                # Left side is typically away team
                if x < 200:  # Adjust based on your HUD layout
                    self.current_game_state.away_score = score
                else:
                    self.current_game_state.home_score = score
    
    def parse_down_distance(self, ocr_results: List[Dict[str, Any]]):
        """Parse down and distance from OCR results."""
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            # Look for down/distance patterns like "1st & 10", "3rd & 7"
            down_distance_pattern = r'(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)'
            match = re.search(down_distance_pattern, text, re.IGNORECASE)
            
            if match:
                self.current_game_state.down = int(match.group(1))
                self.current_game_state.distance = int(match.group(2))
                break
    
    def parse_yard_line(self, ocr_results: List[Dict[str, Any]]):
        """Parse yard line from OCR results."""
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            # Look for yard line patterns like "A35", "H22", "50"
            yard_pattern = r'([AH])(\d{1,2})|(\d{1,2})'
            match = re.search(yard_pattern, text)
            
            if match:
                self.current_game_state.yard_line = text
                
                # Determine field territory
                if match.group(1):  # Has team prefix
                    # A35 means away team's 35-yard line (in own territory)
                    # H35 means home team's 35-yard line (in own territory)
                    self.current_game_state.field_territory = "own"
                else:
                    # Numbers without prefix near midfield
                    yard_num = int(match.group(3))
                    if 40 <= yard_num <= 50:
                        self.current_game_state.field_territory = "neutral"
                break
    
    def parse_team_names(self, ocr_results: List[Dict[str, Any]]):
        """Parse team abbreviations from OCR results."""
        for result in ocr_results:
            text = result.get('text', '').strip()
            
            # Look for 3-letter team abbreviations
            if len(text) == 3 and text.isalpha():
                bbox = result.get('bbox', [0, 0, 0, 0])
                x = bbox[0] if len(bbox) > 0 else 0
                
                # Left side is typically away team
                if x < 200:
                    self.current_game_state.away_team = text.upper()
                else:
                    self.current_game_state.home_team = text.upper()
    
    def validate_with_context(self, candidate: TriangleCandidate, hud_img: np.ndarray) -> float:
        """
        Validate triangle candidate using game context.
        
        Returns:
            Context score (0.0 to 1.0) indicating confidence
        """
        scores = {}
        
        # 1. OCR Consistency Check
        scores['ocr_consistency'] = self.check_ocr_consistency(candidate, hud_img)
        
        # 2. Game Logic Check
        scores['game_logic'] = self.check_game_logic(candidate)
        
        # 3. Position Logic Check
        scores['position_logic'] = self.check_position_logic(candidate)
        
        # 4. Temporal Consistency Check
        scores['temporal_consistency'] = self.check_temporal_consistency(candidate)
        
        # 5. Geometric Quality Check
        scores['geometric_quality'] = self.check_geometric_quality(candidate)
        
        # Calculate weighted score
        total_score = sum(
            scores[key] * self.CONTEXT_WEIGHTS[key] 
            for key in scores
        )
        
        # Add detailed validation reasons
        for key, score in scores.items():
            if score > 0.7:
                candidate.validation_reasons.append(f"Strong {key}: {score:.2f}")
            elif score > 0.4:
                candidate.validation_reasons.append(f"Moderate {key}: {score:.2f}")
            else:
                candidate.validation_reasons.append(f"Weak {key}: {score:.2f}")
        
        return total_score
    
    def check_ocr_consistency(self, candidate: TriangleCandidate, hud_img: np.ndarray) -> float:
        """Check if OCR data supports the presence of this triangle."""
        # For possession triangles, check if we have team names and scores
        if candidate.triangle_type == TriangleType.POSSESSION:
            score = 0.5  # Base score
            
            # Boost score if we have team data
            if self.current_game_state.away_team and self.current_game_state.home_team:
                score += 0.2
            
            # Boost score if we have score data
            if (self.current_game_state.away_score is not None and 
                self.current_game_state.home_score is not None):
                score += 0.2
            
            return min(score, 1.0)
        
        # For territory triangles, check if we have yard line info
        elif candidate.triangle_type == TriangleType.TERRITORY:
            if self.current_game_state.yard_line:
                return 0.8  # Strong OCR support
            else:
                return 0.5  # Moderate support (triangles can exist without OCR)
        
        return 0.5  # Default
    
    def check_game_logic(self, candidate: TriangleCandidate) -> float:
        """Check if triangle makes sense given current game state."""
        # Always give moderate score - triangles should be present in most game states
        base_score = 0.6
        
        # Check for game state changes that would trigger triangle updates
        if self.previous_game_state:
            # Score change suggests possession triangle update
            if (candidate.triangle_type == TriangleType.POSSESSION and
                (self.current_game_state.away_score != self.previous_game_state.away_score or
                 self.current_game_state.home_score != self.previous_game_state.home_score)):
                return 0.9
            
            # Yard line change suggests territory triangle update
            if (candidate.triangle_type == TriangleType.TERRITORY and
                self.current_game_state.yard_line != self.previous_game_state.yard_line):
                return 0.9
        
        return base_score
    
    def check_position_logic(self, candidate: TriangleCandidate) -> float:
        """Check if triangle position makes logical sense."""
        x, y, w, h = candidate.bounding_rect
        
        if candidate.triangle_type == TriangleType.POSSESSION:
            # Possession triangles should be in the left-center area of HUD
            if 50 <= x <= 400:  # More lenient range
                return 0.8
            else:
                return 0.4  # Still possible, just less likely
        
        elif candidate.triangle_type == TriangleType.TERRITORY:
            # Territory triangles should be in the right area of HUD
            if x >= 300:  # More lenient range
                return 0.8
            else:
                return 0.4  # Still possible, just less likely
        
        return 0.5
    
    def check_temporal_consistency(self, candidate: TriangleCandidate) -> float:
        """Check consistency with recent triangle detections."""
        # For now, give moderate score to all candidates
        # This could be enhanced with actual temporal tracking
        return 0.6
    
    def check_geometric_quality(self, candidate: TriangleCandidate) -> float:
        """Basic geometric quality check."""
        # Use basic geometric properties
        area = candidate.area
        x, y, w, h = candidate.bounding_rect
        aspect_ratio = w / h if h > 0 else 0
        
        # Score based on area (prefer medium-sized triangles)
        area_score = 0.5
        if 50 <= area <= 500:
            area_score = 0.8
        elif 25 <= area <= 1000:
            area_score = 0.6
        
        # Score based on aspect ratio (prefer reasonable ratios)
        aspect_score = 0.5
        if 0.5 <= aspect_ratio <= 2.0:
            aspect_score = 0.8
        elif 0.3 <= aspect_ratio <= 3.0:
            aspect_score = 0.6
        
        return (area_score + aspect_score) / 2
    
    def extract_triangle_rois(self, hud_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ROI regions for possession and territory triangles."""
        h, w = hud_img.shape[:2]
        
        # Possession triangle ROI (left side)
        possession_roi = hud_img[:, :w//2]
        
        # Territory triangle ROI (right side)
        territory_roi = hud_img[:, w//2:]
        
        return possession_roi, territory_roi
    
    def create_context_debug_visualization(self, hud_img: np.ndarray,
                                         all_candidates: List[TriangleCandidate],
                                         validated_candidates: List[TriangleCandidate]):
        """Create debug visualization showing context validation results."""
        if not self.debug_output_dir:
            return
        
        # Create visualization canvas
        vis_height = max(hud_img.shape[0], 600)
        vis_width = hud_img.shape[1] + 600
        vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Place HUD image
        vis_img[:hud_img.shape[0], :hud_img.shape[1]] = hud_img
        
        # Draw all candidates
        for i, candidate in enumerate(all_candidates):
            color = (0, 255, 0) if candidate in validated_candidates else (0, 0, 255)
            cv2.drawContours(vis_img, [candidate.contour], -1, color, 2)
            
            # Add context score
            x, y = candidate.center
            cv2.putText(vis_img, f"{candidate.context_score:.2f}", 
                       (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add candidate number
            cv2.putText(vis_img, str(i+1), (x-10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add game state information
        text_x = hud_img.shape[1] + 10
        y_offset = 30
        
        cv2.putText(vis_img, "GAME STATE:", (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        state_info = [
            f"Away: {self.current_game_state.away_team} ({self.current_game_state.away_score})",
            f"Home: {self.current_game_state.home_team} ({self.current_game_state.home_score})",
            f"Down: {self.current_game_state.down} & {self.current_game_state.distance}",
            f"Yard Line: {self.current_game_state.yard_line}",
            f"Territory: {self.current_game_state.field_territory}"
        ]
        
        for info in state_info:
            cv2.putText(vis_img, info, (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
        
        y_offset += 20
        cv2.putText(vis_img, f"CANDIDATES: {len(all_candidates)}", (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(vis_img, f"VALIDATED: {len(validated_candidates)}", (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization
        output_path = self.debug_output_dir / "context_validation_debug.jpg"
        cv2.imwrite(str(output_path), vis_img)
        print(f"Context debug visualization saved: {output_path}") 
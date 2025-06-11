#!/usr/bin/env python3
"""
SpygateAI Game Situation Analysis Engine
Combines triangle detection with HUD OCR to extract structured game intelligence
"""

import cv2
import numpy as np
import easyocr
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
import time

from hybrid_triangle_detector_final import FinalHybridTriangleDetector

@dataclass
class GameSituation:
    """Structured representation of a complete game situation."""
    
    # Core Game State
    game_version: str = "madden_25"  # Auto-detected game type
    timestamp: str = ""  # When this situation was captured
    
    # Score Information
    away_team: str = ""
    home_team: str = ""
    away_score: int = 0
    home_score: int = 0
    
    # Game Clock & Quarter
    game_clock: str = ""
    quarter: int = 0
    
    # Down & Distance
    down: int = 0
    distance: int = 0
    yard_line: int = 0  # Field position (1-50)
    
    # Possession & Territory
    possession_team: str = ""  # "away" or "home"
    field_territory: str = ""  # "own" or "opponent"
    possession_confidence: float = 0.0
    territory_confidence: float = 0.0
    
    # Context
    play_clock: str = ""
    timeouts_away: int = 3
    timeouts_home: int = 3
    
    # Situation Analysis
    situation_type: str = ""  # "red_zone", "goal_line", "mid_field", etc.
    pressure_level: str = ""  # "high", "medium", "low"
    strategic_context: str = ""  # "two_minute_drill", "garbage_time", etc.
    
    # Detection Metadata
    regions_detected: int = 0
    triangles_detected: int = 0
    ocr_confidence: float = 0.0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if this represents a valid game situation."""
        return (
            self.down > 0 and 
            self.distance >= 0 and 
            self.possession_team in ["away", "home"] and
            self.field_territory in ["own", "opponent"]
        )

class GameSituationAnalyzer:
    """
    Complete game situation analysis engine for SpygateAI.
    Combines triangle detection with OCR to extract structured game intelligence.
    """
    
    def __init__(self, gpu_enabled: bool = True):
        """Initialize the game situation analyzer."""
        self.triangle_detector = FinalHybridTriangleDetector()
        
        # Initialize OCR engine
        print("🔧 Initializing OCR engine...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=gpu_enabled)
        print("✅ OCR engine ready")
        
        # HUD text patterns for Madden 25
        self.patterns = {
            'down_distance': re.compile(r'(\d+)(?:st|nd|rd|th)\s*&\s*(\d+)', re.IGNORECASE),
            'score': re.compile(r'(\d+)\s*[-:]\s*(\d+)'),
            'time': re.compile(r'(\d{1,2}):(\d{2})'),
            'quarter': re.compile(r'(\d+)(?:st|nd|rd|th)\s*(?:quarter|qtr|Q)', re.IGNORECASE),
            'yard_line': re.compile(r'(\d+)\s*(?:YD|YARD)', re.IGNORECASE),
            'team_names': re.compile(r'([A-Z]{2,4})\s*(?:VS|V)\s*([A-Z]{2,4})', re.IGNORECASE)
        }
    
    def analyze_frame(self, frame: np.ndarray, frame_timestamp: Optional[str] = None) -> GameSituation:
        """
        Analyze a single frame to extract complete game situation.
        
        Args:
            frame: Input video frame
            frame_timestamp: Optional timestamp for this frame
            
        Returns:
            GameSituation object with extracted intelligence
        """
        start_time = time.time()
        
        # Initialize game situation
        situation = GameSituation()
        situation.timestamp = frame_timestamp or datetime.now().isoformat()
        
        # Step 1: Triangle Detection (possession/territory)
        triangle_results = self.triangle_detector.process_frame(frame)
        situation.regions_detected = triangle_results['regions_detected']
        situation.triangles_detected = sum(tri['triangles_found'] for tri in triangle_results['triangle_results'])
        
        # Step 2: Extract possession from triangles
        self._extract_possession_from_triangles(triangle_results, situation)
        
        # Step 3: HUD Text Recognition
        self._extract_hud_text(frame, triangle_results, situation)
        
        # Step 4: Situational Analysis
        self._analyze_situation_context(situation)
        
        # Finalize
        situation.processing_time_ms = (time.time() - start_time) * 1000
        
        return situation
    
    def _extract_possession_from_triangles(self, triangle_results: Dict, situation: GameSituation) -> None:
        """Extract possession and territory information from triangle detection."""
        
        for tri_result in triangle_results['triangle_results']:
            region_type = tri_result['region_info']['class_name']
            triangles_found = tri_result['triangles_found']
            
            if triangles_found > 0:
                direction = tri_result.get('triangle_direction', 'unknown')
                confidence = tri_result['region_info']['confidence']
                
                # Possession triangle (◀▶ arrows)
                if region_type == 'possession_triangle_area':
                    if direction == 'left':
                        situation.possession_team = "away"  # Left team has possession
                    elif direction == 'right':
                        situation.possession_team = "home"   # Right team has possession
                    situation.possession_confidence = confidence
                
                # Territory triangle (▲▼ field position)
                elif region_type == 'territory_triangle_area':
                    if direction == 'up':
                        situation.field_territory = "opponent"  # ▲ = in opponent territory
                    elif direction == 'down':
                        situation.field_territory = "own"       # ▼ = in own territory
                    situation.territory_confidence = confidence
    
    def _extract_hud_text(self, frame: np.ndarray, triangle_results: Dict, situation: GameSituation) -> None:
        """Extract text information from HUD regions using OCR."""
        
        hud_regions = [r for r in triangle_results['regions'] if r['class_name'] == 'hud']
        
        if not hud_regions:
            return
        
        # Get the main HUD region for OCR
        main_hud = max(hud_regions, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = main_hud['bbox']
        
        # Extract HUD region with padding
        padding = 20
        hud_crop = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                        max(0, x1-padding):min(frame.shape[1], x2+padding)]
        
        # Preprocess for better OCR
        hud_gray = cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY)
        hud_contrast = cv2.convertScaleAbs(hud_gray, alpha=2.0, beta=0)
        
        # Perform OCR
        try:
            ocr_results = self.ocr_reader.readtext(hud_contrast)
            
            # Combine all detected text
            all_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
            
            # Calculate average OCR confidence
            confidences = [result[2] for result in ocr_results if result[2] > 0.3]
            situation.ocr_confidence = np.mean(confidences) if confidences else 0.0
            
            # Extract specific information
            self._parse_hud_text(all_text, situation)
            
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            situation.ocr_confidence = 0.0
    
    def _parse_hud_text(self, text: str, situation: GameSituation) -> None:
        """Parse OCR text to extract specific game information."""
        
        # Down & Distance (e.g., "3rd & 8")
        down_match = self.patterns['down_distance'].search(text)
        if down_match:
            situation.down = int(down_match.group(1))
            situation.distance = int(down_match.group(2))
        
        # Score (e.g., "14-7" or "14:7")
        score_match = self.patterns['score'].search(text)
        if score_match:
            situation.away_score = int(score_match.group(1))
            situation.home_score = int(score_match.group(2))
        
        # Game Clock (e.g., "12:34")
        time_match = self.patterns['time'].search(text)
        if time_match:
            situation.game_clock = f"{time_match.group(1)}:{time_match.group(2)}"
        
        # Quarter (e.g., "3rd Quarter")
        quarter_match = self.patterns['quarter'].search(text)
        if quarter_match:
            situation.quarter = int(quarter_match.group(1))
        
        # Yard Line (e.g., "25 YD")
        yard_match = self.patterns['yard_line'].search(text)
        if yard_match:
            situation.yard_line = int(yard_match.group(1))
        
        # Team Names (e.g., "KC VS SF")
        team_match = self.patterns['team_names'].search(text)
        if team_match:
            situation.away_team = team_match.group(1).upper()
            situation.home_team = team_match.group(2).upper()
    
    def _analyze_situation_context(self, situation: GameSituation) -> None:
        """Analyze the situation to provide strategic context."""
        
        # Determine situation type based on field position
        if situation.yard_line <= 20:
            situation.situation_type = "red_zone"
        elif situation.yard_line <= 5:
            situation.situation_type = "goal_line"
        elif situation.yard_line >= 40:
            situation.situation_type = "long_field"
        else:
            situation.situation_type = "mid_field"
        
        # Determine pressure level
        if situation.down >= 3:
            situation.pressure_level = "high"
        elif situation.down == 2 and situation.distance > 7:
            situation.pressure_level = "medium"
        else:
            situation.pressure_level = "low"
        
        # Strategic context based on game clock
        if situation.game_clock:
            try:
                minutes, seconds = map(int, situation.game_clock.split(':'))
                total_seconds = minutes * 60 + seconds
                
                if situation.quarter >= 4 and total_seconds <= 120:  # Last 2 minutes
                    situation.strategic_context = "two_minute_drill"
                elif situation.quarter >= 4 and abs(situation.away_score - situation.home_score) > 14:
                    situation.strategic_context = "garbage_time"
                else:
                    situation.strategic_context = "normal_play"
            except:
                situation.strategic_context = "unknown"
    
    def analyze_video_sequence(self, video_path: str, 
                             frame_interval: int = 30,
                             max_frames: Optional[int] = None) -> List[GameSituation]:
        """
        Analyze a video to extract all game situations.
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            List of GameSituation objects
        """
        cap = cv2.VideoCapture(video_path)
        situations = []
        frame_count = 0
        processed_frames = 0
        
        print(f"🎬 Analyzing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                timestamp = f"{frame_count / cap.get(cv2.CAP_PROP_FPS):.2f}s"
                situation = self.analyze_frame(frame, timestamp)
                
                if situation.is_valid():
                    situations.append(situation)
                    print(f"✅ Frame {frame_count}: {situation.down} & {situation.distance}, {situation.possession_team} possession")
                
                processed_frames += 1
                if max_frames and processed_frames >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"🎉 Analysis complete: {len(situations)} valid game situations found")
        return situations
    
    def save_situations_to_json(self, situations: List[GameSituation], output_path: str) -> None:
        """Save game situations to JSON file."""
        data = {
            'analysis_metadata': {
                'total_situations': len(situations),
                'analysis_timestamp': datetime.now().isoformat(),
                'spygate_version': '6.8'
            },
            'game_situations': [situation.to_dict() for situation in situations]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(situations)} game situations to: {output_path}")

def test_analyzer_on_images():
    """Test the analyzer on our fresh test images."""
    analyzer = GameSituationAnalyzer()
    
    # Test on a few of our successful images
    test_images = [
        "fresh_test_2_monitor3_screenshot_20250611_031353_61.png",
        "fresh_test_6_monitor3_screenshot_20250611_032339_26.png",
        "fresh_test_9_monitor3_screenshot_20250611_034444_279.png"
    ]
    
    situations = []
    
    print("🔍 Testing Game Situation Analysis on fresh images...")
    print("="*60)
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n📸 Analyzing: {img_path}")
            
            frame = cv2.imread(img_path)
            situation = analyzer.analyze_frame(frame, img_path)
            
            print(f"   🎯 Game State:")
            print(f"      - Down & Distance: {situation.down} & {situation.distance}")
            print(f"      - Possession: {situation.possession_team} (conf: {situation.possession_confidence:.3f})")
            print(f"      - Territory: {situation.field_territory} (conf: {situation.territory_confidence:.3f})")
            print(f"      - Score: {situation.away_score}-{situation.home_score}")
            print(f"      - Quarter: {situation.quarter}, Clock: {situation.game_clock}")
            print(f"      - Situation: {situation.situation_type} ({situation.pressure_level} pressure)")
            print(f"      - Context: {situation.strategic_context}")
            print(f"      - OCR Confidence: {situation.ocr_confidence:.3f}")
            print(f"      - Processing Time: {situation.processing_time_ms:.1f}ms")
            print(f"      - Valid: {situation.is_valid()}")
            
            situations.append(situation)
    
    # Save results
    if situations:
        analyzer.save_situations_to_json(situations, "test_game_situations.json")
    
    print("\n🎉 Game Situation Analysis Test Complete!")
    return situations

if __name__ == "__main__":
    test_analyzer_on_images() 
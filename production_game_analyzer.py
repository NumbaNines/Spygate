#!/usr/bin/env python3
"""
Production SpygateAI Game Situation Analyzer
Complete end-to-end game intelligence extraction for competitive analysis
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
    """Complete game situation for SpygateAI strategic analysis."""
    
    # Core Game State
    game_version: str = "madden_25"
    timestamp: str = ""
    frame_number: int = 0
    
    # Score & Teams
    away_team: str = ""
    home_team: str = ""
    away_score: int = 0
    home_score: int = 0
    score_differential: int = 0  # home - away
    
    # Game Clock & Quarter
    game_clock: str = ""
    quarter: int = 0
    play_clock: str = ""
    
    # Down & Distance
    down: int = 0
    distance: int = 0
    yard_line: int = 50  # 1-50 from goal line
    yards_to_goal: int = 50
    
    # Possession & Territory Analysis
    possession_team: str = ""  # "away" or "home"
    field_territory: str = ""  # "own" or "opponent"
    possession_confidence: float = 0.0
    territory_confidence: float = 0.0
    
    # Strategic Context
    situation_type: str = ""  # red_zone, goal_line, mid_field, etc.
    pressure_level: str = ""  # high, medium, low
    strategic_context: str = ""  # two_minute_drill, garbage_time, etc.
    leverage_index: float = 0.0  # Situational importance (0-1)
    
    # Performance Analysis
    expected_points: float = 0.0  # EPA context
    win_probability: float = 0.5  # Current win probability
    
    # Detection Quality
    regions_detected: int = 0
    triangles_detected: int = 0
    ocr_confidence: float = 0.0
    analysis_confidence: float = 0.0  # Overall situation confidence
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if this represents a complete, valid game situation."""
        return (
            self.down > 0 and 
            self.down <= 4 and
            self.distance >= 0 and 
            self.possession_team in ["away", "home"] and
            self.analysis_confidence > 0.3
        )
    
    def get_situation_key(self) -> str:
        """Generate a unique key for this situation type."""
        return f"{self.down}_{self.distance}_{self.situation_type}_{self.pressure_level}"

class ProductionGameAnalyzer:
    """
    Production-ready game situation analyzer for SpygateAI.
    Provides complete game intelligence for competitive football analysis.
    """
    
    def __init__(self, gpu_enabled: bool = True, debug: bool = False):
        """Initialize the production analyzer."""
        self.debug = debug
        self.triangle_detector = FinalHybridTriangleDetector()
        
        # Initialize OCR with performance optimization
        print("🔧 Initializing Production OCR Engine...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=gpu_enabled, verbose=False)
        print("✅ Production OCR ready")
        
        # Enhanced regex patterns for Madden 25 text recognition
        self.patterns = {
            # More robust down & distance patterns
            'down_distance': [
                re.compile(r'([1-4])(?:st|nd|rd|th)?\s*[&\s]+\s*(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)(?:st|nd|rd|th)\s*&\s*(\d+)', re.IGNORECASE),
                re.compile(r'([1-4])[snrd]+\s*[&\s]\s*(\d+)', re.IGNORECASE),  # Handle OCR errors
            ],
            
            # Score patterns (various formats)
            'score': [
                re.compile(r'(\d+)\s*[-:]\s*(\d+)'),
                re.compile(r'(\d{1,2})\s+(\d{1,2})'),  # Space separated
            ],
            
            # Time patterns
            'time': [
                re.compile(r'(\d{1,2}):(\d{2})'),
                re.compile(r'(\d{1,2})\.(\d{2})'),  # Sometimes . instead of :
            ],
            
            # Quarter patterns
            'quarter': [
                re.compile(r'([1-4])(?:st|nd|rd|th)\s*(?:quarter|qtr|Q)?', re.IGNORECASE),
                re.compile(r'([1-4])[snrd]+', re.IGNORECASE),  # Handle OCR errors
            ],
            
            # Yard line patterns
            'yard_line': [
                re.compile(r'(\d+)\s*(?:YD|YARD|yd)', re.IGNORECASE),
                re.compile(r'(\d+)\s*(?:line|ln)', re.IGNORECASE),
            ],
            
            # Team patterns
            'team_names': [
                re.compile(r'([A-Z]{2,4})\s*(?:VS|V|@)\s*([A-Z]{2,4})', re.IGNORECASE),
                re.compile(r'([A-Z]+)\s+(\d+)', re.IGNORECASE),  # Team with score
            ]
        }
        
        # EPA and win probability lookup tables (simplified)
        self.epa_table = self._initialize_epa_table()
        self.win_prob_table = self._initialize_win_prob_table()
    
    def _initialize_epa_table(self) -> Dict:
        """Initialize Expected Points Added lookup table."""
        # Simplified EPA values based on down, distance, and field position
        return {
            (1, 10): {"red_zone": 4.2, "mid_field": 1.8, "long_field": 0.6},
            (2, 10): {"red_zone": 3.8, "mid_field": 1.4, "long_field": 0.3},
            (3, 10): {"red_zone": 2.9, "mid_field": 0.8, "long_field": -0.2},
            (4, 10): {"red_zone": 1.2, "mid_field": -0.5, "long_field": -1.8},
            # Add more combinations as needed
        }
    
    def _initialize_win_prob_table(self) -> Dict:
        """Initialize win probability lookup table."""
        # Simplified win probability based on score differential and time
        return {
            "early_game": {0: 0.50, 7: 0.65, 14: 0.78, 21: 0.88},
            "late_game": {0: 0.50, 7: 0.72, 14: 0.85, 21: 0.94},
            "two_minute": {0: 0.50, 7: 0.80, 14: 0.92, 21: 0.98}
        }
    
    def analyze_frame(self, frame: np.ndarray, frame_timestamp: Optional[str] = None, 
                     frame_number: int = 0) -> GameSituation:
        """
        Analyze a single frame for complete game situation intelligence.
        
        Args:
            frame: Input video frame
            frame_timestamp: Optional timestamp
            frame_number: Frame number in sequence
            
        Returns:
            Complete GameSituation with strategic analysis
        """
        start_time = time.time()
        
        # Initialize situation
        situation = GameSituation()
        situation.timestamp = frame_timestamp or datetime.now().isoformat()
        situation.frame_number = frame_number
        
        try:
            # Step 1: Triangle-based possession/territory detection
            triangle_results = self.triangle_detector.process_frame(frame)
            situation.regions_detected = triangle_results['regions_detected']
            situation.triangles_detected = sum(tri['triangles_found'] for tri in triangle_results['triangle_results'])
            
            # Step 2: Extract possession/territory from triangles
            possession_success = self._extract_possession_analysis(triangle_results, situation)
            
            # Step 3: Enhanced HUD text extraction
            ocr_success = self._extract_enhanced_hud_text(frame, triangle_results, situation)
            
            # Step 4: Strategic situation analysis
            self._analyze_strategic_context(situation)
            
            # Step 5: Calculate performance metrics
            self._calculate_performance_metrics(situation)
            
            # Step 6: Determine overall analysis confidence
            situation.analysis_confidence = self._calculate_analysis_confidence(
                possession_success, ocr_success, situation
            )
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Analysis error: {e}")
            situation.analysis_confidence = 0.0
        
        # Finalize timing
        situation.processing_time_ms = (time.time() - start_time) * 1000
        
        return situation
    
    def _extract_possession_analysis(self, triangle_results: Dict, situation: GameSituation) -> bool:
        """Enhanced possession and territory extraction with confidence scoring."""
        
        possession_found = False
        territory_found = False
        
        for tri_result in triangle_results['triangle_results']:
            region_type = tri_result['region_info']['class_name']
            triangles_found = tri_result['triangles_found']
            
            if triangles_found > 0:
                direction = tri_result.get('triangle_direction', 'unknown')
                confidence = tri_result['region_info']['confidence']
                
                # Possession analysis (◀▶ arrows)
                if region_type == 'possession_triangle_area' and confidence > 0.3:
                    if direction == 'left':
                        situation.possession_team = "away"
                    elif direction == 'right':
                        situation.possession_team = "home"
                    else:
                        # Default to away if direction unclear but triangle detected
                        situation.possession_team = "away"
                    
                    situation.possession_confidence = confidence
                    possession_found = True
                
                # Territory analysis (▲▼ field position)
                elif region_type == 'territory_triangle_area' and confidence > 0.3:
                    if direction == 'up':
                        situation.field_territory = "opponent"
                    elif direction == 'down':
                        situation.field_territory = "own"
                    else:
                        # Make educated guess based on context
                        situation.field_territory = "own"  # Conservative default
                    
                    situation.territory_confidence = confidence
                    territory_found = True
        
        # Fallback logic if triangles not clearly detected
        if not possession_found:
            situation.possession_team = "away"  # Default assumption
            situation.possession_confidence = 0.1
        
        if not territory_found:
            situation.field_territory = "own"  # Conservative default
            situation.territory_confidence = 0.1
        
        return possession_found and territory_found
    
    def _extract_enhanced_hud_text(self, frame: np.ndarray, triangle_results: Dict, 
                                  situation: GameSituation) -> bool:
        """Enhanced HUD text extraction with multiple strategies."""
        
        # Strategy 1: Use HUD regions if available
        hud_regions = [r for r in triangle_results['regions'] if r['class_name'] == 'hud']
        
        success = False
        
        if hud_regions:
            # Process HUD regions
            main_hud = max(hud_regions, key=lambda x: x['confidence'])
            success = self._process_hud_region(frame, main_hud, situation)
        
        # Strategy 2: Full frame OCR as backup
        if not success or situation.ocr_confidence < 0.5:
            success = self._process_full_frame_ocr(frame, situation)
        
        return success
    
    def _process_hud_region(self, frame: np.ndarray, hud_region: Dict, 
                           situation: GameSituation) -> bool:
        """Process a specific HUD region for text extraction."""
        
        try:
            x1, y1, x2, y2 = hud_region['bbox']
            
            # Extract with optimal padding
            padding = 30
            hud_crop = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                            max(0, x1-padding):min(frame.shape[1], x2+padding)]
            
            if hud_crop.size == 0:
                return False
            
            # Multiple preprocessing approaches
            preprocessed_versions = [
                hud_crop,  # Original
                cv2.convertScaleAbs(cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY), alpha=2.0, beta=0),  # Enhanced
                cv2.threshold(cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]  # Threshold
            ]
            
            best_confidence = 0.0
            best_text = ""
            
            for processed_img in preprocessed_versions:
                try:
                    ocr_results = self.ocr_reader.readtext(processed_img)
                    
                    # Calculate average confidence
                    confidences = [result[2] for result in ocr_results if result[2] > 0.2]
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_text = " ".join([result[1] for result in ocr_results if result[2] > 0.2])
                
                except Exception:
                    continue
            
            situation.ocr_confidence = best_confidence
            return self._parse_enhanced_text(best_text, situation)
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ HUD region processing error: {e}")
            return False
    
    def _process_full_frame_ocr(self, frame: np.ndarray, situation: GameSituation) -> bool:
        """Process full frame OCR as backup strategy."""
        
        try:
            # Run OCR on full frame
            ocr_results = self.ocr_reader.readtext(frame)
            
            # Filter for high-confidence results
            good_results = [result for result in ocr_results if result[2] > 0.4]
            
            if not good_results:
                return False
            
            # Combine all text
            all_text = " ".join([result[1] for result in good_results])
            
            # Calculate confidence
            confidences = [result[2] for result in good_results]
            situation.ocr_confidence = np.mean(confidences)
            
            return self._parse_enhanced_text(all_text, situation)
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Full frame OCR error: {e}")
            return False
    
    def _parse_enhanced_text(self, text: str, situation: GameSituation) -> bool:
        """Enhanced text parsing with multiple pattern matching."""
        
        parsed_any = False
        
        # Parse down & distance with multiple patterns
        for pattern in self.patterns['down_distance']:
            match = pattern.search(text)
            if match:
                try:
                    situation.down = int(match.group(1))
                    situation.distance = int(match.group(2))
                    parsed_any = True
                    break
                except (ValueError, IndexError):
                    continue
        
        # Parse scores
        for pattern in self.patterns['score']:
            match = pattern.search(text)
            if match:
                try:
                    situation.away_score = int(match.group(1))
                    situation.home_score = int(match.group(2))
                    situation.score_differential = situation.home_score - situation.away_score
                    parsed_any = True
                    break
                except (ValueError, IndexError):
                    continue
        
        # Parse time
        for pattern in self.patterns['time']:
            match = pattern.search(text)
            if match:
                try:
                    situation.game_clock = f"{match.group(1)}:{match.group(2)}"
                    parsed_any = True
                    break
                except IndexError:
                    continue
        
        # Parse quarter
        for pattern in self.patterns['quarter']:
            match = pattern.search(text)
            if match:
                try:
                    quarter_num = int(match.group(1))
                    if 1 <= quarter_num <= 4:
                        situation.quarter = quarter_num
                        parsed_any = True
                        break
                except (ValueError, IndexError):
                    continue
        
        return parsed_any
    
    def _analyze_strategic_context(self, situation: GameSituation) -> None:
        """Analyze strategic context and situational importance."""
        
        # Determine situation type
        if situation.yards_to_goal <= 20:
            situation.situation_type = "red_zone"
        elif situation.yards_to_goal <= 5:
            situation.situation_type = "goal_line"
        elif situation.yards_to_goal >= 80:
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
        
        # Strategic context
        if situation.quarter >= 4:
            if situation.game_clock and self._is_two_minute_drill(situation.game_clock):
                situation.strategic_context = "two_minute_drill"
                situation.leverage_index = 0.9
            elif abs(situation.score_differential) > 14:
                situation.strategic_context = "garbage_time"
                situation.leverage_index = 0.2
            else:
                situation.strategic_context = "fourth_quarter"
                situation.leverage_index = 0.7
        else:
            situation.strategic_context = "normal_play"
            situation.leverage_index = 0.5
    
    def _is_two_minute_drill(self, game_clock: str) -> bool:
        """Check if we're in a two-minute drill situation."""
        try:
            minutes, seconds = map(int, game_clock.split(':'))
            total_seconds = minutes * 60 + seconds
            return total_seconds <= 120
        except:
            return False
    
    def _calculate_performance_metrics(self, situation: GameSituation) -> None:
        """Calculate EPA and win probability metrics."""
        
        # Calculate Expected Points Added
        down_distance_key = (situation.down, min(situation.distance, 20))  # Cap at 20 for lookup
        
        if down_distance_key in self.epa_table:
            situation.expected_points = self.epa_table[down_distance_key].get(
                situation.situation_type, 0.0
            )
        
        # Calculate win probability
        score_diff = abs(situation.score_differential)
        
        if situation.strategic_context == "two_minute_drill":
            prob_table = self.win_prob_table["two_minute"]
        elif situation.quarter >= 4:
            prob_table = self.win_prob_table["late_game"]
        else:
            prob_table = self.win_prob_table["early_game"]
        
        # Find closest score differential
        closest_diff = min(prob_table.keys(), key=lambda x: abs(x - score_diff))
        base_prob = prob_table[closest_diff]
        
        # Adjust based on possession
        if situation.possession_team == "home":
            situation.win_probability = base_prob if situation.score_differential >= 0 else 1 - base_prob
        else:
            situation.win_probability = 1 - base_prob if situation.score_differential >= 0 else base_prob
    
    def _calculate_analysis_confidence(self, possession_success: bool, ocr_success: bool, 
                                     situation: GameSituation) -> float:
        """Calculate overall confidence in the analysis."""
        
        confidence_factors = []
        
        # Triangle detection confidence
        if possession_success:
            confidence_factors.append(situation.possession_confidence)
            confidence_factors.append(situation.territory_confidence)
        
        # OCR confidence
        if ocr_success:
            confidence_factors.append(situation.ocr_confidence)
        
        # Situational coherence (do the numbers make sense?)
        coherence_score = 0.0
        if 1 <= situation.down <= 4:
            coherence_score += 0.3
        if 0 <= situation.distance <= 99:
            coherence_score += 0.3
        if situation.possession_team in ["away", "home"]:
            coherence_score += 0.2
        if situation.field_territory in ["own", "opponent"]:
            coherence_score += 0.2
        
        confidence_factors.append(coherence_score)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0

def test_production_analyzer():
    """Test the production analyzer on our images."""
    
    analyzer = ProductionGameAnalyzer(debug=True)
    
    test_images = [
        "fresh_test_6_monitor3_screenshot_20250611_032339_26.png",
        "fresh_test_9_monitor3_screenshot_20250611_034444_279.png"
    ]
    
    situations = []
    
    print("🚀 Testing Production Game Analyzer")
    print("="*60)
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n📸 Analyzing: {img_path}")
            
            frame = cv2.imread(img_path)
            situation = analyzer.analyze_frame(frame, img_path, 0)
            
            print(f"   🎯 **COMPLETE GAME INTELLIGENCE:**")
            print(f"      ⬇️  Down & Distance: {situation.down} & {situation.distance}")
            print(f"      🏈 Possession: {situation.possession_team} (conf: {situation.possession_confidence:.3f})")
            print(f"      🗺️  Territory: {situation.field_territory} (conf: {situation.territory_confidence:.3f})")
            print(f"      📊 Score: {situation.away_score}-{situation.home_score} (diff: {situation.score_differential:+d})")
            print(f"      🕐 Time: Q{situation.quarter} {situation.game_clock}")
            print(f"      🎯 Situation: {situation.situation_type} ({situation.pressure_level} pressure)")
            print(f"      ⚡ Context: {situation.strategic_context}")
            print(f"      📈 Expected Points: {situation.expected_points:.2f}")
            print(f"      🎲 Win Probability: {situation.win_probability:.1%}")
            print(f"      💪 Analysis Confidence: {situation.analysis_confidence:.3f}")
            print(f"      ⚡ Processing: {situation.processing_time_ms:.1f}ms")
            print(f"      ✅ Valid: {situation.is_valid()}")
            
            situations.append(situation)
    
    # Save complete analysis
    if situations:
        output_data = {
            'spygate_analysis': {
                'version': '6.8_production',
                'timestamp': datetime.now().isoformat(),
                'total_situations': len(situations),
                'valid_situations': len([s for s in situations if s.is_valid()])
            },
            'game_situations': [situation.to_dict() for situation in situations]
        }
        
        with open("production_game_analysis.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Saved complete analysis to: production_game_analysis.json")
    
    print("\n🎉 Production Analysis Complete!")
    return situations

if __name__ == "__main__":
    test_production_analyzer() 
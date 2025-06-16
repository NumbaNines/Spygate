"""
Spygate Clip Segmentation Fix
============================

This module fixes the clip segmentation issues by:
1. Using YOLO preplay_indicator and play_call_screen for clip START detection
2. Keeping existing logic for clip END detection (down changes, etc.)
3. Ensuring each down is captured as an individual clip
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from datetime import datetime
from typing import Dict, List, Optional, Tuple


def create_enhanced_play_boundary_analyzer():
    """
    Creates an enhanced play boundary analyzer that uses YOLO indicators
    for precise clip start detection.
    """
    
    class EnhancedPlayBoundaryAnalyzer:
        def __init__(self):
            # Track YOLO indicator states
            self.preplay_indicator_active = False
            self.play_call_screen_active = False
            self.indicator_first_seen_frame = None
            self.indicator_last_seen_frame = None
            
            # Track play state
            self.play_active = False
            self.play_start_frame = None
            self.play_end_frame = None
            
            # Track game state for end detection
            self.previous_down = None
            self.previous_distance = None
            self.previous_possession = None
            self.previous_yard_line = None
            self.previous_game_clock = None
            
            # Clip creation state
            self.pending_clip_start = None
            self.clips_created = []
            
        def analyze_frame(self, frame_number: int, game_state, detections: List[dict]) -> Dict:
            """
            Analyze current frame for play boundaries using YOLO indicators and game state.
            
            Args:
                frame_number: Current frame number
                game_state: Current game state object
                detections: List of YOLO detections
                
            Returns:
                Dictionary with play boundary information
            """
            result = {
                "play_started": False,
                "play_ended": False,
                "create_clip": False,
                "clip_start_frame": None,
                "clip_end_frame": None,
                "clip_reason": None,
                "debug_info": {}
            }
            
            # Extract YOLO indicators
            has_preplay = any(d.get("class") == "preplay_indicator" for d in detections)
            has_playcall = any(d.get("class") == "play_call_screen" for d in detections)
            current_has_indicator = has_preplay or has_playcall
            
            # Debug logging
            if frame_number % 150 == 0:  # Log every 5 seconds at 30fps
                print(f"🔍 Frame {frame_number}: preplay={has_preplay}, playcall={has_playcall}, "
                      f"play_active={self.play_active}, down={getattr(game_state, 'down', None)}")
            
            # === CLIP START DETECTION (YOLO-based) ===
            if current_has_indicator and not (self.preplay_indicator_active or self.play_call_screen_active):
                # Indicators just appeared - mark potential clip start
                self.indicator_first_seen_frame = frame_number
                self.preplay_indicator_active = has_preplay
                self.play_call_screen_active = has_playcall
                
                # Set pending clip start with pre-buffer
                pre_buffer_frames = 90  # 3 seconds at 30fps
                self.pending_clip_start = max(0, frame_number - pre_buffer_frames)
                
                print(f"🎯 CLIP START MARKED: Indicators appeared at frame {frame_number}")
                print(f"   📍 Clip will start at frame {self.pending_clip_start} (with 3s pre-buffer)")
                
            elif not current_has_indicator and (self.preplay_indicator_active or self.play_call_screen_active):
                # Indicators just disappeared - play is starting!
                self.indicator_last_seen_frame = frame_number
                self.preplay_indicator_active = False
                self.play_call_screen_active = False
                self.play_active = True
                self.play_start_frame = frame_number
                
                result["play_started"] = True
                print(f"🏈 PLAY STARTED: Indicators disappeared at frame {frame_number}")
                
            # Update indicator tracking
            if current_has_indicator:
                self.indicator_last_seen_frame = frame_number
            
            # === CLIP END DETECTION (Game state-based) ===
            if self.play_active and game_state:
                play_ended = False
                end_reason = None
                
                # Extract current state
                current_down = getattr(game_state, 'down', None)
                current_distance = getattr(game_state, 'distance', None)
                current_possession = getattr(game_state, 'possession_team', None)
                current_yard_line = getattr(game_state, 'yard_line', None)
                current_game_clock = getattr(game_state, 'time', None)
                
                # 1. DOWN CHANGE - Most reliable indicator
                if (self.previous_down is not None and 
                    current_down is not None and 
                    current_down != self.previous_down):
                    play_ended = True
                    end_reason = f"down_changed_{self.previous_down}_to_{current_down}"
                    print(f"🏁 PLAY ENDED: Down changed {self.previous_down} → {current_down}")
                
                # 2. FIRST DOWN ACHIEVED - Distance reset to 10
                elif (current_down == 1 and 
                      current_distance == 10 and 
                      self.previous_distance is not None and
                      self.previous_distance < 10):
                    play_ended = True
                    end_reason = "first_down_achieved"
                    print(f"🏁 PLAY ENDED: First down achieved")
                
                # 3. POSSESSION CHANGE - Turnover
                elif (self.previous_possession is not None and
                      current_possession is not None and
                      current_possession != self.previous_possession):
                    play_ended = True
                    end_reason = f"possession_changed_to_{current_possession}"
                    print(f"🏁 PLAY ENDED: Possession changed to {current_possession}")
                
                # 4. BIG YARDAGE CHANGE
                elif (self.previous_yard_line is not None and
                      current_yard_line is not None):
                    yard_change = abs(current_yard_line - self.previous_yard_line)
                    if yard_change >= 15:
                        play_ended = True
                        end_reason = f"big_play_{yard_change}_yards"
                        print(f"🏁 PLAY ENDED: Big play for {yard_change} yards")
                
                # 5. MAXIMUM PLAY DURATION
                elif self.play_start_frame and (frame_number - self.play_start_frame) > 300:  # 10 seconds
                    play_ended = True
                    end_reason = "max_duration_exceeded"
                    print(f"🏁 PLAY ENDED: Maximum duration reached")
                
                # If play ended, create the clip
                if play_ended:
                    self.play_active = False
                    self.play_end_frame = frame_number
                    result["play_ended"] = True
                    
                    # Create clip with post-buffer
                    if self.pending_clip_start is not None:
                        post_buffer_frames = 60  # 2 seconds at 30fps
                        clip_end = frame_number + post_buffer_frames
                        
                        result["create_clip"] = True
                        result["clip_start_frame"] = self.pending_clip_start
                        result["clip_end_frame"] = clip_end
                        result["clip_reason"] = end_reason
                        
                        # Calculate clip duration
                        duration_seconds = (clip_end - self.pending_clip_start) / 30.0
                        
                        print(f"🎬 CREATE CLIP: Frames {self.pending_clip_start}-{clip_end} ({duration_seconds:.1f}s)")
                        print(f"   📋 Reason: {end_reason}")
                        print(f"   🏈 Contains: Down {self.previous_down} & {self.previous_distance}")
                        
                        # Record clip creation
                        self.clips_created.append({
                            "start_frame": self.pending_clip_start,
                            "end_frame": clip_end,
                            "reason": end_reason,
                            "down": self.previous_down,
                            "distance": self.previous_distance,
                            "duration": duration_seconds
                        })
                        
                        # Reset clip start tracking
                        self.pending_clip_start = None
                
                # Update previous state for next frame
                self.previous_down = current_down
                self.previous_distance = current_distance
                self.previous_possession = current_possession
                self.previous_yard_line = current_yard_line
                self.previous_game_clock = current_game_clock
            
            # Debug info
            result["debug_info"] = {
                "has_preplay": has_preplay,
                "has_playcall": has_playcall,
                "play_active": self.play_active,
                "pending_clip_start": self.pending_clip_start,
                "current_down": getattr(game_state, 'down', None),
                "clips_created_count": len(self.clips_created)
            }
            
            return result
    
    return EnhancedPlayBoundaryAnalyzer()


def integrate_with_analysis_worker(analyzer_instance):
    """
    Integrates the enhanced play boundary analyzer with the existing AnalysisWorker.
    This function modifies the worker to use YOLO indicators for clip starts.
    """
    
    # Create the enhanced analyzer
    play_boundary_analyzer = create_enhanced_play_boundary_analyzer()
    
    # Store it on the analyzer instance
    analyzer_instance._play_boundary_analyzer = play_boundary_analyzer
    
    # Monkey-patch the analyze_play_boundaries method
    original_analyze_play_boundaries = analyzer_instance._analyze_play_boundaries
    
    def enhanced_analyze_play_boundaries(game_state, current_frame):
        """Enhanced version that uses YOLO indicators for clip starts."""
        
        # Get YOLO detections from the current frame analysis
        # This assumes detections are stored on the game_state or analyzer
        detections = getattr(game_state, 'detections', [])
        if not detections and hasattr(analyzer_instance, 'last_detections'):
            detections = analyzer_instance.last_detections
        
        # Use our enhanced analyzer
        result = play_boundary_analyzer.analyze_frame(current_frame, game_state, detections)
        
        # Convert to the format expected by the existing code
        boundary_info = {
            "play_started": result["play_started"],
            "play_ended": result["play_ended"],
            "play_in_progress": play_boundary_analyzer.play_active,
            "clip_should_start": False,  # We handle this differently now
            "clip_should_end": result["create_clip"],
            "recommended_clip_start": result["clip_start_frame"],
            "recommended_clip_end": result["clip_end_frame"],
            "play_type": result["clip_reason"] or "unknown",
            "play_situation": "normal",
            "confidence": 0.9,
            "create_immediate_clip": result["create_clip"],  # New field
            "debug_info": result["debug_info"]
        }
        
        return boundary_info
    
    # Replace the method
    analyzer_instance._analyze_play_boundaries = enhanced_analyze_play_boundaries
    
    print("✅ Enhanced play boundary analyzer integrated!")
    return analyzer_instance


def print_usage_instructions():
    """Print instructions for using this fix."""
    print("""
    ========================================
    Spygate Clip Segmentation Fix - Usage
    ========================================
    
    This fix enhances the clip segmentation to use YOLO indicators for precise play detection.
    
    Integration Steps:
    1. Import this module in your main application
    2. Call integrate_with_analysis_worker() on your AnalysisWorker instance
    3. The system will now use preplay_indicator/play_call_screen for clip starts
    4. Existing down change logic is preserved for clip ends
    
    Example:
    ```python
    from spygate_clip_segmentation_fix import integrate_with_analysis_worker
    
    # In your AnalysisWorker initialization
    self.analyzer = enhanced_game_analyzer  # Your existing analyzer
    integrate_with_analysis_worker(self)
    ```
    
    Features:
    - ✅ YOLO-based clip start detection (preplay_indicator, play_call_screen)
    - ✅ Game state-based clip end detection (down changes, first downs, etc.)
    - ✅ Individual clips for each down
    - ✅ Configurable pre/post buffers
    - ✅ Comprehensive debug logging
    
    Debug Output:
    - 🎯 CLIP START MARKED - When indicators appear
    - 🏈 PLAY STARTED - When indicators disappear (snap)
    - 🏁 PLAY ENDED - When down changes or other end conditions
    - 🎬 CREATE CLIP - When clip is ready with exact frame ranges
    
    """)


if __name__ == "__main__":
    print_usage_instructions()
    
    # Test the analyzer with sample data
    print("\n=== Testing Enhanced Play Boundary Analyzer ===\n")
    
    analyzer = create_enhanced_play_boundary_analyzer()
    
    # Simulate a sequence of frames
    test_frames = [
        # Pre-play with indicator
        {"frame": 100, "detections": [{"class": "preplay_indicator"}], "down": 1, "distance": 10},
        {"frame": 110, "detections": [{"class": "preplay_indicator"}], "down": 1, "distance": 10},
        
        # Snap - indicator disappears
        {"frame": 120, "detections": [], "down": 1, "distance": 10},
        {"frame": 150, "detections": [], "down": 1, "distance": 10},
        
        # Play ends - down changes
        {"frame": 180, "detections": [], "down": 2, "distance": 7},
        
        # Next play - indicator appears again
        {"frame": 300, "detections": [{"class": "play_call_screen"}], "down": 2, "distance": 7},
        {"frame": 310, "detections": [{"class": "play_call_screen"}], "down": 2, "distance": 7},
        
        # Snap
        {"frame": 320, "detections": [], "down": 2, "distance": 7},
        
        # First down achieved
        {"frame": 380, "detections": [], "down": 1, "distance": 10},
    ]
    
    class MockGameState:
        def __init__(self, down, distance):
            self.down = down
            self.distance = distance
            self.possession_team = "home"
            self.yard_line = 50
            self.time = "10:00"
    
    for frame_data in test_frames:
        game_state = MockGameState(frame_data["down"], frame_data["distance"])
        result = analyzer.analyze_frame(
            frame_data["frame"], 
            game_state, 
            frame_data["detections"]
        )
        
        if result["create_clip"]:
            print(f"\n✅ CLIP CREATED: {result}")
    
    print(f"\n📊 Total clips created: {len(analyzer.clips_created)}")
    for i, clip in enumerate(analyzer.clips_created):
        print(f"   Clip {i+1}: {clip}")
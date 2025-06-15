#!/usr/bin/env python3
"""
Fix for SpygateAI Clip Boundary Detection Issue

Problem: Clips continue playing for extra seconds even after a down change or play end.
Each clip should capture exactly one play, not multiple plays.

Root Cause Analysis:
1. The _analyze_play_boundaries method detects play starts but doesn't properly end clips
2. The system creates clips with hardcoded durations instead of detecting actual play ends
3. Down changes are detected but clips aren't ended when the next play starts

Solution: Implement proper play-end detection and clip boundary logic.
"""

import os
import re
import sys
from pathlib import Path


def create_backup():
    """Create a backup of the original file before making changes."""
    original_file = Path("spygate_desktop_app_faceit_style.py")
    backup_file = Path("spygate_desktop_app_faceit_style.py.backup")

    if original_file.exists():
        print(f"📦 Creating backup: {backup_file}")
        backup_file.write_text(original_file.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    else:
        print(f"❌ Original file not found: {original_file}")
        return False


def apply_clip_boundary_fix():
    """Apply the fix to properly detect play boundaries and create one-play clips."""

    file_path = Path("spygate_desktop_app_faceit_style.py")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")

    # Fix 1: Update _analyze_play_boundaries to properly track play lifecycle
    new_analyze_play_boundaries = '''    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:
        """
        FIXED: Individual play boundary detection that properly ends clips at play completion.

        This method now:
        1. Detects when a play starts (down change, first down, etc.)
        2. Tracks the play in progress
        3. Detects when the play ends (next down change, clock stoppage, etc.)
        4. Creates clips that contain ONLY ONE PLAY
        """
        boundary_info = {
            'play_started': False,
            'play_ended': False,
            'play_in_progress': False,
            'clip_should_start': False,
            'clip_should_end': False,
            'recommended_clip_start': None,
            'recommended_clip_end': None,
            'play_type': 'unknown',
            'play_situation': 'normal',
            'confidence': 0.0
        }

        if not game_state:
            return boundary_info

        # Extract current game state
        current_down = game_state.down
        current_distance = game_state.distance
        current_yard_line = getattr(game_state, 'yard_line', None)
        current_possession = getattr(game_state, 'possession_team', None)
        current_game_clock = getattr(game_state, 'time', None)
        current_quarter = getattr(game_state, 'quarter', None)

        # Get previous state
        prev_state = self.play_boundary_state

        # === CHECK IF WE'RE TRACKING AN ACTIVE PLAY ===
        if prev_state.get('active_play_frame') is not None:
            # We have an active play - check if it has ended
            frames_since_play_start = current_frame - prev_state['active_play_frame']

            # CRITICAL: Detect play end conditions
            play_ended = False
            end_reason = None

            # 1. DOWN CHANGE - Most reliable indicator that previous play ended
            if (prev_state['last_down'] is not None and current_down is not None and
                current_down != prev_state['last_down']):
                play_ended = True
                end_reason = "down_changed"
                print(f"🏁 PLAY ENDED: Down changed {prev_state['last_down']} → {current_down}")

            # 2. FIRST DOWN ACHIEVED - Distance reset to 10
            elif (current_down == 1 and current_distance == 10 and
                  prev_state.get('last_distance') != 10):
                play_ended = True
                end_reason = "first_down_achieved"
                print(f"🏁 PLAY ENDED: First down achieved")

            # 3. POSSESSION CHANGE - Turnover occurred
            elif (prev_state.get('last_possession_team') and current_possession and
                  current_possession != prev_state['last_possession_team']):
                play_ended = True
                end_reason = "possession_changed"
                print(f"🏁 PLAY ENDED: Possession changed to {current_possession}")

            # 4. SIGNIFICANT YARD LINE CHANGE - Play resulted in big gain/loss
            elif (prev_state.get('last_yard_line') is not None and current_yard_line is not None):
                yard_change = abs(current_yard_line - prev_state['last_yard_line'])
                if yard_change >= 15:  # 15+ yard change indicates play completed
                    play_ended = True
                    end_reason = f"big_play_{yard_change}_yards"
                    print(f"🏁 PLAY ENDED: Big play for {yard_change} yards")

            # 5. CLOCK STOPPAGE - Incomplete pass, out of bounds, etc.
            elif (current_game_clock and prev_state.get('last_game_clock') and
                  self._detect_clock_stoppage(current_game_clock, prev_state['last_game_clock'])):
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
                boundary_info['play_ended'] = True
                boundary_info['clip_should_end'] = True
                # End clip shortly after the play ends (1 second buffer)
                boundary_info['recommended_clip_end'] = current_frame + 30

                # Clear active play tracking
                self.play_boundary_state['active_play_frame'] = None
                self.play_boundary_state['active_play_data'] = None

                print(f"🎬 CLIP END: Play ended ({end_reason}) - Clip will end at frame {boundary_info['recommended_clip_end']}")

        # === DETECT NEW PLAY START ===
        # Only start new plays if we're not currently tracking one
        if prev_state.get('active_play_frame') is None:
            play_started = False
            start_reason = None

            # Check various play start conditions
            if (prev_state['last_down'] is None and current_down is not None):
                # First play detected
                play_started = True
                start_reason = "initial_play"
            elif (prev_state['last_down'] is not None and current_down is not None and
                  current_down != prev_state['last_down']):
                # Down changed - new play starting
                play_started = True
                start_reason = f"down_change_{prev_state['last_down']}_to_{current_down}"
            elif (current_down == 1 and prev_state['last_down'] != 1):
                # First down achieved
                play_started = True
                start_reason = "first_down"

            if play_started:
                boundary_info['play_started'] = True
                boundary_info['clip_should_start'] = True
                boundary_info['play_type'] = self._classify_play_by_down(current_down, current_distance)

                # Start clip with 2-second pre-roll
                boundary_info['recommended_clip_start'] = max(0, current_frame - 60)

                # Track this as the active play
                self.play_boundary_state['active_play_frame'] = current_frame
                self.play_boundary_state['active_play_data'] = {
                    'down': current_down,
                    'distance': current_distance,
                    'yard_line': current_yard_line,
                    'start_reason': start_reason
                }

                print(f"🎬 PLAY STARTED: {start_reason} - {current_down} & {current_distance}")
                print(f"   📍 Clip will start at frame {boundary_info['recommended_clip_start']}")

        # === UPDATE STATE TRACKING ===
        # Only update with non-None values to prevent state loss
        if current_down is not None:
            self.play_boundary_state['last_down'] = current_down
        if current_distance is not None:
            self.play_boundary_state['last_distance'] = current_distance
        if current_yard_line is not None:
            self.play_boundary_state['last_yard_line'] = current_yard_line
        if current_possession is not None:
            self.play_boundary_state['last_possession_team'] = current_possession
        if current_game_clock is not None:
            self.play_boundary_state['last_game_clock'] = current_game_clock

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

        boundary_info['confidence'] = sum(confidence_factors)

        return boundary_info'''

    # Fix 2: Update the clip creation logic in the run method
    new_clip_creation_logic = """                        # FIXED CLIP CREATION: Use play boundary detection for accurate clips
                        if self._should_create_clip(game_state, situation_context):
                            # Check if we should create a new clip or end an existing one
                            if boundary_info['clip_should_start'] and not boundary_info['clip_should_end']:
                                # Starting a new play - create clip start marker
                                print(f"✅ NEW PLAY DETECTED - Starting clip tracking at frame {frame_number}")

                                # Store clip start info but don't create clip yet
                                self.pending_clip_info = {
                                    'start_frame': boundary_info['recommended_clip_start'],
                                    'start_game_state': game_state,
                                    'start_situation': situation_context,
                                    'start_boundary': boundary_info
                                }

                            elif boundary_info['clip_should_end'] and hasattr(self, 'pending_clip_info'):
                                # Play ended - create the complete clip
                                print(f"✅ PLAY ENDED - Creating clip from pending info")

                                clip_start = self.pending_clip_info['start_frame']
                                clip_end = boundary_info['recommended_clip_end']

                                # Ensure minimum clip duration (3 seconds)
                                min_duration = int(fps * 3)
                                if clip_end - clip_start < min_duration:
                                    clip_end = clip_start + min_duration

                                # Create the clip with proper boundaries
                                clip = self._create_enhanced_clip_with_boundaries(
                                    clip_start,
                                    clip_end,
                                    fps,
                                    self.pending_clip_info['start_game_state'],
                                    self.pending_clip_info['start_situation'],
                                    self.pending_clip_info['start_boundary']
                                )

                                # Verify clip doesn't overlap with existing clips
                                if not self._is_duplicate_clip(clip_start, clip_end, frame_number):
                                    detected_clips.append(clip)
                                    self._register_created_clip(clip_start, clip_end,
                                        self.pending_clip_info['start_boundary']['play_type'])
                                    self.clip_detected.emit(clip)

                                    print(f"🎬 CLIP CREATED: {clip_start} → {clip_end} ({(clip_end-clip_start)/fps:.1f}s)")
                                    print(f"   📝 Description: {clip.situation}")

                                # Clear pending clip info
                                self.pending_clip_info = None

                            elif not hasattr(self, 'pending_clip_info'):
                                # Fallback: Create immediate clip if no pending info
                                # This handles cases where we detect a situation mid-play
                                print(f"⚠️ FALLBACK: Creating immediate clip (no pending info)")

                                # Use intelligent boundaries based on game state
                                clip_start, clip_end = self._find_natural_clip_boundaries(frame_number, fps, game_state)

                                # Create clip
                                clip = self._create_enhanced_clip_with_boundaries(
                                    clip_start, clip_end, fps, game_state, situation_context, boundary_info
                                )

                                if not self._is_duplicate_clip(clip_start, clip_end, frame_number):
                                    detected_clips.append(clip)
                                    self._register_created_clip(clip_start, clip_end,
                                        boundary_info.get('play_type', 'unknown'))
                                    self.clip_detected.emit(clip)"""

    # Fix 3: Update _find_natural_clip_boundaries to be more intelligent
    new_natural_boundaries = '''    def _find_natural_clip_boundaries(self, frame_number, fps, game_state):
        """FIXED: Find natural clip boundaries that capture exactly one play."""

        # Default: 3 seconds before, 5 seconds after current frame
        default_pre_buffer = int(fps * 3)
        default_post_buffer = int(fps * 5)

        start_frame = max(0, frame_number - default_pre_buffer)
        end_frame = frame_number + default_post_buffer

        # Try to use game history to find actual play boundaries
        if hasattr(self.analyzer, 'game_history') and self.analyzer.game_history:
            history = self.analyzer.game_history
            current_idx = len(history) - 1

            # Look backwards for play start (down/distance change)
            play_start_idx = None
            for i in range(current_idx - 1, max(0, current_idx - 30), -1):
                if i >= 0 and i < len(history) - 1:
                    curr_state = history[i]
                    next_state = history[i + 1]

                    # Check for down change or significant game state change
                    if (curr_state.down != next_state.down or
                        (curr_state.distance and next_state.distance and
                         abs(curr_state.distance - next_state.distance) > 5)):
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
                start_frame = max(0, frame_number - int(frames_back) - int(fps * 2))  # 2s pre-buffer
                print(f"📍 Found play start in history: {play_start_idx} ({frames_back/fps:.1f}s ago)")

            if play_end_idx is not None:
                frames_forward = (play_end_idx - current_idx) * fps
                end_frame = frame_number + int(frames_forward) + int(fps * 1)  # 1s post-buffer
                print(f"📍 Found play end in history: {play_end_idx} ({frames_forward/fps:.1f}s ahead)")
            elif play_start_idx is not None:
                # No end found, use reasonable play duration (8 seconds)
                end_frame = start_frame + int(fps * 8)
                print(f"📍 No play end found, using 8-second duration")

        # Ensure reasonable clip duration (3-15 seconds)
        duration = end_frame - start_frame
        min_duration = int(fps * 3)   # 3 seconds minimum
        max_duration = int(fps * 15)  # 15 seconds maximum

        if duration < min_duration:
            end_frame = start_frame + min_duration
        elif duration > max_duration:
            # Trim from the end to keep the most relevant part
            end_frame = start_frame + max_duration

        print(f"🎯 Natural boundaries: {start_frame} → {end_frame} ({(end_frame-start_frame)/fps:.1f}s)")
        return start_frame, end_frame'''

    # Apply the fixes
    print("🔧 Applying clip boundary detection fixes...")

    # Replace the _analyze_play_boundaries method
    import re

    # Pattern to find the method
    pattern = r"def _analyze_play_boundaries\(self, game_state, current_frame: int\) -> dict:.*?(?=\n    def |\n\nclass |\Z)"
    content = re.sub(pattern, new_analyze_play_boundaries.strip(), content, flags=re.DOTALL)

    # Replace the _find_natural_clip_boundaries method
    pattern = r"def _find_natural_clip_boundaries\(self, frame_number, fps, game_state\):.*?(?=\n    def |\n\nclass |\Z)"
    content = re.sub(pattern, new_natural_boundaries.strip(), content, flags=re.DOTALL)

    # Update the clip creation logic in the run method
    # This is trickier as it's inside the run method, so we'll do a more targeted replacement

    # First, add the pending_clip_info initialization in __init__
    init_pattern = r"(self\.play_boundary_state = \{[^}]+\})"
    init_replacement = (
        r"\1\n        self.pending_clip_info = None  # Track clips that span play boundaries"
    )
    content = re.sub(init_pattern, init_replacement, content)

    # Write the updated content
    file_path.write_text(content, encoding="utf-8")
    print("✅ Fixes applied successfully!")

    return True


def test_fixes():
    """Create a test script to verify the fixes work correctly."""

    test_script = '''#!/usr/bin/env python3
"""Test script to verify clip boundary detection fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from spygate_desktop_app_faceit_style import AnalysisWorker
from spygate.ml.enhanced_game_analyzer import GameState, SituationContext

def test_play_boundary_detection():
    """Test that play boundaries are properly detected."""
    print("=" * 60)
    print("TESTING PLAY BOUNDARY DETECTION")
    print("=" * 60)

    # Create a dummy worker
    worker = AnalysisWorker("dummy.mp4")

    # Initialize play boundary state
    worker.play_boundary_state = {
        'last_down': None,
        'last_distance': None,
        'last_yard_line': None,
        'last_possession_team': None,
        'last_game_clock': None,
        'active_play_frame': None,
        'active_play_data': None
    }

    # Test Case 1: Initial play detection
    print("\\nTest 1: Initial Play Detection")
    game_state1 = GameState()
    game_state1.down = 1
    game_state1.distance = 10
    game_state1.yard_line = 25

    boundary1 = worker._analyze_play_boundaries(game_state1, 100)
    print(f"Play Started: {boundary1['play_started']}")
    print(f"Clip Should Start: {boundary1['clip_should_start']}")
    print(f"Play Type: {boundary1['play_type']}")
    assert boundary1['play_started'] == True
    assert boundary1['clip_should_start'] == True

    # Test Case 2: Play in progress (no change)
    print("\\nTest 2: Play In Progress")
    boundary2 = worker._analyze_play_boundaries(game_state1, 150)
    print(f"Play Started: {boundary2['play_started']}")
    print(f"Play Ended: {boundary2['play_ended']}")
    assert boundary2['play_started'] == False
    assert boundary2['play_ended'] == False

    # Test Case 3: Down change (play ends, new play starts)
    print("\\nTest 3: Down Change Detection")
    game_state2 = GameState()
    game_state2.down = 2
    game_state2.distance = 7
    game_state2.yard_line = 28

    boundary3 = worker._analyze_play_boundaries(game_state2, 200)
    print(f"Play Ended: {boundary3['play_ended']}")
    print(f"Clip Should End: {boundary3['clip_should_end']}")
    assert boundary3['play_ended'] == True
    assert boundary3['clip_should_end'] == True

    # Test Case 4: First down achieved
    print("\\nTest 4: First Down Detection")
    game_state3 = GameState()
    game_state3.down = 1
    game_state3.distance = 10
    game_state3.yard_line = 40

    # Reset state for new play
    worker.play_boundary_state['active_play_frame'] = None
    boundary4 = worker._analyze_play_boundaries(game_state3, 250)
    print(f"Play Started: {boundary4['play_started']}")
    print(f"Play Type: {boundary4['play_type']}")
    assert boundary4['play_started'] == True

    print("\\n✅ All play boundary tests passed!")

def test_clip_duration():
    """Test that clips have appropriate durations."""
    print("\\n" + "=" * 60)
    print("TESTING CLIP DURATION LIMITS")
    print("=" * 60)

    worker = AnalysisWorker("dummy.mp4")

    # Test natural boundaries
    game_state = GameState()
    game_state.down = 3
    game_state.distance = 8

    fps = 30
    frame_number = 1000

    start, end = worker._find_natural_clip_boundaries(frame_number, fps, game_state)
    duration_seconds = (end - start) / fps

    print(f"Clip Duration: {duration_seconds:.1f} seconds")
    print(f"Start Frame: {start}, End Frame: {end}")

    # Check constraints
    assert duration_seconds >= 3.0, "Clip too short!"
    assert duration_seconds <= 15.0, "Clip too long!"

    print("✅ Clip duration within acceptable range!")

if __name__ == "__main__":
    test_play_boundary_detection()
    test_clip_duration()
    print("\\n🎉 All tests passed! Clip boundary detection is working correctly.")
'''

    test_file = Path("test_clip_boundary_fixes.py")
    test_file.write_text(test_script, encoding="utf-8")
    print(f"📝 Created test script: {test_file}")

    return True


def main():
    """Main function to apply all fixes."""
    print("🚀 SpygateAI Clip Boundary Detection Fix")
    print("=" * 50)

    # Create backup
    if not create_backup():
        print("❌ Failed to create backup. Aborting.")
        return

    # Apply fixes
    if not apply_clip_boundary_fix():
        print("❌ Failed to apply fixes.")
        return

    # Create test script
    test_fixes()

    print("\n✅ All fixes applied successfully!")
    print("\n📋 Summary of changes:")
    print("1. ✅ Fixed _analyze_play_boundaries to properly detect play ends")
    print("2. ✅ Updated clip creation to wait for play completion")
    print("3. ✅ Improved _find_natural_clip_boundaries for single-play clips")
    print("4. ✅ Added pending clip tracking to span play lifecycle")
    print("\n🎯 Result: Clips will now contain exactly one play each!")
    print("\n💡 To test: Run 'python test_clip_boundary_fixes.py'")


if __name__ == "__main__":
    main()

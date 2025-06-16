#!/usr/bin/env python3
"""
Quick Fix for Spygate Clip Segmentation
======================================

One-line fix for the clip segmentation issues.
Just run this script and your clips will be properly segmented!
"""

import os
import sys
from pathlib import Path


def apply_quick_fix():
    """Apply the clip segmentation fix with minimal changes."""
    
    print("🚀 Spygate Quick Clip Fix")
    print("=" * 40)
    
    # Find the desktop app
    app_file = Path(__file__).parent / "spygate_desktop_app_faceit_style.py"
    if not app_file.exists():
        print(f"❌ Error: Cannot find {app_file}")
        return False
    
    # Read the file
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "YOLO_CLIP_FIX_APPLIED" in content:
        print("✅ Fix already applied!")
        return True
    
    # Create backup
    backup = app_file.with_suffix('.py.pre_clip_fix_backup')
    with open(backup, 'w') as f:
        f.write(content)
    print(f"💾 Backup saved to {backup}")
    
    # Find the _analyze_play_boundaries method
    method_start = content.find("    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:")
    if method_start < 0:
        print("❌ Error: Cannot find _analyze_play_boundaries method")
        return False
    
    # Find the end of the method
    method_end = content.find("\n    def ", method_start + 1)
    if method_end < 0:
        method_end = len(content)
    
    # Create the new method with YOLO detection
    new_method = '''    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:
        """
        ENHANCED WITH YOLO CLIP FIX: Uses preplay_indicator/play_call_screen for clip starts.
        YOLO_CLIP_FIX_APPLIED = True
        """
        
        # Initialize boundary info
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
        
        # === YOLO-BASED CLIP START DETECTION ===
        # Get YOLO detections if available
        detections = []
        if hasattr(game_state, 'detections'):
            detections = game_state.detections
        elif hasattr(self, 'analyzer') and hasattr(self.analyzer, 'last_detections'):
            detections = self.analyzer.last_detections
        
        # Check for preplay indicators
        has_preplay = any(d.get("class") == "preplay_indicator" for d in detections)
        has_playcall = any(d.get("class") == "play_call_screen" for d in detections)
        has_indicator = has_preplay or has_playcall
        
        # Initialize state tracking if needed
        if not hasattr(self, '_yolo_clip_state'):
            self._yolo_clip_state = {
                'indicator_active': False,
                'play_active': False,
                'clip_start_frame': None,
                'last_down': None,
                'last_distance': None,
                'last_possession': None,
                'last_yard_line': None,
            }
        
        state = self._yolo_clip_state
        
        # Detect clip start (indicator appears)
        if has_indicator and not state['indicator_active']:
            state['indicator_active'] = True
            state['clip_start_frame'] = max(0, current_frame - 90)  # 3 second pre-buffer
            print(f"🎯 YOLO CLIP START: Indicators detected at frame {current_frame}")
            print(f"   📍 Clip will start at frame {state['clip_start_frame']}")
        
        # Detect play start (indicator disappears)
        elif not has_indicator and state['indicator_active']:
            state['indicator_active'] = False
            state['play_active'] = True
            boundary_info["play_started"] = True
            print(f"🏈 PLAY STARTED: Snap detected at frame {current_frame}")
        
        # === EXISTING CLIP END DETECTION ===
        if state['play_active'] and game_state:
            # Extract current state
            current_down = game_state.down
            current_distance = game_state.distance
            current_possession = getattr(game_state, 'possession_team', None)
            current_yard_line = getattr(game_state, 'yard_line', None)
            
            play_ended = False
            end_reason = None
            
            # Check various end conditions (keeping your existing logic)
            if state['last_down'] is not None and current_down != state['last_down']:
                play_ended = True
                end_reason = f"down_change_{state['last_down']}_to_{current_down}"
            elif (current_down == 1 and current_distance == 10 and 
                  state['last_distance'] is not None and state['last_distance'] < 10):
                play_ended = True
                end_reason = "first_down"
            elif (state['last_possession'] and current_possession and 
                  current_possession != state['last_possession']):
                play_ended = True
                end_reason = "possession_change"
            elif state['last_yard_line'] and current_yard_line:
                yard_change = abs(current_yard_line - state['last_yard_line'])
                if yard_change >= 15:
                    play_ended = True
                    end_reason = f"big_play_{yard_change}_yards"
            
            # If play ended, mark for clip creation
            if play_ended and state['clip_start_frame'] is not None:
                state['play_active'] = False
                boundary_info["play_ended"] = True
                boundary_info["clip_should_end"] = True
                boundary_info["recommended_clip_start"] = state['clip_start_frame']
                boundary_info["recommended_clip_end"] = current_frame + 60  # 2 second post-buffer
                boundary_info["play_type"] = end_reason
                
                print(f"🏁 PLAY ENDED: {end_reason} at frame {current_frame}")
                print(f"🎬 CREATE CLIP: Frames {state['clip_start_frame']}-{boundary_info['recommended_clip_end']}")
                
                # Reset for next play
                state['clip_start_frame'] = None
            
            # Update state for next frame
            state['last_down'] = current_down
            state['last_distance'] = current_distance
            state['last_possession'] = current_possession
            state['last_yard_line'] = current_yard_line
        
        # Store play boundary state
        self.play_boundary_state = {
            "last_down": state.get('last_down'),
            "last_distance": state.get('last_distance'),
            "last_possession_team": state.get('last_possession'),
            "last_yard_line": state.get('last_yard_line'),
            "active_play_frame": current_frame if state['play_active'] else None,
        }
        
        return boundary_info
'''
    
    # Replace the method
    content = content[:method_start] + new_method + "\n" + content[method_end:]
    
    # Also need to ensure detections are passed through from analyzer
    # Find analyze_frame calls
    analyze_calls = content.find("game_state = self.analyzer.analyze_frame(")
    if analyze_calls > 0:
        # Add detection extraction after the call
        injection = """
                    
                    # Extract YOLO detections for clip boundary analysis
                    if hasattr(self.analyzer, 'last_detections'):
                        if game_state and hasattr(game_state, '__dict__'):
                            game_state.detections = self.analyzer.last_detections"""
        
        # Find the end of the statement
        stmt_end = content.find("\n", analyze_calls)
        while stmt_end > 0 and content[stmt_end-1] == ')':
            stmt_end = content.find("\n", stmt_end + 1)
        
        if stmt_end > 0:
            content = content[:stmt_end] + injection + content[stmt_end:]
    
    # Write the fixed file
    with open(app_file, 'w') as f:
        f.write(content)
    
    print("✅ Clip segmentation fix applied!")
    print("\n📋 What changed:")
    print("- Clips now start when preplay_indicator or play_call_screen appears")
    print("- Clips end on down changes (your existing logic)")
    print("- Each down is captured as an individual clip")
    print("\n🎮 Just run the app normally - clips will be properly segmented!")
    
    return True


def verify_fix():
    """Verify the fix was applied."""
    app_file = Path(__file__).parent / "spygate_desktop_app_faceit_style.py"
    
    if not app_file.exists():
        return False
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    return "YOLO_CLIP_FIX_APPLIED" in content


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        if verify_fix():
            print("✅ Clip fix is active")
            return 0
        else:
            print("❌ Clip fix not applied")
            return 1
    
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        # Revert to backup
        app_file = Path(__file__).parent / "spygate_desktop_app_faceit_style.py"
        backup = app_file.with_suffix('.py.pre_clip_fix_backup')
        
        if backup.exists():
            import shutil
            shutil.copy(backup, app_file)
            print("✅ Reverted to backup")
            return 0
        else:
            print("❌ No backup found")
            return 1
    
    # Apply the fix
    if apply_quick_fix():
        print("\n✨ Success! Your clips will now be properly segmented.")
        print("\nUsage:")
        print("  python quick_fix_spygate_clips.py          # Apply fix")
        print("  python quick_fix_spygate_clips.py --verify # Check if applied")
        print("  python quick_fix_spygate_clips.py --revert # Revert to backup")
        return 0
    else:
        print("\n❌ Failed to apply fix")
        return 1


if __name__ == "__main__":
    sys.exit(main())
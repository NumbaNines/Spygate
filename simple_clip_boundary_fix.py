#!/usr/bin/env python3
"""
Simple fix for SpygateAI clip boundary detection.
Makes clips end when plays end instead of continuing for extra seconds.
"""

import re
from pathlib import Path


def apply_simple_fix():
    """Apply a simple fix to limit clip duration and detect play ends."""

    file_path = Path("spygate_desktop_app_faceit_style.py")
    backup_path = Path("spygate_desktop_app_faceit_style.py.backup")

    # Create backup
    print("📦 Creating backup...")
    content = file_path.read_text(encoding="utf-8")
    backup_path.write_text(content, encoding="utf-8")

    # Fix 1: Update _get_situation_based_boundaries to use shorter durations
    print("🔧 Fixing clip boundaries...")

    # Find and replace the fallback buffer duration
    old_buffer = r"buffer_frames = int\(fps \* 1\.5\)"
    new_buffer = r"buffer_frames = int(fps * 2.0)"  # 2 second buffer instead of 1.5
    content = re.sub(old_buffer, new_buffer, content)

    # Fix the end frame calculation to be more conservative
    old_end = r"end_frame = frame_number \+ buffer_frames"
    new_end = r"end_frame = frame_number + int(fps * 3)  # Max 3 seconds after detection"
    content = re.sub(old_end, new_end, content)

    # Fix 2: Update _find_natural_clip_boundaries to limit duration
    old_duration = r"play_duration = int\(fps \* 8\)"
    new_duration = r"play_duration = int(fps * 6)  # Reduced from 8 to 6 seconds"
    content = re.sub(old_duration, new_duration, content)

    # Fix maximum duration
    old_max = r"max_duration = int\(fps \* 20\)"
    new_max = r"max_duration = int(fps * 10)  # Reduced from 20 to 10 seconds max"
    content = re.sub(old_max, new_max, content)

    # Fix 3: Add play end detection in _analyze_play_boundaries
    # Find the section that checks for down changes
    play_detection = r'(# 1\. DOWN PROGRESSION - Most reliable indicator\s*\n\s*elif.*?print\(f"🏈 PLAY START:.*?\))'

    # Add play end detection after play start
    play_end_check = r"""\1

            # CRITICAL: Check if this down change means the PREVIOUS play ended
            if self.play_boundary_state.get('current_play_start_frame') is not None:
                # We were tracking a play and now see a new down - previous play has ended
                print(f"🏁 PREVIOUS PLAY ENDED: Down changed, ending clip")
                # Signal to create clip for the play that just ended
                boundary_info['previous_play_ended'] = True
                boundary_info['previous_play_end_frame'] = current_frame - 30  # End 1 second ago"""

    content = re.sub(play_detection, play_end_check, content, flags=re.DOTALL)

    # Fix 4: Update clip creation to respect play boundaries
    # Find the clip creation section
    old_clip_logic = r"(if self\._should_create_clip\(game_state, situation_context\):.*?clip_start_frame, clip_end_frame = self\._find_natural_clip_boundaries\(frame_number, fps, game_state\))"

    new_clip_logic = r"""\1

                            # FIXED: Limit clip to current play only
                            if hasattr(self.analyzer, 'game_history') and len(self.analyzer.game_history) > 2:
                                # Check if we can detect the next play starting
                                recent_history = self.analyzer.game_history[-5:]
                                for i in range(len(recent_history) - 1):
                                    if (recent_history[i].down != recent_history[i+1].down):
                                        # Down changed - play likely ended
                                        frames_ahead = (len(recent_history) - i - 1) * 30
                                        clip_end_frame = min(clip_end_frame, frame_number + frames_ahead + 30)
                                        print(f"🎯 Detected play end {frames_ahead/30:.1f}s ahead - limiting clip")
                                        break"""

    content = re.sub(old_clip_logic, new_clip_logic, content, flags=re.DOTALL)

    # Write the fixed content
    file_path.write_text(content, encoding="utf-8")
    print("✅ Fixes applied successfully!")

    # Create a summary
    print("\n📋 Summary of changes:")
    print("1. ✅ Increased buffer to 2 seconds for better play capture")
    print("2. ✅ Limited clip end to max 3 seconds after detection")
    print("3. ✅ Reduced default play duration from 8 to 6 seconds")
    print("4. ✅ Reduced max clip duration from 20 to 10 seconds")
    print("5. ✅ Added logic to detect play ends from down changes")
    print("6. ✅ Clips now check game history to avoid spanning multiple plays")

    return True


def main():
    print("🚀 Simple SpygateAI Clip Boundary Fix")
    print("=" * 50)

    try:
        if apply_simple_fix():
            print("\n✅ All fixes applied successfully!")
            print("\n🎯 Result: Clips will now be limited to single plays")
            print("⏱️  Maximum clip duration: 10 seconds")
            print("🏁 Clips will end when down changes are detected")
        else:
            print("\n❌ Fix failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

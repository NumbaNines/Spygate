#!/usr/bin/env python3
"""
Apply Clip Segmentation Fix to Spygate
=====================================

This script patches the existing Spygate desktop app to use the enhanced
clip segmentation with YOLO-based play start detection.
"""

import os
import sys
from pathlib import Path

def apply_fix():
    """Apply the clip segmentation fix to the desktop app."""
    
    # Find the desktop app file
    app_file = Path(__file__).parent / "spygate_desktop_app_faceit_style.py"
    if not app_file.exists():
        print(f"❌ Error: Could not find {app_file}")
        return False
    
    print("📝 Reading desktop app file...")
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_file = app_file.with_suffix('.py.backup')
    print(f"💾 Creating backup at {backup_file}")
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Find the _analyze_play_boundaries method and replace it
    print("🔧 Patching _analyze_play_boundaries method...")
    
    # Add import at the top
    import_line = "from spygate_clip_segmentation_fix import create_enhanced_play_boundary_analyzer\n"
    if "spygate_clip_segmentation_fix" not in content:
        # Find imports section
        import_pos = content.find("import time\n")
        if import_pos > 0:
            content = content[:import_pos] + import_line + content[import_pos:]
    
    # Add the enhanced analyzer initialization in __init__
    init_patch = """
        # Enhanced play boundary analyzer for YOLO-based clip detection
        self._play_boundary_analyzer = create_enhanced_play_boundary_analyzer()
        self._last_detections = []  # Store detections for boundary analysis
"""
    
    # Find the __init__ method of AnalysisWorker
    init_pos = content.find("        self.previous_game_state = None")
    if init_pos > 0:
        content = content[:init_pos] + init_patch + "\n" + content[init_pos:]
    
    # Create the new _analyze_play_boundaries method
    new_method = '''
    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:
        """
        ENHANCED: Uses YOLO indicators for clip start, game state for clip end.
        
        This method now:
        1. Uses preplay_indicator/play_call_screen for precise play start detection
        2. Uses down changes and other game events for play end detection
        3. Creates clips that contain exactly one play each
        """
        
        # Get YOLO detections - need to get these from the analyzer
        detections = self._last_detections if hasattr(self, '_last_detections') else []
        
        # Use enhanced analyzer
        result = self._play_boundary_analyzer.analyze_frame(current_frame, game_state, detections)
        
        # Convert to expected format
        boundary_info = {
            "play_started": result["play_started"],
            "play_ended": result["play_ended"],
            "play_in_progress": self._play_boundary_analyzer.play_active,
            "clip_should_start": False,
            "clip_should_end": False,
            "recommended_clip_start": result["clip_start_frame"],
            "recommended_clip_end": result["clip_end_frame"],
            "play_type": result["clip_reason"] or "unknown",
            "play_situation": "normal",
            "confidence": 0.9,
            "create_immediate_clip": result["create_clip"],
            "debug_info": result["debug_info"]
        }
        
        # Log debug info periodically
        if current_frame % 150 == 0:
            debug = result["debug_info"]
            print(f"🔍 Boundary Analysis - Frame {current_frame}:")
            print(f"   YOLO: preplay={debug['has_preplay']}, playcall={debug['has_playcall']}")
            print(f"   State: play_active={debug['play_active']}, clips_created={debug['clips_created_count']}")
        
        return boundary_info
'''
    
    # Find and replace the existing method
    method_start = content.find("    def _analyze_play_boundaries(self, game_state, current_frame: int) -> dict:")
    if method_start > 0:
        # Find the end of the method (next def at same indentation)
        method_end = content.find("\n    def ", method_start + 1)
        if method_end > 0:
            print("✅ Found existing _analyze_play_boundaries method")
            content = content[:method_start] + new_method + content[method_end:]
        else:
            print("⚠️  Could not find end of _analyze_play_boundaries method")
    
    # Modify analyze_frame call to store detections
    print("🔧 Patching analyze_frame calls to store detections...")
    
    # Add detection storage after analyze_frame calls
    analyze_patch = """
                    # Store detections for boundary analysis
                    if hasattr(self.analyzer, 'last_detections'):
                        self._last_detections = self.analyzer.last_detections
                    elif hasattr(self.analyzer, 'model') and hasattr(self.analyzer.model, 'last_detections'):
                        self._last_detections = self.analyzer.model.last_detections
                    else:
                        self._last_detections = []
"""
    
    # Find analyze_frame calls and add detection storage
    analyze_pos = content.find("game_state = self.analyzer.analyze_frame(")
    while analyze_pos > 0:
        # Find the end of the statement
        stmt_end = content.find("\n", analyze_pos)
        if stmt_end > 0:
            # Check if we already added the patch
            if "Store detections for boundary" not in content[stmt_end:stmt_end+200]:
                content = content[:stmt_end+1] + analyze_patch + content[stmt_end+1:]
        analyze_pos = content.find("game_state = self.analyzer.analyze_frame(", analyze_pos + 1)
    
    # Modify clip creation to use boundary info
    print("🔧 Patching clip creation logic...")
    
    # Find the clip creation section
    clip_create_pos = content.find("if self._should_create_clip(game_state, situation_context):")
    if clip_create_pos > 0:
        # Add check for immediate clip creation from boundary analysis
        immediate_check = """
                        # Check if boundary analysis says to create immediate clip
                        if boundary_info.get("create_immediate_clip", False):
                            print(f"🎬 IMMEDIATE CLIP from boundary analysis at frame {frame_number}")
                            clip_start = boundary_info["recommended_clip_start"]
                            clip_end = boundary_info["recommended_clip_end"]
                            
                            if clip_start is not None and clip_end is not None:
                                # Create the clip
                                clip = self._create_enhanced_clip_with_boundaries(
                                    clip_start,
                                    clip_end,
                                    fps,
                                    game_state,
                                    situation_context,
                                    boundary_info,
                                )
                                detected_clips.append(clip)
                                self.clip_detected.emit(clip)
                                
                                # Log the detection
                                self._log_enhanced_detection_with_boundaries(
                                    game_state, situation_context, frame_number, boundary_info
                                )
                                
                                # Skip the normal clip creation flow
                                continue
                        
"""
        # Insert before the existing if statement
        content = content[:clip_create_pos] + immediate_check + " "*24 + content[clip_create_pos:]
    
    # Write the patched file
    print("💾 Writing patched file...")
    with open(app_file, 'w') as f:
        f.write(content)
    
    print("✅ Patch applied successfully!")
    print(f"📋 Backup saved to: {backup_file}")
    print("\n🎯 Next steps:")
    print("1. Make sure spygate_clip_segmentation_fix.py is in the same directory")
    print("2. Run the desktop app - it will now use YOLO indicators for clip starts")
    print("3. Monitor the console output for debug messages")
    
    return True


def verify_fix():
    """Verify that the fix was applied correctly."""
    app_file = Path(__file__).parent / "spygate_desktop_app_faceit_style.py"
    
    with open(app_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("Import added", "spygate_clip_segmentation_fix" in content),
        ("Analyzer initialized", "_play_boundary_analyzer" in content),
        ("Detections stored", "Store detections for boundary" in content),
        ("Immediate clip check", "create_immediate_clip" in content),
    ]
    
    print("\n🔍 Verification Results:")
    all_good = True
    for check_name, result in checks:
        if result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    print("=" * 50)
    print("Spygate Clip Segmentation Fix Patcher")
    print("=" * 50)
    
    if apply_fix():
        verify_fix()
    else:
        print("❌ Failed to apply fix")
        sys.exit(1)
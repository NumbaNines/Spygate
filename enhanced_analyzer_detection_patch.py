#!/usr/bin/env python3
"""
Enhanced Game Analyzer Detection Storage Patch
=============================================

This patch modifies the enhanced_game_analyzer to store YOLO detections
so they can be accessed by the play boundary analyzer.
"""

import os
from pathlib import Path


def create_patched_analyze_frame():
    """
    Create a patched version of analyze_frame that stores detections.
    """
    
    patch_content = '''
# Add this to enhanced_game_analyzer.py after the imports

def patch_analyzer_for_detection_storage(analyzer_class):
    """
    Patches the EnhancedGameAnalyzer to store YOLO detections for boundary analysis.
    """
    
    # Store original methods
    original_analyze_frame_normal = analyzer_class._analyze_frame_normal
    original_analyze_frame_fresh = analyzer_class._analyze_frame_fresh_ocr
    
    def patched_analyze_frame_normal(self, frame, current_time=None, frame_number=None):
        """Normal frame analysis that also stores detections."""
        try:
            # Get detections
            detections = self.model.detect(frame)
            
            # Store detections for boundary analyzer
            self.last_detections = []
            if detections:
                for det in detections:
                    self.last_detections.append({
                        "class": det.class_name if hasattr(det, 'class_name') else None,
                        "confidence": det.confidence if hasattr(det, 'confidence') else 0.0,
                        "bbox": det.bbox if hasattr(det, 'bbox') else None
                    })
            
            # Log YOLO indicators periodically
            if frame_number and frame_number % 150 == 0:
                has_preplay = any(d["class"] == "preplay_indicator" for d in self.last_detections)
                has_playcall = any(d["class"] == "play_call_screen" for d in self.last_detections)
                if has_preplay or has_playcall:
                    print(f"🎯 YOLO Indicators at frame {frame_number}: preplay={has_preplay}, playcall={has_playcall}")
            
            # Continue with original analysis
            result = original_analyze_frame_normal(self, frame, current_time, frame_number)
            
            # Add detections to game state if available
            if result and hasattr(result, '__dict__'):
                result.detections = self.last_detections
            
            return result
            
        except Exception as e:
            import traceback
            print(f"❌ Error in patched analyze_frame: {e}")
            traceback.print_exc()
            # Fallback to original
            return original_analyze_frame_normal(self, frame, current_time, frame_number)
    
    def patched_analyze_frame_fresh(self, frame, current_time=None, frame_number=None):
        """Fresh OCR analysis that also stores detections."""
        try:
            # Get detections
            detections = self.model.detect(frame)
            
            # Store detections
            self.last_detections = []
            if detections:
                for det in detections:
                    self.last_detections.append({
                        "class": det.class_name if hasattr(det, 'class_name') else None,
                        "confidence": det.confidence if hasattr(det, 'confidence') else 0.0,
                        "bbox": det.bbox if hasattr(det, 'bbox') else None
                    })
            
            # Continue with original analysis
            result = original_analyze_frame_fresh(self, frame, current_time, frame_number)
            
            # Add detections to game state
            if result and hasattr(result, '__dict__'):
                result.detections = self.last_detections
            
            return result
            
        except Exception as e:
            # Fallback to original
            return original_analyze_frame_fresh(self, frame, current_time, frame_number)
    
    # Apply patches
    analyzer_class._analyze_frame_normal = patched_analyze_frame_normal
    analyzer_class._analyze_frame_fresh_ocr = patched_analyze_frame_fresh
    
    print("✅ Enhanced Game Analyzer patched for detection storage")
    
    return analyzer_class

'''
    
    return patch_content


def apply_analyzer_patch():
    """Apply the detection storage patch to enhanced_game_analyzer.py"""
    
    analyzer_file = Path(__file__).parent / "src" / "spygate" / "ml" / "enhanced_game_analyzer.py"
    
    if not analyzer_file.exists():
        print(f"❌ Error: Could not find {analyzer_file}")
        return False
    
    print("📝 Reading enhanced_game_analyzer.py...")
    with open(analyzer_file, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_file = analyzer_file.with_suffix('.py.backup')
    print(f"💾 Creating backup at {backup_file}")
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Check if patch already applied
    if "patch_analyzer_for_detection_storage" in content:
        print("⚠️  Patch already applied")
        return True
    
    # Add the patch function after the class definition
    patch_code = create_patched_analyze_frame()
    
    # Find a good place to insert (after the last method of EnhancedGameAnalyzer)
    class_end = content.rfind("class EnhancedGameAnalyzer")
    if class_end > 0:
        # Find the end of the class (next class or end of file)
        next_class = content.find("\nclass ", class_end + 1)
        if next_class > 0:
            insert_pos = next_class
        else:
            insert_pos = len(content)
        
        # Insert the patch
        content = content[:insert_pos] + "\n\n" + patch_code + "\n" + content[insert_pos:]
    
    # Write the patched file
    print("💾 Writing patched analyzer...")
    with open(analyzer_file, 'w') as f:
        f.write(content)
    
    print("✅ Enhanced Game Analyzer patched successfully!")
    
    # Create a simple wrapper to apply the patch when the analyzer is loaded
    wrapper_file = Path(__file__).parent / "analyzer_patch_loader.py"
    wrapper_content = '''
"""Auto-apply detection storage patch when loading the analyzer."""

from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer, patch_analyzer_for_detection_storage

# Apply the patch
EnhancedGameAnalyzer = patch_analyzer_for_detection_storage(EnhancedGameAnalyzer)

print("✅ EnhancedGameAnalyzer patched for YOLO detection storage")
'''
    
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_content)
    
    print(f"📝 Created patch loader at {wrapper_file}")
    
    return True


def create_simple_integration():
    """Create a simple integration file that ties everything together."""
    
    integration_content = '''#!/usr/bin/env python3
"""
Spygate Clip Segmentation Fix - Simple Integration
=================================================

This file provides a simple way to integrate the clip segmentation fix
into your existing Spygate application.
"""

from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from spygate_clip_segmentation_fix import create_enhanced_play_boundary_analyzer


def integrate_clip_fix_simple(analysis_worker):
    """
    Simple integration that adds the enhanced boundary analyzer to an AnalysisWorker.
    
    Args:
        analysis_worker: The AnalysisWorker instance to enhance
    """
    
    # Create and attach the enhanced analyzer
    analysis_worker._play_boundary_analyzer = create_enhanced_play_boundary_analyzer()
    analysis_worker._last_detections = []
    
    # Store original analyze_play_boundaries
    original_method = analysis_worker._analyze_play_boundaries
    
    def enhanced_analyze_play_boundaries(game_state, current_frame):
        """Enhanced version using YOLO indicators."""
        
        # Get detections from game state or analyzer
        detections = []
        if hasattr(game_state, 'detections'):
            detections = game_state.detections
        elif hasattr(analysis_worker, '_last_detections'):
            detections = analysis_worker._last_detections
        elif hasattr(analysis_worker.analyzer, 'last_detections'):
            detections = analysis_worker.analyzer.last_detections
            analysis_worker._last_detections = detections
        
        # Use enhanced analyzer
        result = analysis_worker._play_boundary_analyzer.analyze_frame(
            current_frame, game_state, detections
        )
        
        # If we should create a clip immediately, override the normal flow
        if result["create_clip"]:
            print(f"🎬 YOLO-TRIGGERED CLIP: {result['clip_start_frame']}-{result['clip_end_frame']}")
            
            # Create a boundary info that will trigger immediate clip creation
            return {
                "play_started": result["play_started"],
                "play_ended": result["play_ended"],
                "play_in_progress": analysis_worker._play_boundary_analyzer.play_active,
                "clip_should_start": False,
                "clip_should_end": True,  # This triggers clip creation
                "recommended_clip_start": result["clip_start_frame"],
                "recommended_clip_end": result["clip_end_frame"],
                "play_type": result["clip_reason"] or "yolo_triggered",
                "play_situation": "normal",
                "confidence": 0.95,
            }
        
        # Otherwise use original method
        return original_method(game_state, current_frame)
    
    # Replace the method
    analysis_worker._analyze_play_boundaries = enhanced_analyze_play_boundaries
    
    print("✅ Clip segmentation fix integrated!")
    print("🎯 Now using YOLO indicators (preplay_indicator, play_call_screen) for clip starts")
    print("🏁 Still using game state changes (down changes, etc.) for clip ends")
    
    return analysis_worker


# Example usage
if __name__ == "__main__":
    print("This module provides integrate_clip_fix_simple() function")
    print("Use it to enhance your AnalysisWorker with YOLO-based clip detection")
'''
    
    integration_file = Path(__file__).parent / "integrate_clip_fix.py"
    with open(integration_file, 'w') as f:
        f.write(integration_content)
    
    print(f"📝 Created simple integration at {integration_file}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Game Analyzer Detection Storage Patch")
    print("=" * 60)
    
    if apply_analyzer_patch():
        create_simple_integration()
        print("\n✅ All patches applied successfully!")
        print("\n📋 Integration Instructions:")
        print("1. In your main application, import the integration:")
        print("   from integrate_clip_fix import integrate_clip_fix_simple")
        print("2. After creating your AnalysisWorker, enhance it:")
        print("   analysis_worker = integrate_clip_fix_simple(analysis_worker)")
        print("3. The system will now use YOLO indicators for precise clip detection!")
    else:
        print("❌ Failed to apply patches")
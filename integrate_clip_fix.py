#!/usr/bin/env python3
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
    print("\nExample:")
    print("  from integrate_clip_fix import integrate_clip_fix_simple")
    print("  ")
    print("  # After creating your AnalysisWorker")
    print("  analysis_worker = AnalysisWorker(video_path, preferences)")
    print("  analysis_worker = integrate_clip_fix_simple(analysis_worker)")
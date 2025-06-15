#!/usr/bin/env python3
"""
Test script for SpygateAI Debug Analyzer
This script demonstrates how to use the debug analyzer with sample data
"""

import json
import os
from datetime import datetime


def create_sample_debug_data():
    """Create sample debug data files for testing"""

    # Sample clips data based on the actual console log output
    clips_data = [
        {
            "index": 0,
            "situation": "Unknown Down & Distance",
            "start_time": 2.5,
            "end_time": 5.0,
            "start_frame": 75,
            "end_frame": 150,
            "confidence": 0.3,
            "play_type": "unknown",
            "expected_situation": "2nd & 10",
            "frame_number": 112,
        },
        {
            "index": 1,
            "situation": "2nd & 10",
            "start_time": 6.0,
            "end_time": 12.0,
            "start_frame": 180,
            "end_frame": 360,
            "confidence": 0.85,
            "play_type": "normal",
            "expected_situation": "2nd & 10",
            "frame_number": 270,
        },
        {
            "index": 2,
            "situation": "2nd & 10",
            "start_time": 22.0,
            "end_time": 28.0,
            "start_frame": 660,
            "end_frame": 840,
            "confidence": 0.78,
            "play_type": "normal",
            "expected_situation": "2nd & 10",
            "frame_number": 750,
        },
        {
            "index": 3,
            "situation": "2nd & 10",
            "start_time": 38.0,
            "end_time": 44.0,
            "start_frame": 1140,
            "end_frame": 1320,
            "confidence": 0.82,
            "play_type": "normal",
            "expected_situation": "2nd & 10",
            "frame_number": 1230,
        },
        {
            "index": 4,
            "situation": "2nd & 10",
            "start_time": 54.0,
            "end_time": 60.0,
            "start_frame": 1620,
            "end_frame": 1800,
            "confidence": 0.91,
            "play_type": "normal",
            "expected_situation": "2nd & 10",
            "frame_number": 1710,
        },
        {
            "index": 5,
            "situation": "3rd & 3 (Big Play)",
            "start_time": 70.0,
            "end_time": 82.0,
            "start_frame": 2100,
            "end_frame": 2460,
            "confidence": 0.92,
            "play_type": "big_play",
            "expected_situation": "3rd & 3",
            "frame_number": 2280,
        },
        {
            "index": 6,
            "situation": "1st & 10",
            "start_time": 86.0,
            "end_time": 92.0,
            "start_frame": 2580,
            "end_frame": 2760,
            "confidence": 0.89,
            "play_type": "normal",
            "expected_situation": "1st & 10",
            "frame_number": 2670,
        },
        {
            "index": 20,
            "situation": "2nd & 10",
            "start_time": 302.0,
            "end_time": 317.0,
            "start_frame": 18120,
            "end_frame": 19020,
            "confidence": 0.87,
            "play_type": "turnover_recovery",
            "expected_situation": "2nd & 10",
            "frame_number": 18720,
            "game_state": {
                "down": 2,
                "distance": 10,
                "quarter": 4,
                "game_clock": "08:24",
                "possession": "away_team",
            },
            "ocr_results": {
                "down_distance_area": {
                    "raw_text": "2nd & 10",
                    "parsed_value": "2nd & 10",
                    "confidence": 0.87,
                },
                "game_clock_area": {
                    "raw_text": "08:24",
                    "parsed_value": "08:24",
                    "confidence": 0.93,
                },
            },
            "yolo_detections": [
                {"class": "down_distance_area", "confidence": 0.853, "bbox": [120, 50, 200, 80]},
                {
                    "class": "possession_triangle_area",
                    "confidence": 0.91,
                    "bbox": [300, 45, 350, 85],
                },
            ],
        },
    ]

    # Save clips data
    with open("debug_clips_data.json", "w", encoding="utf-8") as f:
        json.dump(clips_data, f, indent=2, ensure_ascii=False)

    # Sample analysis logs based on the actual console output
    log_lines = [
        "CLIP REGISTERED: fourth_quarter (18120-19020) - Total tracked: 7",
        "EMITTING CLIP SIGNAL for fourth_quarter",
        "BOUNDARY-AWARE Detection at 624.0s (Frame 18720):",
        "FINAL CLIP RECEIVED IN DESKTOP APP:",
        "CLIP FORMATTING DEBUG: Down=2, Distance=10, Formatted='2nd & 10'",
        "New Play Detected: turnover_recovery",
        "Play Ended",
        "NO YOLO DETECTIONS in this frame",
        "STATE PERSISTENCE: Preserved down=2 from previous frame (9/10)",
        "STATE PERSISTENCE: Preserved distance=10 from previous frame (9/10)",
        "STATE PERSISTENCE: Preserved quarter=4 from previous frame (19/60)",
        "STATE PERSISTENCE: Preserved time=08:24 from previous frame (3/30)",
        "STARTING FULL PLAY TRACKING: turnover_recovery at frame 18720",
        "USING NATURAL BOUNDARIES: 18360 → 19260 (15.0s)",
        "DUPLICATE DETECTED: 73.3% overlap with existing clip",
        "SKIPPED DUPLICATE CLIP at frame 18960",
        "ANALYSIS COMPLETE - RECEIVED 21 CLIPS FROM WORKER:",
        "   Clip 1: 'Unknown Down & Distance' at 2.5s",
        "   Clip 2: '2nd & 10' at 6.0s",
        "   Clip 3: '2nd & 10' at 22.0s",
        "   Clip 4: '2nd & 10' at 38.0s",
        "   Clip 5: '2nd & 10' at 54.0s",
        "   Clip 6: '3rd & 3 (Big Play)' at 70.0s",
        "   Clip 7: '1st & 10' at 86.0s",
        "   Clip 21: '2nd & 10' at 302.0s",
    ]

    # Save analysis logs
    with open("analysis_logs.txt", "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    print("Sample debug data created:")
    print("- debug_clips_data.json (5 sample clips)")
    print("- analysis_logs.txt (24 sample log lines)")
    print("\nTo test real-time monitoring, save console output to a text file")
    print("and point the debug analyzer to that file.")


def save_console_log_template():
    """Create a template for saving console logs"""
    console_log_content = """
# Save your SpygateAI console output to this file for real-time monitoring
# Example usage:
# 1. Run your SpygateAI analysis
# 2. Copy the console output to this file
# 3. Use the debug analyzer's "Browse..." button to select this file
# 4. Click "Start Real-time Monitoring"

# The debug analyzer will automatically parse lines like:
# ✅ CLIP REGISTERED: situation_name (start_frame-end_frame) - Total tracked: N
# 🔍 CLIP FORMATTING DEBUG: Down=X, Distance=Y, Formatted='text'
# 🔄 STATE PERSISTENCE: Preserved property=value from previous frame (count/max)
# 🚫 DUPLICATE DETECTED: X.X% overlap with existing clip
# 🆕 New Play Detected: play_type
# ❌ NO YOLO DETECTIONS in this frame
# 🎯 BOUNDARY-AWARE Detection at Xs (Frame N):
# 🚨 ANALYSIS COMPLETE - RECEIVED N CLIPS FROM WORKER:

# Paste your console output below this line:
    """

    with open("console_log_template.txt", "w", encoding="utf-8") as f:
        f.write(console_log_content)

    print("Created console_log_template.txt")
    print("Use this file to save your SpygateAI console output for analysis")


def main():
    print("SpygateAI Debug Analyzer Test Setup")
    print("=" * 50)

    # Create sample data
    create_sample_debug_data()
    save_console_log_template()

    print("\nNext steps:")
    print("1. Run: python debug_clip_analyzer.py")
    print("2. Click 'Load Analysis Data' to load sample data")
    print("3. Try the filters to analyze different types of issues:")
    print("   - 'Show Only Incorrect' to see problematic clips")
    print("   - 'Show Only Duplicates' to see overlap issues")
    print("   - 'Show State Persistence Issues' to see OCR fallback patterns")
    print("4. For real-time monitoring:")
    print("   - Save SpygateAI console output to a text file")
    print("   - Use 'Browse...' to select the file")
    print("   - Click 'Start Real-time Monitoring'")

    print("\nKey features of the debug analyzer:")
    print("- Real-time log parsing and clip detection")
    print("- Automated suspicious clip identification")
    print("- Enhanced filtering for different issue types")
    print("- Statistics tracking for analysis patterns")
    print("- Memory-based improvements for known SpygateAI issues")

    print("\nCommon issues the analyzer can detect:")
    print("- State persistence overuse (OCR failures)")
    print("- Duplicate clip detection")
    print("- 'Unknown Down & Distance' OCR problems")
    print("- Turnover recovery misclassification")
    print("- Excessive '1st & 10' repetition")


if __name__ == "__main__":
    main()

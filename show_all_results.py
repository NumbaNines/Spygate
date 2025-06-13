#!/usr/bin/env python3
"""
Show All Template Matching Results
Displays information about each visualization file
"""

import os
import cv2
from pathlib import Path

def main():
    print("🎯 SPYGATE TEMPLATE MATCHING - ALL RESULTS")
    print("=" * 60)
    
    # Find all visualization files
    vis_files = []
    for i in range(1, 26):
        pattern = f"template_test_result_{i:02d}_*.jpg"
        matches = list(Path(".").glob(pattern))
        if matches:
            vis_files.append((i, matches[0]))
    
    print(f"📁 Found {len(vis_files)} visualization files")
    print()
    
    # Based on our console output, here's what each image detected:
    detections_summary = {
        1: {"yolo": 1, "templates": 3, "main": "Territory UP triangles (▲)"},
        2: {"yolo": 3, "templates": 11, "main": "Possession RIGHT + Territory DOWN"},
        3: {"yolo": 4, "templates": 11, "main": "Possession RIGHT + Territory DOWN"},
        4: {"yolo": 3, "templates": 8, "main": "Territory UP + Possession LEFT"},
        5: {"yolo": 4, "templates": 7, "main": "Possession RIGHT + Territory UP"},
        6: {"yolo": 3, "templates": 6, "main": "Territory UP + Possession LEFT"},
        7: {"yolo": 3, "templates": 13, "main": "Possession RIGHT + Territory DOWN"},
        8: {"yolo": 3, "templates": 12, "main": "Possession RIGHT + Territory DOWN"},
        9: {"yolo": 3, "templates": 7, "main": "Possession RIGHT + Territory UP"},
        10: {"yolo": 3, "templates": 9, "main": "Possession RIGHT + Territory DOWN"},
        11: {"yolo": 3, "templates": 8, "main": "Possession RIGHT + Territory DOWN"},
        12: {"yolo": 4, "templates": 8, "main": "Possession RIGHT + Territory DOWN"},
        13: {"yolo": 3, "templates": 10, "main": "Territory UP + Possession RIGHT"},
        14: {"yolo": 3, "templates": 12, "main": "Possession RIGHT + Territory DOWN"},
        15: {"yolo": 3, "templates": 8, "main": "Possession RIGHT + Territory UP"},
        16: {"yolo": 0, "templates": 0, "main": "No detections (menu/transition)"},
        17: {"yolo": 3, "templates": 10, "main": "Territory UP + Possession RIGHT"},
        18: {"yolo": 4, "templates": 15, "main": "Possession RIGHT + Territory DOWN"},
        19: {"yolo": 3, "templates": 10, "main": "Possession RIGHT + Territory UP"},
        20: {"yolo": 1, "templates": 0, "main": "YOLO only, no template matches"},
        21: {"yolo": 3, "templates": 6, "main": "Possession LEFT + Territory DOWN"},
        22: {"yolo": 3, "templates": 8, "main": "Territory UP + Possession LEFT"},
        23: {"yolo": 0, "templates": 0, "main": "No detections (menu/transition)"},
        24: {"yolo": 3, "templates": 12, "main": "Possession RIGHT + Territory DOWN"},
        25: {"yolo": 3, "templates": 8, "main": "Possession LEFT + Territory DOWN"}
    }
    
    for i, vis_file in vis_files:
        summary = detections_summary.get(i, {"yolo": "?", "templates": "?", "main": "Unknown"})
        
        print(f"📸 Image {i:2d}: {vis_file.name}")
        print(f"   🎯 YOLO: {summary['yolo']} detections")
        print(f"   🔍 Templates: {summary['templates']} matches")
        print(f"   📋 Main findings: {summary['main']}")
        print(f"   🖼️  File: {vis_file}")
        
        # Check file size
        if vis_file.exists():
            size_kb = vis_file.stat().st_size // 1024
            print(f"   📊 Size: {size_kb}KB")
        
        print()
    
    print("=" * 60)
    print("🔍 WHAT TO LOOK FOR IN THE IMAGES:")
    print("=" * 60)
    print("🟢 GREEN BOXES = YOLO HUD detections (main game info bar)")
    print("🔵 BLUE BOXES = YOLO possession triangle areas (left side)")
    print("🔴 RED BOXES = YOLO territory triangle areas (right side)")
    print("🟡 YELLOW BOXES = Template matches (actual triangles found!)")
    print()
    print("📍 POSSESSION TRIANGLES:")
    print("   → Right arrow = Team on right has ball")
    print("   ← Left arrow = Team on left has ball")
    print()
    print("🗺️  TERRITORY TRIANGLES:")
    print("   ▲ Up triangle = In opponent's territory")
    print("   ▼ Down triangle = In own territory")
    print()
    print("✨ SUCCESS INDICATORS:")
    print("   🎯 Yellow boxes inside blue/red YOLO boxes = Perfect!")
    print("   📊 High confidence scores (0.8-0.999) = Excellent!")
    print("   🔥 Multiple template matches = System working!")
    print()
    print("📁 TO VIEW THE IMAGES:")
    print("   Open any template_test_result_*.jpg file")
    print("   Look for the colored bounding boxes")
    print("   Yellow boxes show the actual triangles found!")

if __name__ == "__main__":
    main() 
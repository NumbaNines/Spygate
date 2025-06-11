#!/usr/bin/env python3
"""
Correct coordinate regions visualization using the working grid system
"""

import cv2
import numpy as np
from pathlib import Path

def visualize_correct_regions():
    """Visualize coordinate regions using the correct grid system."""
    
    # Load the HUD image
    hud_image_path = "found_and_frame_3000.png"
    
    if not Path(hud_image_path).exists():
        print(f"❌ Image not found: {hud_image_path}")
        return
    
    # Load image
    image = cv2.imread(hud_image_path)
    if image is None:
        print(f"❌ Failed to load image: {hud_image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Create visualization copy
    viz_image = image.copy()
    
    # USER SPECIFICATIONS (converted from grid coordinates):
    # Team scores: vertically 0-2, horizontally 0-11 
    # Grid system: 0.1 intervals, so 0-2 = 0.0-0.2, 0-11 = 0.0-1.0
    team_scores_coords = {
        'x_start': 0.0,      # Grid column 0
        'x_end': 1.0,        # Grid column 10 (full width, since 11 would be beyond 1.0)
        'y_start': 0.0,      # Grid row 0  
        'y_end': 0.2         # Grid row 2
    }
    
    # Possession triangle: vertically 1, horizontally 5-6
    # Grid system: 1 = 0.1-0.2, 5-6 = 0.5-0.6
    possession_coords = {
        'x_start': 0.5,      # Grid column 5
        'x_end': 0.6,        # Grid column 6
        'y_start': 0.1,      # Grid row 1
        'y_end': 0.2         # Grid row 2 (giving height to row 1)
    }
    
    # Convert to pixel coordinates using the same method as working down/distance
    def coords_to_pixels(coords):
        return {
            'x1': int(coords['x_start'] * width),
            'x2': int(coords['x_end'] * width),
            'y1': int(coords['y_start'] * height),
            'y2': int(coords['y_end'] * height)
        }
    
    team_pixels = coords_to_pixels(team_scores_coords)
    poss_pixels = coords_to_pixels(possession_coords)
    
    # Also show existing down/distance for reference
    down_coords = {'x_start': 0.750, 'x_end': 0.900, 'y_start': 0.200, 'y_end': 0.800}
    down_pixels = coords_to_pixels(down_coords)
    
    print(f"\n📍 TEAM SCORES REGION (User spec: vertically 0-2, horizontally 0-11):")
    print(f"   Normalized: {team_scores_coords}")
    print(f"   Pixels: {team_pixels}")
    
    print(f"\n🔺 POSSESSION TRIANGLE (User spec: vertically 1, horizontally 5-6):")
    print(f"   Normalized: {possession_coords}")
    print(f"   Pixels: {poss_pixels}")
    
    print(f"\n📏 DOWN/DISTANCE (For reference - working region):")
    print(f"   Normalized: {down_coords}")
    print(f"   Pixels: {down_pixels}")
    
    # Draw Team Scores region (GREEN - thick border)
    cv2.rectangle(viz_image, 
                  (team_pixels['x1'], team_pixels['y1']),
                  (team_pixels['x2'], team_pixels['y2']),
                  (0, 255, 0), 4)  # Green thick outline
    
    # Draw Possession region (BLUE - thick border)
    cv2.rectangle(viz_image,
                  (poss_pixels['x1'], poss_pixels['y1']),
                  (poss_pixels['x2'], poss_pixels['y2']),
                  (255, 0, 0), 4)  # Blue thick outline
    
    # Draw Down/Distance for reference (RED - thin border)
    cv2.rectangle(viz_image,
                  (down_pixels['x1'], down_pixels['y1']),
                  (down_pixels['x2'], down_pixels['y2']),
                  (0, 0, 255), 2)  # Red thin outline
    
    # Add labels with background for visibility
    def add_label(img, text, pos, color):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (pos[0]-2, pos[1]-text_height-5), 
                     (pos[0]+text_width+2, pos[1]+5), (0, 0, 0), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    add_label(viz_image, "TEAM SCORES (0-11, 0-2)", 
              (team_pixels['x1'], team_pixels['y1'] - 10), (0, 255, 0))
    
    add_label(viz_image, "POSSESSION (5-6, 1)", 
              (poss_pixels['x1'], poss_pixels['y1'] - 10), (255, 0, 0))
    
    add_label(viz_image, "DOWN/DIST (working)", 
              (down_pixels['x1'], down_pixels['y1'] - 10), (0, 0, 255))
    
    # Add coordinate info at bottom
    coord_info = [
        f"Team Scores: x({team_scores_coords['x_start']:.1f}-{team_scores_coords['x_end']:.1f}) y({team_scores_coords['y_start']:.1f}-{team_scores_coords['y_end']:.1f})",
        f"Possession: x({possession_coords['x_start']:.1f}-{possession_coords['x_end']:.1f}) y({possession_coords['y_start']:.1f}-{possession_coords['y_end']:.1f})",
        f"Down/Dist: x({down_coords['x_start']:.3f}-{down_coords['x_end']:.3f}) y({down_coords['y_start']:.1f}-{down_coords['y_end']:.1f})"
    ]
    
    # Black background for text
    cv2.rectangle(viz_image, (5, height-85), (width-5, height-5), (0, 0, 0), -1)
    
    for i, text in enumerate(coord_info):
        cv2.putText(viz_image, text, (10, height - 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    output_path = "correct_coordinate_regions.jpg"
    cv2.imwrite(output_path, viz_image)
    print(f"\n✅ Correct visualization saved: {output_path}")
    
    # Create individual crops for verification
    team_crop = image[team_pixels['y1']:team_pixels['y2'],
                     team_pixels['x1']:team_pixels['x2']]
    cv2.imwrite("team_scores_correct_crop.jpg", team_crop)
    print(f"✅ Team scores crop: team_scores_correct_crop.jpg")
    
    poss_crop = image[poss_pixels['y1']:poss_pixels['y2'],
                     poss_pixels['x1']:poss_pixels['x2']]
    cv2.imwrite("possession_correct_crop.jpg", poss_crop)
    print(f"✅ Possession crop: possession_correct_crop.jpg")
    
    return team_scores_coords, possession_coords

if __name__ == "__main__":
    print("🎯 Creating CORRECT coordinate regions based on user specifications...")
    print("   Team Scores: vertically 0-2, horizontally 0-11")  
    print("   Possession: vertically 1, horizontally 5-6")
    print("   Using grid system where each unit = 0.1")
    
    team_coords, poss_coords = visualize_correct_regions()
    
    print(f"\n📋 SUMMARY - Coordinates in format used by working down/distance system:")
    print(f"   Team Scores: {team_coords['x_start']}, {team_coords['x_end']}, {team_coords['y_start']}, {team_coords['y_end']}")
    print(f"   Possession: {poss_coords['x_start']}, {poss_coords['x_end']}, {poss_coords['y_start']}, {poss_coords['y_end']}") 
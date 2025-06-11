#!/usr/bin/env python3
"""
Final correct coordinate regions using the 20-section grid I created
"""

import cv2
import numpy as np
from pathlib import Path

def visualize_final_correct_regions():
    """Visualize coordinate regions using the 20-section grid system."""
    
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
    
    # CORRECT COORDINATES using 20-section grid (each unit = 0.05):
    # Team scores: horizontally 0-11, vertically 0-2
    team_scores_coords = {
        'x_start': 0.0,       # Grid 0
        'x_end': 0.55,        # Grid 11 (11 × 0.05 = 0.55)
        'y_start': 0.0,       # Grid 0  
        'y_end': 0.10         # Grid 2 (2 × 0.05 = 0.10)
    }
    
    # Possession triangle: horizontally 5-6, vertically 1
    possession_coords = {
        'x_start': 0.25,      # Grid 5 (5 × 0.05 = 0.25)
        'x_end': 0.30,        # Grid 6 (6 × 0.05 = 0.30)
        'y_start': 0.05,      # Grid 1 (1 × 0.05 = 0.05)
        'y_end': 0.10         # Grid 2 (2 × 0.05 = 0.10)
    }
    
    # Convert to pixel coordinates
    def coords_to_pixels(coords):
        return {
            'x1': int(coords['x_start'] * width),
            'x2': int(coords['x_end'] * width),
            'y1': int(coords['y_start'] * height),
            'y2': int(coords['y_end'] * height)
        }
    
    team_pixels = coords_to_pixels(team_scores_coords)
    poss_pixels = coords_to_pixels(possession_coords)
    
    # Reference: working down/distance coordinates
    down_coords = {'x_start': 0.750, 'x_end': 0.900, 'y_start': 0.200, 'y_end': 0.800}
    down_pixels = coords_to_pixels(down_coords)
    
    print(f"\n📍 TEAM SCORES (Grid 0-11, 0-2 in 20-section grid):")
    print(f"   Normalized: {team_scores_coords}")
    print(f"   Pixels: {team_pixels}")
    
    print(f"\n🔺 POSSESSION TRIANGLE (Grid 5-6, 1-2 in 20-section grid):")
    print(f"   Normalized: {possession_coords}")
    print(f"   Pixels: {poss_pixels}")
    
    print(f"\n📏 DOWN/DISTANCE (Reference - working):")
    print(f"   Normalized: {down_coords}")
    print(f"   Pixels: {down_pixels}")
    
    # Draw Team Scores region (GREEN)
    cv2.rectangle(viz_image, 
                  (team_pixels['x1'], team_pixels['y1']),
                  (team_pixels['x2'], team_pixels['y2']),
                  (0, 255, 0), 4)
    
    # Draw Possession region (BLUE)
    cv2.rectangle(viz_image,
                  (poss_pixels['x1'], poss_pixels['y1']),
                  (poss_pixels['x2'], poss_pixels['y2']),
                  (255, 0, 0), 4)
    
    # Draw Down/Distance for reference (RED)
    cv2.rectangle(viz_image,
                  (down_pixels['x1'], down_pixels['y1']),
                  (down_pixels['x2'], down_pixels['y2']),
                  (0, 0, 255), 2)
    
    # Add labels
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
        f"Team Scores: x({team_scores_coords['x_start']:.2f}-{team_scores_coords['x_end']:.2f}) y({team_scores_coords['y_start']:.2f}-{team_scores_coords['y_end']:.2f})",
        f"Possession: x({possession_coords['x_start']:.2f}-{possession_coords['x_end']:.2f}) y({possession_coords['y_start']:.2f}-{possession_coords['y_end']:.2f})",
        f"Down/Dist: x({down_coords['x_start']:.3f}-{down_coords['x_end']:.3f}) y({down_coords['y_start']:.3f}-{down_coords['y_end']:.3f})"
    ]
    
    # Black background for text
    cv2.rectangle(viz_image, (5, height-85), (width-5, height-5), (0, 0, 0), -1)
    
    for i, text in enumerate(coord_info):
        cv2.putText(viz_image, text, (10, height - 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    output_path = "final_correct_regions.jpg"
    cv2.imwrite(output_path, viz_image)
    print(f"\n✅ Final correct visualization: {output_path}")
    
    # Create crops
    team_crop = image[team_pixels['y1']:team_pixels['y2'],
                     team_pixels['x1']:team_pixels['x2']]
    cv2.imwrite("team_scores_final_crop.jpg", team_crop)
    print(f"✅ Team scores final crop: team_scores_final_crop.jpg")
    
    poss_crop = image[poss_pixels['y1']:poss_pixels['y2'],
                     poss_pixels['x1']:poss_pixels['x2']]
    cv2.imwrite("possession_final_crop.jpg", poss_crop)
    print(f"✅ Possession final crop: possession_final_crop.jpg")
    
    return team_scores_coords, possession_coords

if __name__ == "__main__":
    print("🎯 FINAL CORRECT regions using 20-section grid (each unit = 0.05)...")
    print("   Team Scores: Grid 0-11, 0-2 → Coordinates 0.0-0.55, 0.0-0.10")  
    print("   Possession: Grid 5-6, 1-2 → Coordinates 0.25-0.30, 0.05-0.10")
    
    team_coords, poss_coords = visualize_final_correct_regions()
    
    print(f"\n📋 FINAL COORDINATES (format: x_start, x_end, y_start, y_end):")
    print(f"   Team Scores: {team_coords['x_start']:.2f}, {team_coords['x_end']:.2f}, {team_coords['y_start']:.2f}, {team_coords['y_end']:.2f}")
    print(f"   Possession: {poss_coords['x_start']:.2f}, {poss_coords['x_end']:.2f}, {poss_coords['y_start']:.2f}, {poss_coords['y_end']:.2f}") 
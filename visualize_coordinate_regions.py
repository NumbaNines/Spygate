#!/usr/bin/env python3
"""
Visualize Team Scores and Possession Triangle coordinate regions
"""

import cv2
import numpy as np
from pathlib import Path

def visualize_coordinate_regions():
    """Visualize the specified coordinate regions on the HUD image."""
    
    # Load the HUD image
    hud_image_path = "found_and_frame_3000.png"  # Using one of the full HUD frames
    
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
    
    # Create a copy for visualization
    viz_image = image.copy()
    
    # Define coordinate regions based on user specifications
    # Team Scores Region: vertically 0-2, horizontally 0-11
    team_scores_coords = {
        'x_start': 0.0,      # Column 0
        'x_end': 1.0,        # Column 11 (but capped at 1.0)
        'y_start': 0.0,      # Row 0  
        'y_end': 0.2         # Row 2
    }
    
    # Possession Triangle Region: vertically 1, horizontally 5-6
    possession_coords = {
        'x_start': 0.5,      # Column 5
        'x_end': 0.6,        # Column 6
        'y_start': 0.1,      # Row 1
        'y_end': 0.2         # Row 1 (giving it some height)
    }
    
    # Convert normalized coordinates to pixel coordinates
    def norm_to_pixel(coords):
        return {
            'x1': int(coords['x_start'] * width),
            'x2': int(coords['x_end'] * width),
            'y1': int(coords['y_start'] * height),
            'y2': int(coords['y_end'] * height)
        }
    
    team_scores_pixels = norm_to_pixel(team_scores_coords)
    possession_pixels = norm_to_pixel(possession_coords)
    
    print(f"\n📍 Team Scores Region:")
    print(f"   Normalized: {team_scores_coords}")
    print(f"   Pixels: {team_scores_pixels}")
    
    print(f"\n🔺 Possession Triangle Region:")
    print(f"   Normalized: {possession_coords}")
    print(f"   Pixels: {possession_pixels}")
    
    # Draw Team Scores region (Green outline, semi-transparent fill)
    cv2.rectangle(viz_image, 
                  (team_scores_pixels['x1'], team_scores_pixels['y1']),
                  (team_scores_pixels['x2'], team_scores_pixels['y2']),
                  (0, 255, 0), 3)  # Green outline
    
    # Create overlay for semi-transparent fill
    overlay = viz_image.copy()
    cv2.rectangle(overlay,
                  (team_scores_pixels['x1'], team_scores_pixels['y1']),
                  (team_scores_pixels['x2'], team_scores_pixels['y2']),
                  (0, 255, 0), -1)  # Green fill
    cv2.addWeighted(viz_image, 0.8, overlay, 0.2, 0, viz_image)
    
    # Draw Possession Triangle region (Blue outline, semi-transparent fill)
    cv2.rectangle(viz_image,
                  (possession_pixels['x1'], possession_pixels['y1']),
                  (possession_pixels['x2'], possession_pixels['y2']),
                  (255, 0, 0), 3)  # Blue outline
    
    # Create overlay for possession region
    overlay2 = viz_image.copy()
    cv2.rectangle(overlay2,
                  (possession_pixels['x1'], possession_pixels['y1']),
                  (possession_pixels['x2'], possession_pixels['y2']),
                  (255, 0, 0), -1)  # Blue fill
    cv2.addWeighted(viz_image, 0.8, overlay2, 0.2, 0, viz_image)
    
    # Add labels
    cv2.putText(viz_image, "TEAM SCORES REGION", 
                (team_scores_pixels['x1'], team_scores_pixels['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(viz_image, "POSSESSION", 
                (possession_pixels['x1'], possession_pixels['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add coordinate text
    coord_text = [
        f"Team Scores: x({team_scores_coords['x_start']:.1f}-{team_scores_coords['x_end']:.1f}) y({team_scores_coords['y_start']:.1f}-{team_scores_coords['y_end']:.1f})",
        f"Possession: x({possession_coords['x_start']:.1f}-{possession_coords['x_end']:.1f}) y({possession_coords['y_start']:.1f}-{possession_coords['y_end']:.1f})"
    ]
    
    for i, text in enumerate(coord_text):
        cv2.putText(viz_image, text, (10, height - 50 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    output_path = "coordinate_regions_visualization.jpg"
    cv2.imwrite(output_path, viz_image)
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Also create cropped versions for detailed inspection
    team_scores_crop = image[team_scores_pixels['y1']:team_scores_pixels['y2'],
                            team_scores_pixels['x1']:team_scores_pixels['x2']]
    cv2.imwrite("team_scores_region_crop.jpg", team_scores_crop)
    print(f"✅ Team scores crop saved: team_scores_region_crop.jpg")
    
    possession_crop = image[possession_pixels['y1']:possession_pixels['y2'],
                           possession_pixels['x1']:possession_pixels['x2']]
    cv2.imwrite("possession_region_crop.jpg", possession_crop)
    print(f"✅ Possession crop saved: possession_region_crop.jpg")
    
    return team_scores_coords, possession_coords

if __name__ == "__main__":
    print("🎯 Visualizing coordinate regions for team scores and possession detection...")
    team_coords, poss_coords = visualize_coordinate_regions()
    
    print(f"\n📋 Summary:")
    print(f"   • Team Scores Region: Full width (0.0-1.0), top rows (0.0-0.2)")
    print(f"   • Possession Triangle: Center area (0.5-0.6), middle row (0.1-0.2)")
    print(f"   • Images created: coordinate_regions_visualization.jpg")
    print(f"   • Crops created: team_scores_region_crop.jpg, possession_region_crop.jpg") 
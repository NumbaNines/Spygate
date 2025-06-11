#!/usr/bin/env python3
"""
Test Hybrid HUD Detector with Frame Capture
==========================================
Captures a frame and saves visualization of all detections:
- Yellow boxes: YOLO HUD regions
- Colored boxes within HUD: OpenCV detected elements
"""

import cv2
import numpy as np
import mss
import time
from hybrid_hud_element_detector import HybridHUDDetector

def capture_and_visualize():
    """Capture a frame and save visualization of detections"""
    print("🎯 Testing Hybrid HUD Detector with Visualization")
    print("=" * 50)
    
    # Initialize detector
    detector = HybridHUDDetector()
    
    # Screen capture
    sct = mss.mss()
    monitor = sct.monitors[1]
    
    print("📸 Capturing screenshot...")
    
    # Capture screenshot
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    print(f"   Screenshot size: {screenshot.shape}")
    
    # Detect elements
    print("🔍 Running hybrid detection...")
    start_time = time.time()
    hud_regions, stats = detector.detect_elements(screenshot)
    detection_time = time.time() - start_time
    
    print(f"   Detection completed in {detection_time:.3f} seconds")
    print(f"   Found {len(hud_regions)} HUD regions")
    
    # Show detailed results
    total_elements = 0
    element_summary = {}
    
    for i, hud_region in enumerate(hud_regions):
        print(f"\n📋 HUD Region {i+1}:")
        print(f"   Confidence: {hud_region.confidence:.3f}")
        print(f"   Bbox: {hud_region.bbox}")
        print(f"   Elements found: {len(hud_region.elements)}")
        
        total_elements += len(hud_region.elements)
        
        # Count element types
        for element in hud_region.elements:
            element_type = element.element_type
            element_summary[element_type] = element_summary.get(element_type, 0) + 1
            
        # Show top elements by confidence
        sorted_elements = sorted(hud_region.elements, key=lambda x: x.confidence, reverse=True)
        for j, element in enumerate(sorted_elements[:3]):  # Top 3
            print(f"     {j+1}. {element.element_type}: {element.confidence:.2f} - {element.bbox}")
    
    print(f"\n📊 Detection Summary:")
    print(f"   Total HUD regions: {len(hud_regions)}")
    print(f"   Total elements: {total_elements}")
    print(f"   Detection time: {detection_time:.3f}s")
    
    print(f"\n🏷️ Element Types:")
    for element_type, count in sorted(element_summary.items()):
        print(f"   {element_type}: {count}")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    vis_image = detector.visualize_detections(screenshot, hud_regions)
    
    # Add summary text overlay
    overlay_y = 30
    overlay_text = [
        f"Hybrid HUD Detector Results",
        f"HUD Regions: {len(hud_regions)} | Elements: {total_elements}",
        f"YOLO Time: {stats['avg_yolo_time_ms']:.1f}ms | OpenCV Time: {stats['avg_opencv_time_ms']:.1f}ms",
        f"Yellow=HUD | Orange=Possession | Purple=Territory | White=Generic | Green=Text | Red=Score"
    ]
    
    for text in overlay_text:
        cv2.putText(vis_image, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        overlay_y += 35
    
    # Save original and visualization
    timestamp = int(time.time())
    
    original_path = f"hybrid_test_original_{timestamp}.jpg"
    visualization_path = f"hybrid_test_results_{timestamp}.jpg"
    
    cv2.imwrite(original_path, screenshot)
    cv2.imwrite(visualization_path, vis_image)
    
    print(f"\n💾 Files saved:")
    print(f"   Original: {original_path}")
    print(f"   Results: {visualization_path}")
    
    # Performance stats
    final_stats = detector.get_stats()
    print(f"\n⚡ Performance Stats:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return visualization_path, hud_regions, stats

def create_detailed_hud_breakdown(hud_regions, output_path="hud_breakdown.jpg"):
    """Create a detailed breakdown showing each HUD region separately"""
    print(f"\n🔬 Creating detailed HUD breakdown...")
    
    if not hud_regions:
        print("   No HUD regions to break down")
        return
    
    # Calculate grid layout
    num_huds = len(hud_regions)
    cols = min(3, num_huds)  # Max 3 columns
    rows = (num_huds + cols - 1) // cols
    
    # Find max HUD dimensions for consistent sizing
    max_width = max(region.region_image.shape[1] for region in hud_regions)
    max_height = max(region.region_image.shape[0] for region in hud_regions)
    
    # Create grid image
    grid_width = cols * (max_width + 20) + 20
    grid_height = rows * (max_height + 60) + 20
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for i, hud_region in enumerate(hud_regions):
        row = i // cols
        col = i % cols
        
        # Position in grid
        x_offset = col * (max_width + 20) + 10
        y_offset = row * (max_height + 60) + 10
        
        # Resize HUD region to fit
        hud_img = hud_region.region_image.copy()
        if hud_img.shape[1] != max_width or hud_img.shape[0] != max_height:
            hud_img = cv2.resize(hud_img, (max_width, max_height))
        
        # Draw elements on HUD
        for element in hud_region.elements:
            x1, y1, x2, y2 = element.local_bbox
            
            # Scale coordinates if resized
            if hud_region.region_image.shape[1] != max_width:
                scale_x = max_width / hud_region.region_image.shape[1]
                scale_y = max_height / hud_region.region_image.shape[0]
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Color code by element type
            colors = {
                'possession_triangle': (0, 165, 255),  # Orange
                'territory_triangle': (128, 0, 128),   # Purple
                'generic_triangle': (255, 255, 255),   # White
                'text_region': (0, 255, 0),            # Green
                'score_number': (0, 0, 255)            # Red
            }
            
            color = colors.get(element.element_type, (128, 128, 128))
            cv2.rectangle(hud_img, (x1, y1), (x2, y2), color, 1)
        
        # Place in grid
        grid_image[y_offset:y_offset+max_height, x_offset:x_offset+max_width] = hud_img
        
        # Add label
        label = f"HUD {i+1} (conf: {hud_region.confidence:.2f})"
        cv2.putText(grid_image, label, (x_offset, y_offset + max_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add element count
        element_text = f"Elements: {len(hud_region.elements)}"
        cv2.putText(grid_image, element_text, (x_offset, y_offset + max_height + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imwrite(output_path, grid_image)
    print(f"   Breakdown saved: {output_path}")

if __name__ == "__main__":
    # Run test
    vis_path, hud_regions, stats = capture_and_visualize()
    
    # Create detailed breakdown
    create_detailed_hud_breakdown(hud_regions)
    
    print(f"\n✅ Test complete! Check the saved images:")
    print(f"   - Main visualization: {vis_path}")
    print(f"   - HUD breakdown: hud_breakdown.jpg") 
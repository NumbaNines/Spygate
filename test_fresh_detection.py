#!/usr/bin/env python3
"""
Test Fresh HUD Region Detection System
Tests the newly trained model with the hybrid triangle detection approach
"""

import cv2
import numpy as np
from pathlib import Path
import time
from hybrid_triangle_detector_optimized import OptimizedHybridTriangleDetector

def create_test_frame():
    """Create a synthetic test frame with HUD-like elements."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Create a dark background (football field color)
    frame[:] = (34, 67, 35)  # Dark green
    
    # Create HUD bar at bottom
    hud_y1, hud_y2 = 650, 720
    frame[hud_y1:hud_y2, :] = (20, 20, 20)  # Dark gray HUD
    
    # Add some HUD text areas
    cv2.rectangle(frame, (50, 660), (200, 690), (40, 40, 40), -1)  # Team info
    cv2.rectangle(frame, (1080, 660), (1230, 690), (40, 40, 40), -1)  # Team info
    cv2.rectangle(frame, (580, 660), (700, 690), (40, 40, 40), -1)  # Score/time
    
    # Add some triangle-like shapes in potential areas
    # Possession triangle area (left side)
    cv2.fillPoly(frame, [np.array([[300, 670], [310, 680], [320, 670]])], (255, 255, 255))
    
    # Territory triangle area (right side)  
    cv2.fillPoly(frame, [np.array([[950, 670], [960, 660], [970, 670]])], (255, 255, 255))
    
    return frame

def test_with_real_image():
    """Test with a real image if available."""
    # Look for test images in NEW MADDEN DATA
    test_images = list(Path("NEW MADDEN DATA").glob("*.png"))
    
    if test_images:
        print(f"📸 Found {len(test_images)} test images")
        return cv2.imread(str(test_images[0]))
    else:
        print("📸 No real test images found, using synthetic frame")
        return create_test_frame()

def main():
    """Main testing function."""
    print("🧪 Testing Fresh HUD Region Detection System")
    print("=" * 60)
    
    # Initialize the detector
    print("🔧 Initializing Optimized Hybrid Triangle Detector...")
    detector = OptimizedHybridTriangleDetector()
    
    # Check if model exists
    if not detector.load_model():
        print("❌ Fresh model not ready yet!")
        print("🔄 Please wait for training to complete or run train_hud_regions_optimized.py")
        return
    
    print("✅ Fresh model loaded successfully!")
    print(f"📍 Model path: {detector.model_path}")
    print()
    
    # Test with different frame types
    test_frames = [
        ("Synthetic Test Frame", create_test_frame()),
        ("Real Madden Frame", test_with_real_image())
    ]
    
    for frame_name, frame in test_frames:
        if frame is None:
            continue
            
        print(f"🎮 Testing with {frame_name}")
        print("-" * 40)
        
        # Analyze the frame
        start_time = time.time()
        results = detector.analyze_frame(frame, conf_threshold=0.25)
        analysis_time = time.time() - start_time
        
        # Display results
        print(f"⏱️  Analysis time: {analysis_time:.3f}s")
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            continue
        
        print(f"🔍 HUD Regions found: {results.get('region_count', 0)}")
        for i, region in enumerate(results.get('hud_regions', [])):
            print(f"   {i+1}. {region['class_name']} (conf: {region['confidence']:.3f})")
        
        print(f"📐 Triangles found: {results.get('triangle_count', 0)}")
        for i, triangle in enumerate(results.get('triangles', [])):
            print(f"   {i+1}. Triangle in {triangle['region_class']} (area: {triangle['area']:.0f})")
        
        # Visualize results
        vis_frame = detector.visualize_results(frame, results)
        
        # Save visualization
        output_path = f"test_detection_{frame_name.lower().replace(' ', '_')}.png"
        cv2.imwrite(output_path, vis_frame)
        print(f"💾 Visualization saved: {output_path}")
        
        print()
    
    print("✅ Fresh detection system testing complete!")
    print(f"🎯 Model used: {detector.model_path}")

if __name__ == "__main__":
    main() 
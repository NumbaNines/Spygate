#!/usr/bin/env python3
"""
Test Hybrid HUD Detector with Sample Image
==========================================
Uses existing sample images to demonstrate the hybrid detection system
"""

import cv2
import numpy as np
import time
from hybrid_hud_element_detector import HybridHUDDetector

def test_with_sample_image(image_path="triangle_visualization_3.jpg"):
    """Test hybrid detector with a sample image"""
    print("🎯 Testing Hybrid HUD Detector with Sample Image")
    print("=" * 50)
    
    # Load sample image
    print(f"📸 Loading sample image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"   Image size: {image.shape}")
    
    # Initialize detector
    detector = HybridHUDDetector()
    
    # Detect elements
    print("🔍 Running hybrid detection...")
    start_time = time.time()
    hud_regions, stats = detector.detect_elements(image)
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
        
        # Show top elements by confidence (only triangles for clarity)
        triangle_elements = [e for e in hud_region.elements if 'triangle' in e.element_type]
        if triangle_elements:
            print(f"   🔺 Triangle detections:")
            sorted_triangles = sorted(triangle_elements, key=lambda x: x.confidence, reverse=True)
            for j, element in enumerate(sorted_triangles[:5]):  # Top 5 triangles
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
    vis_image = detector.visualize_detections(image, hud_regions)
    
    # Add summary text overlay
    overlay_y = 30
    overlay_text = [
        f"Hybrid HUD Detector Results - Sample Image Test",
        f"HUD Regions: {len(hud_regions)} | Elements: {total_elements}",
        f"YOLO Time: {stats['avg_yolo_time_ms']:.1f}ms | OpenCV Time: {stats['avg_opencv_time_ms']:.1f}ms",
        f"Yellow=HUD | Orange=Possession | Purple=Territory | White=Generic | Green=Text | Red=Score"
    ]
    
    for text in overlay_text:
        cv2.putText(vis_image, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        overlay_y += 35
    
    # Save results
    timestamp = int(time.time())
    
    # Save original and visualization
    visualization_path = f"hybrid_sample_test_results_{timestamp}.jpg"
    cv2.imwrite(visualization_path, vis_image)
    
    print(f"\n💾 Visualization saved: {visualization_path}")
    
    # Performance stats
    final_stats = detector.get_stats()
    print(f"\n⚡ Performance Stats:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Triangle-specific analysis
    triangle_analysis(hud_regions)
    
    return visualization_path, hud_regions, stats

def triangle_analysis(hud_regions):
    """Analyze triangle detections specifically"""
    print(f"\n🔺 Triangle Analysis:")
    
    triangle_types = ['possession_triangle', 'territory_triangle', 'generic_triangle']
    
    for triangle_type in triangle_types:
        triangles = []
        for hud_region in hud_regions:
            triangles.extend([e for e in hud_region.elements if e.element_type == triangle_type])
        
        if triangles:
            print(f"\n   {triangle_type.upper()}:")
            print(f"     Count: {len(triangles)}")
            confidences = [t.confidence for t in triangles]
            print(f"     Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"     Average confidence: {np.mean(confidences):.3f}")
            
            # Show best detections
            sorted_triangles = sorted(triangles, key=lambda x: x.confidence, reverse=True)
            for i, triangle in enumerate(sorted_triangles[:3]):
                props = triangle.properties
                print(f"     #{i+1}: conf={triangle.confidence:.3f}, area={props.get('area', 0):.0f}, vertices={props.get('vertices', 0)}")

def test_all_sample_images():
    """Test with all available sample images"""
    sample_images = [
        "triangle_visualization_1.jpg",
        "triangle_visualization_2.jpg", 
        "triangle_visualization_3.jpg"
    ]
    
    print("🎮 Testing Hybrid System with Multiple Sample Images")
    print("=" * 60)
    
    for i, image_path in enumerate(sample_images):
        print(f"\n{'='*20} TEST {i+1}: {image_path} {'='*20}")
        try:
            vis_path, hud_regions, stats = test_with_sample_image(image_path)
            print(f"✅ Test {i+1} completed successfully")
        except Exception as e:
            print(f"❌ Test {i+1} failed: {e}")
        
        if i < len(sample_images) - 1:
            print("\n" + "-"*80)

if __name__ == "__main__":
    # Test with single sample image first
    try:
        print("🚀 Single Image Test")
        test_with_sample_image()
    except Exception as e:
        print(f"❌ Single test failed: {e}")
    
    print("\n" + "="*80)
    
    # Test with all sample images
    try:
        print("🚀 Multiple Image Test")
        test_all_sample_images()
    except Exception as e:
        print(f"❌ Multiple test failed: {e}")
    
    print(f"\n✅ Testing complete!") 
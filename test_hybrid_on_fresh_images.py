#!/usr/bin/env python3
"""
Test New Hybrid Triangle Detection System on Fresh Images
Uses different images from madden 6111 folder that we haven't tested yet
"""

import cv2
import numpy as np
from pathlib import Path
import random
from hybrid_triangle_detector_final import FinalHybridTriangleDetector

def test_on_fresh_images():
    """Test our new hybrid system on different untested images."""
    
    # Get all available images from madden 6111
    image_folder = Path("madden 6111")
    all_images = list(image_folder.glob("*.png"))
    
    print(f"📂 Found {len(all_images)} total images in {image_folder}")
    
    # Select 10 random different images (not the ones we tested before)
    previously_tested = [
        "monitor3_screenshot_20250611_031428_268.png",
        "monitor3_screenshot_20250611_032429_236.png", 
        "monitor3_screenshot_20250611_033814_201.png"
    ]
    
    # Filter out previously tested images
    fresh_images = [img for img in all_images if img.name not in previously_tested]
    
    # Randomly select 10 fresh images
    test_images = random.sample(fresh_images, min(10, len(fresh_images)))
    
    print(f"🎯 Testing on {len(test_images)} completely fresh images...")
    print("📸 Selected images:")
    for img in test_images:
        print(f"   - {img.name}")
    
    # Initialize detector
    detector = FinalHybridTriangleDetector()
    
    results_summary = {
        'total_images': len(test_images),
        'images_with_detections': 0,
        'total_regions': 0,
        'total_triangles': 0,
        'avg_processing_time': 0,
        'detailed_results': []
    }
    
    print(f"\n🔍 Starting analysis...")
    print("="*60)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n📸 Image {i}/{len(test_images)}: {img_path.name}")
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"❌ Could not load {img_path}")
            continue
            
        # Process with hybrid system
        results = detector.process_frame(frame)
        
        # Extract key metrics
        regions_found = results['regions_detected']
        triangle_results = results['triangle_results']
        processing_time = results['processing_time_ms']
        
        # Count triangles found
        triangles_found = sum([tri['triangles_found'] for tri in triangle_results])
        
        # Update summary
        results_summary['total_regions'] += regions_found
        results_summary['total_triangles'] += triangles_found
        results_summary['avg_processing_time'] += processing_time
        
        if regions_found > 0:
            results_summary['images_with_detections'] += 1
        
        # Display results for this image
        print(f"   ✅ Regions detected: {regions_found}")
        print(f"   🔺 Triangles found: {triangles_found}")
        print(f"   ⚡ Processing time: {processing_time:.1f}ms")
        
        # Show detailed region breakdown
        if regions_found > 0:
            print(f"   📊 Region breakdown:")
            for region in results['regions']:
                print(f"      - {region['class_name']}: {region['confidence']:.3f}")
        
        # Show triangle detection details
        if triangles_found > 0:
            print(f"   🎯 Triangle details:")
            for tri_result in triangle_results:
                region_name = tri_result['region_info']['class_name']
                tri_count = tri_result['triangles_found']
                if tri_count > 0:
                    direction = tri_result.get('triangle_direction', 'unknown')
                    print(f"      - {region_name}: {tri_count} triangles ({direction})")
        
        # Create and save visualization
        vis_frame = detector.create_visualization(frame, results)
        output_path = f"fresh_test_{i}_{img_path.stem}.png"
        cv2.imwrite(output_path, vis_frame)
        print(f"   💾 Saved: {output_path}")
        
        # Store detailed results
        results_summary['detailed_results'].append({
            'image': img_path.name,
            'regions': regions_found,
            'triangles': triangles_found,
            'processing_time': processing_time,
            'regions_detail': results['regions'],
            'triangle_detail': triangle_results
        })
    
    # Calculate final summary
    if results_summary['total_images'] > 0:
        results_summary['avg_processing_time'] /= results_summary['total_images']
        detection_rate = (results_summary['images_with_detections'] / results_summary['total_images']) * 100
    else:
        detection_rate = 0
    
    print("\n" + "="*60)
    print("🎉 FRESH IMAGE TESTING COMPLETE!")
    print("="*60)
    print(f"📊 **SUMMARY RESULTS:**")
    print(f"   🖼️  Total images tested: {results_summary['total_images']}")
    print(f"   ✅ Images with detections: {results_summary['images_with_detections']} ({detection_rate:.1f}%)")
    print(f"   🎯 Total regions found: {results_summary['total_regions']}")
    print(f"   🔺 Total triangles detected: {results_summary['total_triangles']}")
    print(f"   ⚡ Average processing time: {results_summary['avg_processing_time']:.1f}ms")
    print(f"   📈 Regions per image: {results_summary['total_regions'] / results_summary['total_images']:.1f}")
    print(f"   📈 Triangles per image: {results_summary['total_triangles'] / results_summary['total_images']:.1f}")
    
    # Show class breakdown
    class_counts = {}
    for detail in results_summary['detailed_results']:
        for region in detail['regions_detail']:
            class_name = region['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print(f"\n📋 **CLASS DETECTION BREAKDOWN:**")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / results_summary['total_regions']) * 100
            print(f"   - {class_name}: {count} detections ({percentage:.1f}%)")
    
    print(f"\n💾 All visualizations saved as: fresh_test_*.png")
    print("="*60)
    
    return results_summary

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    test_on_fresh_images() 
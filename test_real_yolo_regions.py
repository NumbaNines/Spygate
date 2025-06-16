#!/usr/bin/env python3
"""
Test SpygateAI Down Template Detection with Real YOLO Regions

This script tests using the actual down_distance_area coordinates that YOLO would detect
from full Madden screenshots, not the small cropped test images.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time
import json

# Add src to path for imports
sys.path.append('src')

from spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext

def load_template_metadata():
    """Load the template metadata with real coordinates"""
    metadata_path = Path("down_templates_real/templates_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def test_real_yolo_regions():
    """Test using real YOLO down_distance_area regions from full screenshots"""
    print("=" * 60)
    print("SpygateAI Down Template Detection - Real YOLO Regions Test")
    print("=" * 60)
    
    # Initialize detector
    print("\n1. Initializing DownTemplateDetector...")
    detector = DownTemplateDetector(
        templates_dir=Path("down_templates_real"),
        debug_output_dir=Path("debug_real_yolo_test")
    )
    
    print(f"   Templates loaded: {len(detector.templates)}")
    
    # Load metadata to get real coordinates
    metadata = load_template_metadata()
    
    # Test cases using full screenshots and real coordinates
    test_cases = [
        {
            "screenshot": "down templates/1.png",
            "expected_down": 1,
            "expected_template": "1ST",
            "metadata_key": "1ST"
        },
        {
            "screenshot": "down templates/2.png", 
            "expected_down": 2,
            "expected_template": "2ND",
            "metadata_key": "2ND"
        },
        {
            "screenshot": "down templates/1st goal.png",
            "expected_down": 1,
            "expected_template": "1ST_GOAL", 
            "metadata_key": "1ST_GOAL"
        },
        {
            "screenshot": "down templates/4th goal.png",
            "expected_down": 4,
            "expected_template": "4TH_GOAL",
            "metadata_key": "4TH_GOAL"
        }
    ]
    
    print(f"\n2. Testing {len(test_cases)} real YOLO regions...")
    
    results = []
    correct_detections = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Test {i+1}: {Path(test_case['screenshot']).name}")
        
        # Load full screenshot
        screenshot_path = Path(test_case["screenshot"])
        if not screenshot_path.exists():
            print(f"   ✗ Screenshot not found: {screenshot_path}")
            continue
            
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            print(f"   ✗ Failed to load screenshot: {screenshot_path}")
            continue
            
        print(f"   Screenshot shape: {screenshot.shape}")
        
        # Get real YOLO coordinates from metadata
        metadata_key = test_case["metadata_key"]
        if metadata_key not in metadata:
            print(f"   ✗ Metadata not found for: {metadata_key}")
            continue
            
        crop_coords = metadata[metadata_key]["crop_coords"]
        x1, y1, x2, y2 = crop_coords
        
        # Convert to YOLO bbox format (x, y, w, h)
        yolo_bbox = (x1, y1, x2 - x1, y2 - y1)
        print(f"   Real YOLO bbox: {yolo_bbox}")
        
        # Create context for GOAL situations
        context = None
        if "GOAL" in test_case["expected_template"]:
            context = DownDetectionContext(
                field_position="GOAL",
                is_goal_line=True
            )
        
        # Run detection with real YOLO region
        start_time = time.time()
        match = detector.detect_down_in_yolo_region(screenshot, yolo_bbox, context)
        detection_time = time.time() - start_time
        
        # Analyze results
        expected_down = test_case["expected_down"]
        expected_template = test_case["expected_template"]
        
        if match:
            detected_down = match.down
            detected_template = match.template_name
            confidence = match.confidence
            
            is_correct = (detected_down == expected_down)
            if is_correct:
                correct_detections += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"
            
            print(f"   {status}: Detected {detected_down} (expected {expected_down})")
            print(f"   Template: {detected_template} (confidence: {confidence:.3f})")
            print(f"   Position: {match.position}, Scale: {match.scale_factor:.2f}")
            print(f"   Time: {detection_time*1000:.1f}ms")
            
            results.append({
                "test_case": Path(test_case["screenshot"]).name,
                "expected": expected_down,
                "detected": detected_down,
                "correct": is_correct,
                "confidence": confidence,
                "template": detected_template,
                "time_ms": detection_time * 1000
            })
        else:
            print(f"   ✗ NO DETECTION")
            results.append({
                "test_case": Path(test_case["screenshot"]).name,
                "expected": expected_down,
                "detected": None,
                "correct": False,
                "confidence": 0.0,
                "template": None,
                "time_ms": detection_time * 1000
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL YOLO REGIONS TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    accuracy = (correct_detections / total_tests) * 100 if total_tests > 0 else 0
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")
    
    if results:
        avg_time = np.mean([r["time_ms"] for r in results])
        valid_confidences = [r["confidence"] for r in results if r["confidence"] > 0]
        avg_confidence = np.mean(valid_confidences) if valid_confidences else 0
        
        print(f"Average Detection Time: {avg_time:.1f}ms")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"FPS Capability: {1000/avg_time:.1f} FPS")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        conf_str = f"{result['confidence']:.3f}" if result['confidence'] > 0 else "N/A"
        template_str = result['template'] or "None"
        print(f"  {status} {result['test_case']}: {result['detected']} (conf: {conf_str}, template: {template_str})")
    
    return results

def visualize_yolo_regions():
    """Visualize the YOLO regions on the screenshots"""
    print("\n" + "=" * 60)
    print("VISUALIZING YOLO REGIONS")
    print("=" * 60)
    
    metadata = load_template_metadata()
    
    test_cases = [
        ("down templates/1.png", "1ST"),
        ("down templates/2.png", "2ND"),
        ("down templates/1st goal.png", "1ST_GOAL"),
        ("down templates/4th goal.png", "4TH_GOAL")
    ]
    
    for screenshot_path, metadata_key in test_cases:
        if not Path(screenshot_path).exists():
            continue
            
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            continue
            
        if metadata_key not in metadata:
            continue
            
        # Get coordinates
        crop_coords = metadata[metadata_key]["crop_coords"]
        x1, y1, x2, y2 = crop_coords
        
        # Draw rectangle on screenshot
        vis_screenshot = screenshot.copy()
        cv2.rectangle(vis_screenshot, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label
        cv2.putText(vis_screenshot, f"YOLO: {metadata_key}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        output_path = f"debug_yolo_region_{metadata_key}.png"
        cv2.imwrite(output_path, vis_screenshot)
        print(f"   Saved: {output_path}")

if __name__ == "__main__":
    # Visualize YOLO regions first
    visualize_yolo_regions()
    
    # Run the real test
    results = test_real_yolo_regions()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60) 
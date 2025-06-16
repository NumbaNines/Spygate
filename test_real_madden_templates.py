#!/usr/bin/env python3
"""
Test Real Madden Templates - Final Verification
"""

import cv2
import numpy as np
from pathlib import Path
from down_template_detector import DownTemplateDetector

def test_real_templates():
    """Test the real Madden templates."""
    print("🎮 Testing REAL Madden Templates")
    print("=" * 50)
    
    # Initialize detector
    detector = DownTemplateDetector()
    print(f"✅ Loaded {len(detector.templates)} templates")
    
    # Test files - use the same screenshots we created templates from
    test_cases = [
        {"file": "down templates/1.png", "expected": 1, "desc": "1st down"},
        {"file": "down templates/2.png", "expected": 2, "desc": "2nd down"},
        {"file": "down templates/3rd.png", "expected": 3, "desc": "3rd down"},
        {"file": "down templates/4th.png", "expected": 4, "desc": "4th down"},
        {"file": "down templates/1st goal.png", "expected": 1, "desc": "1st & goal"},
        {"file": "down templates/2nd goal.png", "expected": 2, "desc": "2nd & goal"},
        {"file": "down templates/3rd goal.png", "expected": 3, "desc": "3rd & goal"},
        {"file": "down templates/4th goal.png", "expected": 4, "desc": "4th & goal"},
    ]
    
    # Crop coordinates (same as template creation)
    crop_coords = {"x": 1300, "y": 50, "width": 150, "height": 50}
    
    correct = 0
    total = 0
    
    print("\n🔍 Testing Template Detection:")
    print("-" * 50)
    
    for test_case in test_cases:
        if not Path(test_case["file"]).exists():
            print(f"⚠️  {test_case['file']} not found, skipping...")
            continue
            
        # Load and crop the test image
        img = cv2.imread(test_case["file"])
        if img is None:
            print(f"❌ Failed to load {test_case['file']}")
            continue
            
        # Crop to down/distance area
        x, y, w, h = crop_coords["x"], crop_coords["y"], crop_coords["width"], crop_coords["height"]
        cropped = img[y:y+h, x:x+w]
        
        if cropped.size == 0:
            print(f"❌ Empty crop for {test_case['desc']}")
            continue
            
        # Create a fake YOLO detection for the cropped area
        fake_detection = {
            'bbox': [0, 0, w, h],  # Full cropped area
            'confidence': 0.9
        }
        
        # Detect down number
        result = detector.detect_down_from_region(cropped, fake_detection)
        
        total += 1
        expected = test_case["expected"]
        detected = result.get("down_number", 0) if result else 0
        confidence = result.get("confidence", 0.0) if result else 0.0
        
        if detected == expected:
            print(f"✅ {test_case['desc']}: Expected {expected}, Got {detected} (conf: {confidence:.3f})")
            correct += 1
        else:
            print(f"❌ {test_case['desc']}: Expected {expected}, Got {detected} (conf: {confidence:.3f})")
    
    print()
    print("=" * 50)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"🎯 FINAL RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    
    if accuracy >= 80:
        print("🎉 EXCELLENT! Templates are working great!")
    elif accuracy >= 60:
        print("👍 GOOD! Templates are working well.")
    else:
        print("⚠️  Templates need improvement.")
    
    return accuracy >= 70

if __name__ == "__main__":
    test_real_templates() 
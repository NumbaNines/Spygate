#!/usr/bin/env python3
"""
Test Down Template Detector with Real Madden Screenshots

This tests the detector using actual Madden game screenshots instead of synthetic text.
"""

import cv2
import numpy as np
from pathlib import Path
from down_template_detector import DownTemplateDetector


def test_with_real_screenshots():
    """Test template detector with real Madden screenshots."""
    print("🎮 Testing with Real Madden Screenshots")
    print("=" * 50)
    
    # Initialize detector
    detector = DownTemplateDetector()
    print(f"✅ Loaded {len(detector.templates)} templates")
    
    # Real screenshot files and their expected results
    # NOTE: These are pre-cropped down_distance_area regions (80x30), not full screenshots
    test_files = [
        {"file": "templates/raw_gameplay/1st_10.png", "expected_down": 1, "is_goal": False, "description": "1st & 10"},
        {"file": "templates/raw_gameplay/2nd_7.png", "expected_down": 2, "is_goal": False, "description": "2nd & 7"},
        {"file": "templates/raw_gameplay/2nd_10.png", "expected_down": 2, "is_goal": False, "description": "2nd & 10"},
        {"file": "templates/raw_gameplay/3rd_3.png", "expected_down": 3, "is_goal": False, "description": "3rd & 3"},
        {"file": "templates/raw_gameplay/3rd_10.png", "expected_down": 3, "is_goal": False, "description": "3rd & 10"},
        {"file": "templates/raw_gameplay/4th_1.png", "expected_down": 4, "is_goal": False, "description": "4th & 1"},
        {"file": "templates/raw_gameplay/4th_10.png", "expected_down": 4, "is_goal": False, "description": "4th & 10"},
        {"file": "templates/raw_gameplay/3rd_goal.png", "expected_down": 3, "is_goal": True, "description": "3rd & Goal"},
        {"file": "templates/raw_gameplay/4th_goal.png", "expected_down": 4, "is_goal": True, "description": "4th & Goal"},
    ]
    
    successful = 0
    total = 0
    confidences = []
    
    print("\\n🧪 Testing Real Screenshots:")
    
    for test_case in test_files:
        file_path = Path(test_case["file"])
        
        if not file_path.exists():
            print(f"   ⚠️ {test_case['description']}: File not found ({file_path})")
            continue
        
        # Load the screenshot
        frame = cv2.imread(str(file_path))
        if frame is None:
            print(f"   ❌ {test_case['description']}: Failed to load image")
            continue
        
        total += 1
        
        # Since these are pre-cropped regions (80x30), use the full image as bbox
        height, width = frame.shape[:2]
        bbox = (0, 0, width, height)  # Use entire pre-cropped region
        
        # Test detection
        result = detector.detect_down_in_yolo_region(
            frame, bbox, test_case["is_goal"]
        )
        
        if result:
            correct = result.down == test_case["expected_down"]
            if correct:
                successful += 1
                confidences.append(result.confidence)
            
            print(f"   {'✅' if correct else '❌'} {test_case['description']}: "
                  f"Detected {result.down} (conf: {result.confidence:.3f}, "
                  f"template: {result.template_name})")
        else:
            print(f"   ❌ {test_case['description']}: No detection")
    
    # Calculate results
    if total > 0:
        accuracy = successful / total
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        print(f"\\n📊 Real Screenshot Results:")
        print(f"   🎯 Accuracy: {accuracy:.1%} ({successful}/{total})")
        print(f"   📈 Average Confidence: {avg_confidence:.3f}")
        print(f"   📋 Files Tested: {total}")
        
        # Expert recommendation for real data
        if accuracy >= 0.8:
            recommendation = "🚀 EXCELLENT - Ready for production with real data!"
        elif accuracy >= 0.6:
            recommendation = "✅ GOOD - Minor tuning needed"
        elif accuracy >= 0.4:
            recommendation = "⚠️ MODERATE - Needs improvement"
        else:
            recommendation = "❌ POOR - Major rework required"
        
        print(f"   🎯 Real Data Recommendation: {recommendation}")
        
        return accuracy, avg_confidence, total
    else:
        print("   ❌ No valid test files found!")
        return 0.0, 0.0, 0


def test_template_direct_matching():
    """Test direct template matching on pre-cropped regions."""
    print("\\n🔍 Testing Direct Template Matching...")
    
    detector = DownTemplateDetector()
    
    # Test with one known file to debug template matching
    test_file = "templates/raw_gameplay/1st_10.png"
    if not Path(test_file).exists():
        print("   ⚠️ Test file not found, skipping direct matching test")
        return
    
    frame = cv2.imread(test_file)
    if frame is None:
        print("   ❌ Failed to load test image")
        return
    
    print(f"   📏 Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    # Apply preprocessing
    preprocessed = detector._apply_spygate_preprocessing(frame)
    print(f"   🔧 Preprocessed dimensions: {preprocessed.shape[1]}x{preprocessed.shape[0]}")
    
    # Test template matching directly
    matches = detector._match_templates_with_context(
        preprocessed, (0, 0), is_goal_situation=False
    )
    
    print(f"   🎯 Found {len(matches)} template matches")
    for match in matches:
        print(f"      - {match.template_name}: {match.confidence:.3f}")
    
    # Test with GOAL situation
    goal_matches = detector._match_templates_with_context(
        preprocessed, (0, 0), is_goal_situation=True
    )
    
    print(f"   🎯 Found {len(goal_matches)} GOAL template matches")
    for match in goal_matches:
        print(f"      - {match.template_name}: {match.confidence:.3f}")


def main():
    """Run comprehensive real screenshot testing."""
    print("🎮 SpygateAI Real Screenshot Template Test")
    print("=" * 50)
    
    # Test 1: Real Screenshots (pre-cropped regions)
    accuracy, avg_confidence, total_files = test_with_real_screenshots()
    
    # Test 2: Direct Template Matching Debug
    test_template_direct_matching()
    
    # Final Assessment
    print("\\n" + "=" * 50)
    print("🎯 FINAL EXPERT ASSESSMENT")
    print("=" * 50)
    
    if total_files > 0:
        print(f"Real Data Accuracy: {accuracy:.1%}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Files Successfully Tested: {total_files}")
        
        if accuracy >= 0.7:
            print("\\n✅ EXPERT VERDICT: Template system shows promise with real data!")
            print("🚀 NEXT STEPS:")
            print("   1. Integrate into enhanced_game_analyzer.py")
            print("   2. Test with full SpygateAI video pipeline")
            print("   3. Fine-tune confidence thresholds based on real performance")
        else:
            print("\\n⚠️ EXPERT VERDICT: Needs improvement with real data")
            print("🔧 NEXT STEPS:")
            print("   1. Analyze failed detections")
            print("   2. Improve template preprocessing")
            print("   3. Consider template re-creation with better quality")
            print("   4. Debug template matching confidence scores")
    else:
        print("❌ No real screenshots available for testing")
        print("📋 RECOMMENDATION: Obtain real Madden screenshots for proper validation")


if __name__ == "__main__":
    main() 
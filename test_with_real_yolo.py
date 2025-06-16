#!/usr/bin/env python3
"""
Test SpygateAI Down Template Detection with Real YOLOv8 Model

This script uses the actual 8-class YOLOv8 model to detect down_distance_area regions
and then applies template matching on those real detected regions.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path for imports
sys.path.append('src')

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext
from spygate.core.hardware import HardwareDetector

def test_with_real_yolo():
    """Test using real YOLOv8 model to detect down_distance_area regions"""
    print("=" * 60)
    print("SpygateAI Down Template Detection - Real YOLOv8 Test")
    print("=" * 60)
    
    # Initialize hardware detector
    print("\n1. Initializing hardware detection...")
    hardware = HardwareDetector()
    print(f"   Hardware tier: {hardware.detect_tier().name}")
    
    # Initialize enhanced game analyzer (contains YOLOv8 model)
    print("\n2. Initializing Enhanced Game Analyzer...")
    try:
        analyzer = EnhancedGameAnalyzer(
            model_path="hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
            hardware=hardware
        )
        print("   ✓ Enhanced Game Analyzer initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize analyzer: {e}")
        return
    
    # Initialize down template detector
    print("\n3. Initializing Down Template Detector...")
    detector = DownTemplateDetector(
        templates_dir=Path("down_templates_real"),
        debug_output_dir=Path("debug_real_yolo_test")
    )
    print(f"   ✓ Templates loaded: {len(detector.templates)}")
    
    # Test cases using full Madden screenshots
    test_cases = [
        {
            "screenshot": "down templates/1.png",
            "expected_down": 1,
            "expected_template": "1ST",
            "description": "1st down normal"
        },
        {
            "screenshot": "down templates/2.png", 
            "expected_down": 2,
            "expected_template": "2ND",
            "description": "2nd down normal"
        },
        {
            "screenshot": "down templates/1st goal.png",
            "expected_down": 1,
            "expected_template": "1ST_GOAL", 
            "description": "1st down GOAL"
        },
        {
            "screenshot": "down templates/4th goal.png",
            "expected_down": 4,
            "expected_template": "4TH_GOAL",
            "description": "4th down GOAL"
        }
    ]
    
    print(f"\n4. Testing {len(test_cases)} screenshots with real YOLO detection...")
    
    results = []
    correct_detections = 0
    
    for i, test_case in enumerate(test_cases):
        print(f"\n   Test {i+1}: {test_case['description']}")
        
        # Load screenshot
        screenshot_path = Path(test_case["screenshot"])
        if not screenshot_path.exists():
            print(f"   ✗ Screenshot not found: {screenshot_path}")
            continue
            
        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            print(f"   ✗ Failed to load screenshot: {screenshot_path}")
            continue
            
        print(f"   Screenshot shape: {screenshot.shape}")
        
        # Use YOLO to detect down_distance_area
        print("   Running YOLO detection...")
        start_yolo = time.time()
        
        try:
            # Use the analyzer's YOLO model to detect regions
            detections = analyzer.model.detect(screenshot)
            yolo_time = time.time() - start_yolo
            
            print(f"   YOLO detection time: {yolo_time*1000:.1f}ms")
            print(f"   Total detections: {len(detections)}")
            
            # Debug: Show all detections first
            print("   All YOLO detections:")
            for j, detection in enumerate(detections):
                class_name = detection.get("class_name", "unknown")  # Fixed: use "class_name"
                confidence = detection.get("confidence", 0.0)
                bbox = detection.get("bbox", [0, 0, 0, 0])
                print(f"     {j}: {class_name} (conf: {confidence:.3f}) bbox: {bbox}")
            
            # Find down_distance_area detection
            down_distance_detection = None
            for detection in detections:
                if detection.get("class_name") == "down_distance_area":  # Fixed: use "class_name"
                    confidence = detection.get("confidence", 0.0)
                    if confidence > 0.3:  # Lower confidence threshold
                        down_distance_detection = detection
                        break
            
            if down_distance_detection is None:
                print("   ✗ No down_distance_area detected by YOLO (conf > 0.3)")
                results.append({
                    "test_case": test_case["description"],
                    "expected": test_case["expected_down"],
                    "detected": None,
                    "correct": False,
                    "confidence": 0.0,
                    "template": None,
                    "yolo_found": False,
                    "time_ms": yolo_time * 1000
                })
                continue
            
            # Extract YOLO bbox
            bbox = down_distance_detection["bbox"]
            yolo_confidence = down_distance_detection["confidence"]
            x1, y1, x2, y2 = map(int, bbox)
            yolo_bbox = (x1, y1, x2, y2)  # Keep in (x1, y1, x2, y2) format for detector
            
            print(f"   ✓ YOLO found down_distance_area: conf={yolo_confidence:.3f}, bbox={yolo_bbox}")
            
            # Create context for GOAL situations
            context = None
            if "GOAL" in test_case["expected_template"]:
                context = DownDetectionContext(
                    field_position="GOAL",
                    is_goal_line=True
                )
            
            # Run template detection on YOLO-detected region
            print("   Running template matching on YOLO region...")
            start_template = time.time()
            match = detector.detect_down_in_yolo_region(screenshot, yolo_bbox, context)
            template_time = time.time() - start_template
            
            total_time = yolo_time + template_time
            
            # Analyze results
            expected_down = test_case["expected_down"]
            
            if match:
                detected_down = match.down
                detected_template = match.template_name
                template_confidence = match.confidence
                
                is_correct = (detected_down == expected_down)
                if is_correct:
                    correct_detections += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ WRONG"
                
                print(f"   {status}: Detected {detected_down} (expected {expected_down})")
                print(f"   Template: {detected_template} (confidence: {template_confidence:.3f})")
                print(f"   Template time: {template_time*1000:.1f}ms")
                print(f"   Total time: {total_time*1000:.1f}ms")
                
                results.append({
                    "test_case": test_case["description"],
                    "expected": expected_down,
                    "detected": detected_down,
                    "correct": is_correct,
                    "confidence": template_confidence,
                    "template": detected_template,
                    "yolo_found": True,
                    "yolo_confidence": yolo_confidence,
                    "time_ms": total_time * 1000
                })
            else:
                print(f"   ✗ Template matching failed on YOLO region")
                results.append({
                    "test_case": test_case["description"],
                    "expected": expected_down,
                    "detected": None,
                    "correct": False,
                    "confidence": 0.0,
                    "template": None,
                    "yolo_found": True,
                    "yolo_confidence": yolo_confidence,
                    "time_ms": total_time * 1000
                })
                
        except Exception as e:
            print(f"   ✗ Detection failed: {e}")
            results.append({
                "test_case": test_case["description"],
                "expected": test_case["expected_down"],
                "detected": None,
                "correct": False,
                "confidence": 0.0,
                "template": None,
                "yolo_found": False,
                "time_ms": 0.0
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("REAL YOLO + TEMPLATE MATCHING RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    accuracy = (correct_detections / total_tests) * 100 if total_tests > 0 else 0
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")
    
    yolo_found_count = sum(1 for r in results if r["yolo_found"])
    print(f"YOLO Detection Rate: {yolo_found_count}/{total_tests} ({yolo_found_count/total_tests*100:.1f}%)")
    
    if results:
        valid_times = [r["time_ms"] for r in results if r["time_ms"] > 0]
        if valid_times:
            avg_time = np.mean(valid_times)
            print(f"Average Total Time: {avg_time:.1f}ms")
            print(f"FPS Capability: {1000/avg_time:.1f} FPS")
        
        valid_confidences = [r["confidence"] for r in results if r["confidence"] > 0]
        if valid_confidences:
            avg_confidence = np.mean(valid_confidences)
            print(f"Average Template Confidence: {avg_confidence:.3f}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        yolo_status = "YOLO✓" if result["yolo_found"] else "YOLO✗"
        conf_str = f"{result['confidence']:.3f}" if result['confidence'] > 0 else "N/A"
        template_str = result['template'] or "None"
        print(f"  {status} {yolo_status} {result['test_case']}: {result['detected']} (conf: {conf_str}, template: {template_str})")
    
    return results

if __name__ == "__main__":
    results = test_with_real_yolo()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60) 
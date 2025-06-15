#!/usr/bin/env python3
"""
OCR Diagnostic Test - Isolate the "1st & 10" Default Issue
=========================================================
This test will isolate each step of the OCR extraction process to identify
exactly where the system is defaulting to "1st & 10" instead of extracting
the actual down/distance from the video.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.enhanced_ocr import EnhancedOCR


def test_isolated_ocr_extraction():
    """Test OCR extraction in complete isolation."""
    print("🔍 ISOLATED OCR DIAGNOSTIC TEST")
    print("=" * 50)

    # Load test video
    video_path = "1 min 30 test clip.mp4"
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return

    # Initialize components in isolation
    print("\n📋 STEP 1: Initialize OCR Engine Only")
    try:
        ocr_engine = EnhancedOCR()
        print("✅ OCR engine initialized successfully")
        print(f"   Custom OCR available: {ocr_engine.custom_ocr is not None}")
        print(f"   EasyOCR available: {ocr_engine.reader is not None}")
    except Exception as e:
        print(f"❌ OCR engine initialization failed: {e}")
        return

    # Load video and get frame
    print("\n📋 STEP 2: Load Video Frame")
    cap = cv2.VideoCapture(video_path)

    # Jump to frame 2222 (known to have game content)
    target_frame = 2222
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ Failed to read frame {target_frame}")
        cap.release()
        return

    print(f"✅ Loaded frame {target_frame}, shape: {frame.shape}")

    # Initialize YOLO model separately
    print("\n📋 STEP 3: Initialize YOLO Only")
    try:
        analyzer = EnhancedGameAnalyzer()
        print("✅ Analyzer (with YOLO) initialized")
    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        cap.release()
        return

    # Get YOLO detections
    print("\n📋 STEP 4: Run YOLO Detection Only")
    try:
        # Run inference using the correct method - use the model's detect method
        detections = analyzer.model.detect(frame)

        # Filter for down_distance_area
        down_distance_detections = [
            d for d in detections if d["class_name"] == "down_distance_area"
        ]

        print(f"✅ YOLO found {len(detections)} total detections")
        print(f"✅ Found {len(down_distance_detections)} down_distance_area detections")

        for i, det in enumerate(down_distance_detections):
            print(f"   Detection {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")

    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        cap.release()
        return

    # Extract regions for OCR testing
    print("\n📋 STEP 5: Extract Region for Direct OCR")
    if not down_distance_detections:
        print("❌ No down_distance_area detected - cannot test OCR")
        cap.release()
        return

    # Use the best detection
    best_detection = max(down_distance_detections, key=lambda x: x["confidence"])
    x1, y1, x2, y2 = best_detection["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Extract region
    region = frame[y1:y2, x1:x2]
    print(f"✅ Extracted region: {region.shape} from bbox [{x1},{y1},{x2},{y2}]")

    # Save region for visual inspection
    cv2.imwrite("debug_down_distance_region.png", region)
    print("✅ Saved region as 'debug_down_distance_region.png'")

    # Test 1: Raw OCR extraction
    print("\n📋 STEP 6: Test Raw OCR Extraction")
    try:
        # Test multi-engine OCR directly
        ocr_result = ocr_engine._extract_text_multi_engine(region, "down_distance_area")
        print(f"✅ Multi-engine OCR result:")
        print(f"   Text: '{ocr_result.get('text', 'N/A')}'")
        print(f"   Confidence: {ocr_result.get('confidence', 0):.3f}")
        print(f"   Source: {ocr_result.get('source', 'N/A')}")

    except Exception as e:
        print(f"❌ Multi-engine OCR failed: {e}")
        ocr_result = {"text": "", "confidence": 0.0, "source": "error"}

    # Test 2: Dedicated down/distance extraction
    print("\n📋 STEP 7: Test Dedicated Down/Distance Extraction")
    try:
        down_distance_text = ocr_engine.extract_down_distance(region)
        print(f"✅ Dedicated extraction result: '{down_distance_text}'")

    except Exception as e:
        print(f"❌ Dedicated extraction failed: {e}")
        down_distance_text = None

    # Test 3: Test enhanced analyzer's parsing
    print("\n📋 STEP 8: Test Down/Distance Parsing")
    if down_distance_text:
        try:
            parsed_result = analyzer._parse_down_distance_text(down_distance_text)
            print(f"✅ Parsing result: {parsed_result}")

        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            parsed_result = None
    else:
        print("⚠️  No text to parse")
        parsed_result = None

    # Test 4: Test the full pipeline
    print("\n📋 STEP 9: Test Full Pipeline (Problematic)")
    try:
        # Create region data as the analyzer expects
        region_data = {
            "roi": region,
            "bbox": best_detection["bbox"],
            "confidence": best_detection["confidence"],
        }

        # Test the full extraction method (this is where the problem likely occurs)
        full_result = analyzer._extract_down_distance_from_region(region_data, current_time=None)
        print(f"✅ Full pipeline result: {full_result}")

        if full_result:
            print(f"   Down: {full_result.get('down')}")
            print(f"   Distance: {full_result.get('distance')}")
            print(f"   Method: {full_result.get('method')}")
            print(f"   Confidence: {full_result.get('confidence')}")

            # Check for hybrid corrections
            if full_result.get("hybrid_correction"):
                print(f"   🚨 HYBRID CORRECTION APPLIED!")
                print(f"   Original OCR: {full_result.get('original_ocr')}")
                print(f"   Logic Reasoning: {full_result.get('logic_reasoning')}")

    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        full_result = None

    # Test 5: Check if situational predictor is interfering
    print("\n📋 STEP 10: Test Situational Predictor Isolation")
    if hasattr(analyzer, "situational_predictor"):
        try:
            # Create a minimal game situation
            from spygate.ml.situational_predictor import GameSituation

            test_situation = GameSituation(
                down=None,  # Force unknown
                distance=None,  # Force unknown
                yard_line=35,
                territory="own",
                possession_team="home",
            )

            logic_prediction = analyzer.situational_predictor._predict_from_game_logic(
                test_situation
            )
            print(f"✅ Logic prediction (when OCR fails):")
            print(f"   Predicted down: {logic_prediction.predicted_down}")
            print(f"   Predicted distance: {logic_prediction.predicted_distance}")
            print(f"   Confidence: {logic_prediction.confidence:.3f}")
            print(f"   Reasoning: {logic_prediction.reasoning}")

            # This might be the culprit!
            if logic_prediction.predicted_down == 1 and logic_prediction.predicted_distance == 10:
                print(f"🚨 FOUND THE PROBLEM: Logic predictor defaults to 1st & 10!")

        except Exception as e:
            print(f"❌ Situational predictor test failed: {e}")

    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    if ocr_result.get("text"):
        print(f"✅ OCR extracted text: '{ocr_result['text']}'")
    else:
        print("❌ OCR failed to extract any text")

    if down_distance_text:
        print(f"✅ Dedicated method extracted: '{down_distance_text}'")
    else:
        print("❌ Dedicated method failed")

    if parsed_result:
        print(
            f"✅ Parsing successful: down={parsed_result.get('down')}, distance={parsed_result.get('distance')}"
        )
    else:
        print("❌ Parsing failed")

    if full_result:
        final_down = full_result.get("down")
        final_distance = full_result.get("distance")
        print(f"🎯 FINAL RESULT: down={final_down}, distance={final_distance}")

        if final_down == 1 and final_distance == 10:
            print("🚨 CONFIRMED: System is defaulting to 1st & 10")

            # Identify the source of the default
            if full_result.get("hybrid_correction"):
                print("🔍 DEFAULT SOURCE: Hybrid logic correction")
            elif full_result.get("logic_only"):
                print("🔍 DEFAULT SOURCE: Pure logic fallback")
            elif not ocr_result.get("text"):
                print("🔍 DEFAULT SOURCE: OCR extraction failure")
            else:
                print("🔍 DEFAULT SOURCE: Unknown - needs further investigation")
        else:
            print("✅ System extracted actual down/distance (not defaulting)")
    else:
        print("❌ Full pipeline failed completely")

    cap.release()
    print("\n🏁 Diagnostic test completed!")


if __name__ == "__main__":
    test_isolated_ocr_extraction()

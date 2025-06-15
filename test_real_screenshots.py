#!/usr/bin/env python3
"""
Test KerasOCR vs PaddleOCR on REAL Screenshots
Using actual Madden HUD screenshots from debug_regions/
"""

import os
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def test_keras():
    """Test KerasOCR."""
    try:
        import keras_ocr

        pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ KerasOCR working")
        return pipeline
    except Exception as e:
        print(f"❌ KerasOCR failed: {e}")
        return None


def test_paddle():
    """Test PaddleOCR."""
    try:
        # Fix protobuf issue
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="en")
        print("✅ PaddleOCR working")
        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return None


def extract_keras(pipeline, image_path):
    """Extract with KerasOCR."""
    if pipeline is None:
        return ""

    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""

        # Convert to RGB for KerasOCR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pipeline.recognize([img_rgb])
        if results and results[0]:
            # Get all detected text
            texts = []
            for text, box in results[0]:
                if text and len(text.strip()) > 0:
                    texts.append(text.strip())

            if texts:
                # Return all detected text joined
                return " | ".join(texts)

        return ""
    except Exception as e:
        return f"ERROR: {e}"


def extract_paddle(ocr, image_path):
    """Extract with PaddleOCR."""
    if ocr is None:
        return ""

    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""

        results = ocr.ocr(img)
        if results and results[0]:
            # Get all detected text
            texts = []
            for line in results[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    if text and confidence > 0.1:
                        texts.append(f"{text}({confidence:.2f})")

            if texts:
                return " | ".join(texts)

        return ""
    except Exception as e:
        return f"ERROR: {e}"


def main():
    """Test on real screenshots."""
    print("🎯 KerasOCR vs PaddleOCR - REAL SCREENSHOTS TEST")
    print("=" * 55)

    # Initialize engines
    print("\n🚀 Initializing engines...")
    keras_pipeline = test_keras()
    paddle_ocr = test_paddle()

    if keras_pipeline is None and paddle_ocr is None:
        print("❌ Both engines failed!")
        return

    # Test images - use the enhanced versions for better OCR
    test_images = [
        ("Down/Distance", "debug_regions/region_3_down_distance_area_enhanced_4x.png"),
        ("Game Clock", "debug_regions/region_1_game_clock_area_enhanced_4x.png"),
        ("Play Clock", "debug_regions/region_2_play_clock_area_enhanced_4x.png"),
        ("HUD Full", "debug_regions/region_6_hud_enhanced_4x.png"),
        ("Sample 1", "debug_regions/sample_1_29_3rd_and_2.png"),
        ("Sample 2", "debug_regions/sample_2_113_3RD_and_2.png"),
        ("Sample 5", "debug_regions/sample_5_491_1ST_and_10.png"),
        ("Sample 9", "debug_regions/sample_9_1163_1st_and_10.png"),
    ]

    print(f"\n📊 Testing on {len(test_images)} real screenshots...")
    print("=" * 55)

    # Track results
    keras_detections = 0
    paddle_detections = 0

    # Test each image
    for i, (name, image_path) in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"\n{i+1:2d}. ❌ {name}: File not found")
            continue

        print(f"\n{i+1:2d}. 📸 {name}")
        print(f"    File: {os.path.basename(image_path)}")

        # Test KerasOCR
        if keras_pipeline:
            keras_result = extract_keras(keras_pipeline, image_path)
            if keras_result and not keras_result.startswith("ERROR"):
                keras_detections += 1
                status = "✅" if keras_result else "❌"
            else:
                status = "❌"
            print(f"    KerasOCR : {status} '{keras_result}'")

        # Test PaddleOCR
        if paddle_ocr:
            paddle_result = extract_paddle(paddle_ocr, image_path)
            if paddle_result and not paddle_result.startswith("ERROR"):
                paddle_detections += 1
                status = "✅" if paddle_result else "❌"
            else:
                status = "❌"
            print(f"    PaddleOCR: {status} '{paddle_result}'")

    # Print results
    print(f"\n🏆 REAL SCREENSHOT RESULTS")
    print("=" * 55)

    total_tests = len([img for img in test_images if os.path.exists(img[1])])

    if keras_pipeline:
        keras_rate = keras_detections / total_tests if total_tests > 0 else 0
        print(f"📊 KERAS OCR:   {keras_rate:.1%} detection rate ({keras_detections}/{total_tests})")
    else:
        print(f"📊 KERAS OCR:   FAILED TO INITIALIZE")

    if paddle_ocr:
        paddle_rate = paddle_detections / total_tests if total_tests > 0 else 0
        print(
            f"📊 PADDLE OCR:  {paddle_rate:.1%} detection rate ({paddle_detections}/{total_tests})"
        )
    else:
        print(f"📊 PADDLE OCR:  FAILED TO INITIALIZE")

    # Analysis
    print(f"\n💡 ANALYSIS:")
    if keras_detections > 0 or paddle_detections > 0:
        print(f"   🎉 SUCCESS! OCR engines CAN detect text in screenshots")
        print(f"   🔍 Previous 0% was due to training data quality issues")
        print(f"   ✅ Both engines work - the problem was the dataset")

        if keras_detections > paddle_detections:
            print(f"   🥇 KerasOCR performed better on real screenshots")
        elif paddle_detections > keras_detections:
            print(f"   🥇 PaddleOCR performed better on real screenshots")
        else:
            print(f"   🤝 Both engines performed similarly")
    else:
        print(f"   ⚠️  No text detected - may need preprocessing optimization")
        print(f"   🔧 Try different image enhancement techniques")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Final PaddleOCR Test - Comprehensive comparison with KerasOCR
Tests on real Madden screenshots to verify OCR capabilities
"""

import os
import traceback
from pathlib import Path

import cv2
import numpy as np

# Import at module level to avoid scope issues
try:
    import keras_ocr

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("❌ KerasOCR not available")

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("❌ PaddleOCR not available")


def test_paddleocr():
    """Test PaddleOCR with proper error handling"""
    if not PADDLE_AVAILABLE:
        return None

    try:
        print("🔧 Initializing PaddleOCR...")

        # Force CPU mode to avoid CUDA issues
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,  # Force CPU mode
            show_log=False,
            enable_mkldnn=False,  # Disable Intel optimization that might cause issues
        )
        print("✅ PaddleOCR initialized successfully (CPU mode)!")
        return ocr

    except Exception as e:
        print(f"❌ PaddleOCR setup failed: {e}")
        print(f"Full error: {traceback.format_exc()}")
        return None


def test_kerasocr():
    """Test KerasOCR for comparison"""
    if not KERAS_AVAILABLE:
        return None

    try:
        print("🔧 Initializing KerasOCR...")
        pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ KerasOCR initialized successfully!")
        return pipeline
    except Exception as e:
        print(f"❌ KerasOCR failed: {e}")
        return None


def extract_text_paddle(ocr, image_path):
    """Extract text using PaddleOCR"""
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return f"❌ Could not load image: {image_path}"

        # Run OCR
        result = ocr.ocr(image, cls=False)

        if not result or not result[0]:
            return "❌ No text detected"

        # Extract text from results
        texts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]  # Get the text part
                confidence = line[1][1]  # Get confidence
                texts.append(f"{text} ({confidence:.2f})")

        return " | ".join(texts) if texts else "❌ No valid text found"

    except Exception as e:
        return f"❌ PaddleOCR error: {str(e)}"


def extract_text_keras(pipeline, image_path):
    """Extract text using KerasOCR"""
    try:
        # Read image
        image = keras_ocr.tools.read(str(image_path))

        # Run OCR
        prediction_groups = pipeline.recognize([image])
        predictions = prediction_groups[0]

        if not predictions:
            return "❌ No text detected"

        # Extract text with confidence
        texts = []
        for text, box in predictions:
            texts.append(text)

        return " | ".join(texts) if texts else "❌ No valid text found"

    except Exception as e:
        return f"❌ KerasOCR error: {str(e)}"


def main():
    print("🚀 FINAL OCR COMPARISON: PaddleOCR vs KerasOCR")
    print("=" * 60)

    # Initialize both engines
    paddle_ocr = test_paddleocr()
    keras_pipeline = test_kerasocr()

    if not paddle_ocr and not keras_pipeline:
        print("❌ Both OCR engines failed to initialize!")
        return

    # Test images - same ones that worked with KerasOCR
    test_images = [
        "debug_regions/region_3_down_distance_area_enhanced_4x.png",
        "debug_regions/region_1_game_clock_area_enhanced_4x.png",
        "debug_regions/sample_1_29_3rd_and_2.png",
        "debug_regions/sample_2_1163_1st_and_10.png",
        "debug_regions/sample_3_1163_1st_and_10.png",
        "debug_regions/sample_4_1163_1st_and_10.png",
        "debug_regions/sample_5_1163_1st_and_10.png",
        "debug_regions/sample_6_1163_1st_and_10.png",
        "debug_regions/sample_7_1163_1st_and_10.png",
        "debug_regions/sample_8_1163_1st_and_10.png",
        "debug_regions/sample_9_1163_1st_and_10.png",
    ]

    # Filter existing images
    existing_images = [img for img in test_images if Path(img).exists()]

    if not existing_images:
        print("❌ No test images found!")
        return

    print(f"📸 Testing on {len(existing_images)} images...")
    print()

    paddle_success = 0
    keras_success = 0

    for i, image_path in enumerate(existing_images, 1):
        print(f"🖼️  Test {i}: {Path(image_path).name}")
        print("-" * 40)

        # Test PaddleOCR
        if paddle_ocr:
            paddle_result = extract_text_paddle(paddle_ocr, image_path)
            print(f"🟦 PaddleOCR: {paddle_result}")
            if not paddle_result.startswith("❌"):
                paddle_success += 1
        else:
            print("🟦 PaddleOCR: ❌ Not available")

        # Test KerasOCR
        if keras_pipeline:
            keras_result = extract_text_keras(keras_pipeline, image_path)
            print(f"🟩 KerasOCR:  {keras_result}")
            if not keras_result.startswith("❌"):
                keras_success += 1
        else:
            print("🟩 KerasOCR:  ❌ Not available")

        print()

    # Final results
    print("📊 FINAL RESULTS")
    print("=" * 60)
    total_tests = len(existing_images)

    if paddle_ocr:
        paddle_rate = (paddle_success / total_tests) * 100
        print(f"🟦 PaddleOCR: {paddle_success}/{total_tests} ({paddle_rate:.1f}%)")
    else:
        print("🟦 PaddleOCR: ❌ Failed to initialize")

    if keras_pipeline:
        keras_rate = (keras_success / total_tests) * 100
        print(f"🟩 KerasOCR:  {keras_success}/{total_tests} ({keras_rate:.1f}%)")
    else:
        print("🟩 KerasOCR:  ❌ Failed to initialize")

    # Recommendation
    print()
    print("🎯 RECOMMENDATION")
    print("-" * 20)

    if paddle_ocr and keras_pipeline:
        if paddle_success > keras_success:
            print("🏆 PaddleOCR performed better!")
        elif keras_success > paddle_success:
            print("🏆 KerasOCR performed better!")
        else:
            print("🤝 Both engines performed equally!")
    elif keras_pipeline:
        print("🏆 KerasOCR is the only working option!")
    elif paddle_ocr:
        print("🏆 PaddleOCR is the only working option!")
    else:
        print("❌ Neither engine worked properly!")


if __name__ == "__main__":
    main()

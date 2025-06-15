#!/usr/bin/env python3
"""
Working KerasOCR vs PaddleOCR Comparison
Handles import conflicts properly
"""

import json
import os
import random
import sys

import cv2
import numpy as np


def test_keras_ocr():
    """Test KerasOCR with proper import handling."""
    try:
        # Fix TensorFlow import issues
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")

        import keras_ocr

        print("✅ KerasOCR imported successfully")

        # Initialize pipeline with error handling
        pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ KerasOCR pipeline initialized")

        return pipeline
    except Exception as e:
        print(f"❌ KerasOCR failed: {str(e)[:100]}...")
        return None


def test_paddle_ocr():
    """Test PaddleOCR with proper import handling."""
    try:
        # Suppress PaddlePaddle warnings
        import warnings

        warnings.filterwarnings("ignore")

        # Try different import approaches
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            import paddleocr

            PaddleOCR = paddleocr.PaddleOCR

        print("✅ PaddleOCR imported successfully")

        # Initialize with minimal settings and error handling
        ocr = PaddleOCR(lang="en", show_log=False)
        print("✅ PaddleOCR initialized")

        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR failed: {str(e)[:100]}...")
        return None


def preprocess_for_ocr(image_path):
    """Enhanced preprocessing for OCR."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Get dimensions
        h, w = img.shape[:2]

        # Resize if too small (critical for OCR)
        min_size = 64
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w, 3.0)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhanced preprocessing for dark HUD text
        # 1. Strong brightness boost
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)

        # 2. CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # 3. Noise reduction
        gray = cv2.bilateralFilter(gray, 3, 50, 50)

        # Return both grayscale and color versions
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return img_color, img_rgb
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None


def extract_text_keras(pipeline, image_path):
    """Extract text using KerasOCR."""
    if pipeline is None:
        return ""

    try:
        img_color, img_rgb = preprocess_for_ocr(image_path)
        if img_rgb is None:
            return ""

        # KerasOCR expects RGB format
        results = pipeline.recognize([img_rgb])

        if results and results[0]:
            # Get all detected text with confidence
            texts = []
            for text, box in results[0]:
                if text and len(text.strip()) > 0:
                    texts.append(text.strip())

            if texts:
                # Return the longest text (usually most complete)
                return max(texts, key=len)

        return ""
    except Exception as e:
        print(f"KerasOCR extraction error: {e}")
        return ""


def extract_text_paddle(ocr, image_path):
    """Extract text using PaddleOCR."""
    if ocr is None:
        return ""

    try:
        img_color, img_rgb = preprocess_for_ocr(image_path)
        if img_color is None:
            return ""

        # PaddleOCR can handle BGR format
        results = ocr.ocr(img_color, cls=True)

        if results and results[0]:
            # Get all detected text with confidence
            texts = []
            for line in results[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    if text and confidence > 0.1:
                        texts.append((text, confidence))

            if texts:
                # Return text with highest confidence
                return max(texts, key=lambda x: x[1])[0]

        return ""
    except Exception as e:
        print(f"PaddleOCR extraction error: {e}")
        return ""


def run_comparison():
    """Run the KerasOCR vs PaddleOCR comparison."""
    print("🎯 KerasOCR vs PaddleOCR - EXPERT COMPARISON")
    print("=" * 55)

    # Initialize engines
    print("\n🚀 Initializing OCR engines...")
    keras_pipeline = test_keras_ocr()
    paddle_ocr = test_paddle_ocr()

    # Check if at least one engine works
    working_engines = []
    if keras_pipeline is not None:
        working_engines.append("KerasOCR")
    if paddle_ocr is not None:
        working_engines.append("PaddleOCR")

    if not working_engines:
        print("❌ BOTH OCR engines failed to initialize!")
        print("🔧 Try fixing dependency conflicts first")
        return

    print(f"✅ Working engines: {', '.join(working_engines)}")

    # Load test data
    try:
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} training samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Get exact same test samples as custom OCR
    random.seed(42)
    test_samples = random.sample(data, 15)  # Test 15 samples

    print(f"\n📊 Testing on same 15 samples as failed custom OCR...")
    print("=" * 55)

    # Track results
    keras_results = {"correct": 0, "total": 0, "predictions": []}
    paddle_results = {"correct": 0, "total": 0, "predictions": []}

    # Test each sample
    for i, sample in enumerate(test_samples):
        ground_truth = sample["ground_truth_text"]
        image_path = sample["image_path"]

        print(f"\n{i+1:2d}. GT: '{ground_truth}'")

        # Test KerasOCR
        if keras_pipeline is not None:
            keras_pred = extract_text_keras(keras_pipeline, image_path)
            keras_correct = keras_pred == ground_truth
            keras_results["total"] += 1
            keras_results["predictions"].append(
                {"gt": ground_truth, "pred": keras_pred, "correct": keras_correct}
            )
            if keras_correct:
                keras_results["correct"] += 1

            status = "✅" if keras_correct else "❌"
            print(f"    KerasOCR  : {status} '{keras_pred}'")

        # Test PaddleOCR
        if paddle_ocr is not None:
            paddle_pred = extract_text_paddle(paddle_ocr, image_path)
            paddle_correct = paddle_pred == ground_truth
            paddle_results["total"] += 1
            paddle_results["predictions"].append(
                {"gt": ground_truth, "pred": paddle_pred, "correct": paddle_correct}
            )
            if paddle_correct:
                paddle_results["correct"] += 1

            status = "✅" if paddle_correct else "❌"
            print(f"    PaddleOCR : {status} '{paddle_pred}'")

    # Calculate accuracies
    keras_acc = (
        keras_results["correct"] / keras_results["total"] if keras_results["total"] > 0 else 0
    )
    paddle_acc = (
        paddle_results["correct"] / paddle_results["total"] if paddle_results["total"] > 0 else 0
    )

    # Print final results
    print(f"\n🏆 FINAL COMPARISON RESULTS")
    print("=" * 55)
    print(f"📊 CUSTOM OCR (Baseline): 0.0% (0/20) - Complete failure")

    if keras_results["total"] > 0:
        print(
            f"📊 KERAS OCR:             {keras_acc:.1%} ({keras_results['correct']}/{keras_results['total']})"
        )
    else:
        print(f"📊 KERAS OCR:             FAILED TO INITIALIZE")

    if paddle_results["total"] > 0:
        print(
            f"📊 PADDLE OCR:            {paddle_acc:.1%} ({paddle_results['correct']}/{paddle_results['total']})"
        )
    else:
        print(f"📊 PADDLE OCR:            FAILED TO INITIALIZE")

    # Determine winner and recommendation
    print(f"\n🥇 EXPERT RECOMMENDATION:")
    if keras_results["total"] > 0 and paddle_results["total"] > 0:
        if keras_acc > paddle_acc:
            improvement = keras_acc - paddle_acc
            print(f"   🎉 KERAS OCR WINS! ({keras_acc:.1%} vs {paddle_acc:.1%})")
            print(f"   📈 KerasOCR is {improvement:.1%} better than PaddleOCR")
            print(f"   🚀 INTEGRATE KERAS OCR into SpygateAI")
        elif paddle_acc > keras_acc:
            improvement = paddle_acc - keras_acc
            print(f"   🎉 PADDLE OCR WINS! ({paddle_acc:.1%} vs {keras_acc:.1%})")
            print(f"   📈 PaddleOCR is {improvement:.1%} better than KerasOCR")
            print(f"   🚀 INTEGRATE PADDLE OCR into SpygateAI")
        else:
            print(f"   🤝 TIE! Both achieved {keras_acc:.1%} accuracy")
            print(f"   🎯 Choose based on performance/dependencies")
    elif keras_results["total"] > 0:
        print(f"   🥇 KERAS OCR wins by default (PaddleOCR failed)")
        print(f"   🚀 INTEGRATE KERAS OCR into SpygateAI")
    elif paddle_results["total"] > 0:
        print(f"   🥇 PADDLE OCR wins by default (KerasOCR failed)")
        print(f"   🚀 INTEGRATE PADDLE OCR into SpygateAI")
    else:
        print(f"   ❌ NO WINNER - Both failed")
        print(f"   🔧 Fix dependency conflicts first")

    # Performance analysis
    best_acc = max(keras_acc, paddle_acc)
    if best_acc > 0.7:
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        print(f"   🎉 EXCELLENT! {best_acc:.1%} accuracy is production-ready")
        print(f"   🚀 Ready for immediate integration")
    elif best_acc > 0.5:
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        print(f"   ✅ GOOD! {best_acc:.1%} accuracy shows strong potential")
        print(f"   🎯 Consider fine-tuning for better results")
    elif best_acc > 0.3:
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        print(f"   ⚠️  FAIR! {best_acc:.1%} accuracy needs optimization")
        print(f"   🔧 Focus on preprocessing improvements")
    else:
        print(f"\n💡 PERFORMANCE ANALYSIS:")
        print(f"   ❌ POOR! {best_acc:.1%} accuracy indicates data issues")
        print(f"   🔍 Investigate image quality problems")

    # Save detailed results
    results = {
        "summary": {
            "keras_accuracy": keras_acc,
            "paddle_accuracy": paddle_acc,
            "winner": (
                "keras" if keras_acc > paddle_acc else "paddle" if paddle_acc > keras_acc else "tie"
            ),
            "custom_ocr_baseline": 0.0,
        },
        "keras_results": keras_results,
        "paddle_results": paddle_results,
        "test_samples": [s["ground_truth_text"] for s in test_samples],
    }

    with open("keras_vs_paddle_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: keras_vs_paddle_comparison.json")


if __name__ == "__main__":
    run_comparison()

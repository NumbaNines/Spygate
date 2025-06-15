#!/usr/bin/env python3
"""
Simple KerasOCR vs PaddleOCR Comparison
Handles dependency issues gracefully
"""

import json
import os
import random

import cv2
import numpy as np


def test_keras_ocr():
    """Test KerasOCR if available."""
    try:
        import keras_ocr

        print("✅ KerasOCR imported successfully")

        # Initialize pipeline
        pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ KerasOCR pipeline initialized")

        return pipeline
    except Exception as e:
        print(f"❌ KerasOCR failed: {e}")
        return None


def test_paddle_ocr():
    """Test PaddleOCR if available."""
    try:
        from paddleocr import PaddleOCR

        print("✅ PaddleOCR imported successfully")

        # Initialize with minimal settings
        ocr = PaddleOCR(lang="en")
        print("✅ PaddleOCR initialized")

        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return None


def preprocess_image(image_path):
    """Basic preprocessing for OCR."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Resize if too small
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        scale = max(32 / h, 64 / w, 2.0)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Basic enhancement
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)

    return img


def extract_text_keras(pipeline, image_path):
    """Extract text using KerasOCR."""
    if pipeline is None:
        return ""

    try:
        img = preprocess_image(image_path)
        if img is None:
            return ""

        # KerasOCR expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run OCR
        results = pipeline.recognize([img_rgb])

        if results and results[0]:
            # Get all detected text
            texts = [text for text, box in results[0]]
            if texts:
                return texts[0]  # Return first detected text

        return ""
    except Exception as e:
        print(f"KerasOCR error: {e}")
        return ""


def extract_text_paddle(ocr, image_path):
    """Extract text using PaddleOCR."""
    if ocr is None:
        return ""

    try:
        img = preprocess_image(image_path)
        if img is None:
            return ""

        # Run OCR
        results = ocr.ocr(img)

        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2:
                    text = line[1][0].strip()
                    if text:
                        return text

        return ""
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return ""


def run_comparison():
    """Run the comparison test."""
    print("🎯 KerasOCR vs PaddleOCR Comparison")
    print("=" * 50)

    # Initialize both engines
    print("\n🚀 Initializing OCR engines...")
    keras_pipeline = test_keras_ocr()
    paddle_ocr = test_paddle_ocr()

    if keras_pipeline is None and paddle_ocr is None:
        print("❌ Both OCR engines failed to initialize!")
        return

    # Load test data
    try:
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} training samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Get same test samples as custom OCR
    random.seed(42)
    test_samples = random.sample(data, 10)  # Test 10 samples

    print(f"\n📊 Testing on same 10 samples as failed custom OCR...")
    print("=" * 50)

    # Track results
    keras_results = {"correct": 0, "total": 0, "predictions": []}
    paddle_results = {"correct": 0, "total": 0, "predictions": []}

    # Test each sample
    for i, sample in enumerate(test_samples):
        ground_truth = sample["ground_truth_text"]
        image_path = sample["image_path"]

        print(f"\n{i+1:2d}. Testing: '{ground_truth}'")

        # Test KerasOCR
        if keras_pipeline is not None:
            keras_pred = extract_text_keras(keras_pipeline, image_path)
            keras_correct = keras_pred == ground_truth
            keras_results["total"] += 1
            keras_results["predictions"].append(keras_pred)
            if keras_correct:
                keras_results["correct"] += 1

            status = "✅" if keras_correct else "❌"
            print(f"    KerasOCR  : {status} '{keras_pred}'")

        # Test PaddleOCR
        if paddle_ocr is not None:
            paddle_pred = extract_text_paddle(paddle_ocr, image_path)
            paddle_correct = paddle_pred == ground_truth
            paddle_results["total"] += 1
            paddle_results["predictions"].append(paddle_pred)
            if paddle_correct:
                paddle_results["correct"] += 1

            status = "✅" if paddle_correct else "❌"
            print(f"    PaddleOCR : {status} '{paddle_pred}'")

    # Print final results
    print(f"\n🏆 FINAL COMPARISON RESULTS")
    print("=" * 50)
    print(f"📊 CUSTOM OCR (Previous): 0.0% (0/20) - Complete failure")

    if keras_results["total"] > 0:
        keras_acc = keras_results["correct"] / keras_results["total"]
        print(
            f"📊 KERAS OCR:             {keras_acc:.1%} ({keras_results['correct']}/{keras_results['total']})"
        )
    else:
        print(f"📊 KERAS OCR:             FAILED TO INITIALIZE")

    if paddle_results["total"] > 0:
        paddle_acc = paddle_results["correct"] / paddle_results["total"]
        print(
            f"📊 PADDLE OCR:            {paddle_acc:.1%} ({paddle_results['correct']}/{paddle_results['total']})"
        )
    else:
        print(f"📊 PADDLE OCR:            FAILED TO INITIALIZE")

    # Determine winner
    print(f"\n🥇 WINNER:")
    if keras_results["total"] > 0 and paddle_results["total"] > 0:
        if keras_acc > paddle_acc:
            print(f"   KerasOCR wins with {keras_acc:.1%} accuracy!")
        elif paddle_acc > keras_acc:
            print(f"   PaddleOCR wins with {paddle_acc:.1%} accuracy!")
        else:
            print(f"   TIE! Both achieved {keras_acc:.1%} accuracy")
    elif keras_results["total"] > 0:
        print(f"   KerasOCR wins by default (PaddleOCR failed)")
    elif paddle_results["total"] > 0:
        print(f"   PaddleOCR wins by default (KerasOCR failed)")
    else:
        print(f"   NO WINNER - Both failed to initialize")

    # Save results
    results = {
        "keras": keras_results,
        "paddle": paddle_results,
        "test_samples": [s["ground_truth_text"] for s in test_samples],
    }

    with open("keras_vs_paddle_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: keras_vs_paddle_results.json")


if __name__ == "__main__":
    run_comparison()

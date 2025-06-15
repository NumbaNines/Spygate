#!/usr/bin/env python3
"""
Simple KerasOCR vs PaddleOCR Comparison - Fixed Version
"""

import json
import random
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
        import paddleocr

        ocr = paddleocr.PaddleOCR(lang="en", show_log=False)
        print("✅ PaddleOCR working")
        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return None


def preprocess(image_path):
    """Basic preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if h < 64 or w < 64:
        scale = max(64 / h, 128 / w, 3.0)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Enhance for dark text
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return img_color, img_rgb


def extract_keras(pipeline, image_path):
    """Extract with KerasOCR."""
    if pipeline is None:
        return ""

    try:
        img_color, img_rgb = preprocess(image_path)
        if img_rgb is None:
            return ""

        results = pipeline.recognize([img_rgb])
        if results and results[0]:
            texts = [text for text, box in results[0]]
            if texts:
                return texts[0].strip()
        return ""
    except Exception as e:
        return ""


def extract_paddle(ocr, image_path):
    """Extract with PaddleOCR."""
    if ocr is None:
        return ""

    try:
        img_color, img_rgb = preprocess(image_path)
        if img_color is None:
            return ""

        results = ocr.ocr(img_color)
        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2:
                    text = line[1][0].strip()
                    if text:
                        return text
        return ""
    except Exception as e:
        return ""


def main():
    """Run comparison."""
    print("🎯 KerasOCR vs PaddleOCR Comparison")
    print("=" * 40)

    # Test engines
    keras_pipeline = test_keras()
    paddle_ocr = test_paddle()

    if keras_pipeline is None and paddle_ocr is None:
        print("❌ Both failed!")
        return

    # Load data
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    random.seed(42)
    test_samples = random.sample(data, 10)

    print(f"\n📊 Testing 10 samples...")

    keras_correct = 0
    paddle_correct = 0

    for i, sample in enumerate(test_samples):
        gt = sample["ground_truth_text"]
        path = sample["image_path"]

        print(f"\n{i+1}. GT: '{gt}'")

        if keras_pipeline:
            keras_pred = extract_keras(keras_pipeline, path)
            keras_ok = keras_pred == gt
            if keras_ok:
                keras_correct += 1
            status = "✅" if keras_ok else "❌"
            print(f"   Keras : {status} '{keras_pred}'")

        if paddle_ocr:
            paddle_pred = extract_paddle(paddle_ocr, path)
            paddle_ok = paddle_pred == gt
            if paddle_ok:
                paddle_correct += 1
            status = "✅" if paddle_ok else "❌"
            print(f"   Paddle: {status} '{paddle_pred}'")

    print(f"\n🏆 RESULTS:")
    print(f"Custom OCR: 0.0% (failed)")

    if keras_pipeline:
        keras_acc = keras_correct / 10
        print(f"KerasOCR:   {keras_acc:.1%} ({keras_correct}/10)")

    if paddle_ocr:
        paddle_acc = paddle_correct / 10
        print(f"PaddleOCR:  {paddle_acc:.1%} ({paddle_correct}/10)")


if __name__ == "__main__":
    main()

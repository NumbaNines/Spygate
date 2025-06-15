#!/usr/bin/env python3
"""
Minimal PaddleOCR Test - Just the basics
"""

import json
import random

import cv2
import numpy as np
from paddleocr import PaddleOCR


def test_paddle_minimal():
    print("🚀 Testing PaddleOCR with minimal setup...")

    # Initialize with minimal parameters
    try:
        ocr = PaddleOCR(lang="en")
        print("✅ PaddleOCR initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PaddleOCR: {e}")
        return

    # Load test data
    try:
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} training samples")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Get same test samples
    random.seed(42)
    test_samples = random.sample(data, 5)  # Start with just 5 samples

    print(f"\n🎯 Testing PaddleOCR on 5 samples...")
    print("=" * 50)

    correct = 0
    total = 0

    for i, sample in enumerate(test_samples):
        ground_truth = sample["ground_truth_text"]
        image_path = sample["image_path"]

        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"{i+1}. ❌ Could not load image: {image_path}")
                continue

            # Basic preprocessing
            h, w = img.shape[:2]
            if h < 32 or w < 32:
                scale = max(32 / h, 64 / w, 2.0)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))

            # Run OCR
            results = ocr.ocr(img)

            # Extract text
            prediction = ""
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0].strip()
                        if text:
                            prediction = text
                            break

            # Check result
            is_correct = prediction == ground_truth
            if is_correct:
                correct += 1
            total += 1

            status = "✅" if is_correct else "❌"
            print(f"{i+1}. {status} GT: '{ground_truth}' | Pred: '{prediction}'")

        except Exception as e:
            print(f"{i+1}. ❌ Error processing {image_path}: {e}")
            total += 1

    # Results
    if total > 0:
        accuracy = correct / total
        print(f"\n🏆 RESULTS:")
        print(f"   Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"   Custom OCR was: 0.0% (complete failure)")
        print(f"   Improvement: +{accuracy:.1%}")

        if accuracy > 0.6:
            print(f"   🎉 EXCELLENT! PaddleOCR works well")
        elif accuracy > 0.3:
            print(f"   ✅ GOOD! Much better than custom")
        elif accuracy > 0:
            print(f"   ⚠️  FAIR! Some improvement")
        else:
            print(f"   ❌ POOR! No improvement")
    else:
        print(f"❌ No samples processed successfully")


if __name__ == "__main__":
    test_paddle_minimal()

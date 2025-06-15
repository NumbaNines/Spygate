#!/usr/bin/env python3
"""
Quick OCR test on a few samples to see immediate results.
"""

import json
import random

import cv2
import easyocr
import numpy as np


def preprocess_image(image_path):
    """Same preprocessing as our custom model."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Resize to standard size
        img = cv2.resize(img, (128, 32))

        # Enhanced preprocessing for dark text
        img = cv2.convertScaleAbs(img, alpha=2.5, beta=40)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        img = clahe.apply(img)
        gamma = 1.2
        img = np.power(img / 255.0, 1.0 / gamma) * 255.0
        img = img.astype(np.uint8)

        return img
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("🔍 Quick OCR Test")
    print("=" * 40)

    # Load data
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    # Get same samples as before
    random.seed(42)
    test_samples = random.sample(data, 5)

    # Initialize EasyOCR
    reader = easyocr.Reader(["en"], gpu=True)

    print("Testing EasyOCR on 5 samples:")

    for i, sample in enumerate(test_samples):
        gt = sample["ground_truth_text"]
        img_path = sample["image_path"]

        # Preprocess
        img = preprocess_image(img_path)
        if img is None:
            continue

        # OCR
        try:
            results = reader.readtext(img)
            if results:
                pred = max(results, key=lambda x: x[2])[1].strip()
            else:
                pred = ""
        except:
            pred = "ERROR"

        status = "✅" if pred == gt else "❌"
        print(f"{i+1}. {status} GT: '{gt}' | EasyOCR: '{pred}'")

    print("\n💡 This gives us a quick baseline while full benchmark runs!")


if __name__ == "__main__":
    main()

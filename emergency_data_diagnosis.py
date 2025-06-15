#!/usr/bin/env python3
"""
EMERGENCY Data Diagnosis - Check for data corruption
"""

import json
import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np


def emergency_data_check():
    print("🚨 EMERGENCY DATA DIAGNOSIS")
    print("=" * 60)

    # Load data
    with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data):,}")

    # Check 1: Text distribution
    print("\n🔍 TEXT ANALYSIS:")
    texts = [sample["ground_truth_text"] for sample in data]
    text_counter = Counter(texts)

    print(f"Unique texts: {len(text_counter)}")
    print("Most common texts:")
    for text, count in text_counter.most_common(10):
        print(f"  '{text}': {count:,} times ({count/len(data)*100:.1f}%)")

    # Check 2: Image file existence
    print("\n🔍 IMAGE FILE CHECK:")
    missing_files = 0
    corrupted_images = 0
    valid_images = 0

    for i, sample in enumerate(data[:100]):  # Check first 100
        img_path = sample["image_path"]

        if not os.path.exists(img_path):
            missing_files += 1
            continue

        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                corrupted_images += 1
            else:
                valid_images += 1
        except Exception as e:
            corrupted_images += 1

    print(f"Valid images: {valid_images}/100")
    print(f"Missing files: {missing_files}/100")
    print(f"Corrupted images: {corrupted_images}/100")

    # Check 3: Character analysis
    print("\n🔍 CHARACTER ANALYSIS:")
    all_chars = set()
    for text in texts:
        all_chars.update(text)

    sorted_chars = sorted(list(all_chars))
    print(f"Unique characters: {len(sorted_chars)}")
    print(f"Characters: {sorted_chars}")

    # Check 4: Text length distribution
    print("\n🔍 TEXT LENGTH ANALYSIS:")
    lengths = [len(text) for text in texts]
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {np.mean(lengths):.1f}")

    length_counter = Counter(lengths)
    print("Length distribution:")
    for length, count in sorted(length_counter.items()):
        print(f"  {length} chars: {count:,} samples ({count/len(data)*100:.1f}%)")

    # Check 5: Suspicious patterns
    print("\n🚨 SUSPICIOUS PATTERN CHECK:")

    # Check for identical consecutive samples
    identical_consecutive = 0
    for i in range(1, len(data)):
        if data[i]["ground_truth_text"] == data[i - 1]["ground_truth_text"]:
            identical_consecutive += 1

    print(
        f"Identical consecutive texts: {identical_consecutive:,} ({identical_consecutive/len(data)*100:.1f}%)"
    )

    # Check for empty or very short texts
    empty_texts = sum(1 for text in texts if len(text.strip()) == 0)
    very_short = sum(1 for text in texts if len(text.strip()) <= 2)

    print(f"Empty texts: {empty_texts}")
    print(f"Very short texts (≤2 chars): {very_short}")

    # Check 6: Sample a few images and texts
    print("\n🔍 SAMPLE VERIFICATION:")
    import random

    sample_indices = random.sample(range(len(data)), min(5, len(data)))

    for idx in sample_indices:
        sample = data[idx]
        img_path = sample["image_path"]
        text = sample["ground_truth_text"]

        print(f"Sample {idx}:")
        print(f"  Text: '{text}'")
        print(f"  Path: {img_path}")
        print(f"  Exists: {os.path.exists(img_path)}")

        if os.path.exists(img_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"  Image shape: {img.shape}")
                else:
                    print(f"  Image: CORRUPTED")
            except:
                print(f"  Image: ERROR LOADING")

    # Check 7: Data quality score
    print("\n📊 DATA QUALITY SCORE:")

    quality_issues = 0

    # Issue 1: Too many identical texts
    if text_counter.most_common(1)[0][1] > len(data) * 0.1:  # >10% identical
        quality_issues += 1
        print("❌ Too many identical texts")

    # Issue 2: Missing/corrupted files
    if missing_files > 5 or corrupted_images > 5:
        quality_issues += 1
        print("❌ Missing or corrupted image files")

    # Issue 3: Empty or very short texts
    if empty_texts > 0 or very_short > len(data) * 0.05:  # >5% very short
        quality_issues += 1
        print("❌ Too many empty or very short texts")

    # Issue 4: Suspicious character set
    if len(sorted_chars) < 20 or len(sorted_chars) > 100:
        quality_issues += 1
        print("❌ Suspicious character set size")

    if quality_issues == 0:
        print("✅ Data appears healthy")
    else:
        print(f"🚨 {quality_issues} quality issues detected!")

    print("\n" + "=" * 60)
    print("🚨 EMERGENCY DIAGNOSIS COMPLETE")

    return {
        "total_samples": len(data),
        "unique_texts": len(text_counter),
        "most_common_text": text_counter.most_common(1)[0],
        "quality_issues": quality_issues,
        "character_count": len(sorted_chars),
        "characters": sorted_chars,
    }


if __name__ == "__main__":
    result = emergency_data_check()

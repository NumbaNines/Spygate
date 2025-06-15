#!/usr/bin/env python3
"""
Analyze training data quality to identify why model learned incorrect patterns.
"""

import json
import os
import sqlite3
from collections import Counter, defaultdict


def analyze_training_database():
    """Analyze the training database for data quality issues."""

    print("🔍 Analyzing Training Data Quality")
    print("=" * 60)

    db_path = "madden_ocr_training.db"
    if not os.path.exists(db_path):
        print(f"❌ Training database not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get total samples
        cursor.execute("SELECT COUNT(*) FROM training_samples")
        total_samples = cursor.fetchone()[0]
        print(f"📊 Total training samples: {total_samples:,}")

        # Analyze text patterns
        cursor.execute("SELECT text FROM training_samples WHERE text IS NOT NULL AND text != ''")
        texts = [row[0] for row in cursor.fetchall()]

        print(f"\n📝 Text Analysis:")
        print(f"   Valid text samples: {len(texts):,}")
        print(f"   Unique texts: {len(set(texts)):,}")

        # Character frequency analysis
        all_chars = "".join(texts)
        char_freq = Counter(all_chars)

        print(f"\n🔤 Character Frequency (Top 20):")
        for char, count in char_freq.most_common(20):
            char_display = repr(char) if char in [" ", "\n", "\t"] else char
            print(f"   {char_display}: {count:,} ({count/len(all_chars)*100:.1f}%)")

        # Look for suspicious patterns
        print(f"\n🚨 Suspicious Patterns:")

        # Check for repeated characters
        repeated_chars = [text for text in texts if len(set(text)) == 1 and len(text) > 1]
        if repeated_chars:
            print(f"   Repeated character texts: {len(repeated_chars)}")
            print(f"   Examples: {repeated_chars[:5]}")

        # Check for very long texts
        long_texts = [text for text in texts if len(text) > 20]
        if long_texts:
            print(f"   Very long texts (>20 chars): {len(long_texts)}")
            print(f"   Examples: {long_texts[:3]}")

        # Check for common patterns
        text_patterns = defaultdict(int)
        for text in texts:
            if "K1tC" in text.upper():
                text_patterns["Contains K1tC"] += 1
            if "&" in text:
                text_patterns["Contains &"] += 1
            if ":" in text:
                text_patterns["Contains :"] += 1
            if text.isdigit():
                text_patterns["Pure numbers"] += 1
            if text.isalpha():
                text_patterns["Pure letters"] += 1

        print(f"\n📈 Text Pattern Analysis:")
        for pattern, count in text_patterns.items():
            print(f"   {pattern}: {count:,} ({count/len(texts)*100:.1f}%)")

        # Sample some actual training data
        cursor.execute(
            "SELECT text, region_type FROM training_samples WHERE text IS NOT NULL ORDER BY RANDOM() LIMIT 20"
        )
        samples = cursor.fetchall()

        print(f"\n🎯 Random Training Samples:")
        for i, (text, region_type) in enumerate(samples[:10], 1):
            print(f"   {i:2d}. '{text}' ({region_type})")

        conn.close()

    except Exception as e:
        print(f"❌ Database analysis failed: {e}")
        import traceback

        traceback.print_exc()


def analyze_character_mappings():
    """Analyze the character mappings used in the model."""

    print("\n🔤 Analyzing Character Mappings")
    print("=" * 60)

    model_path = "models/fixed_ocr_20250614_150024/best_fixed_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")

        char_to_idx = checkpoint.get("char_to_idx", {})
        idx_to_char = checkpoint.get("idx_to_char", {})

        print(f"📊 Character Mapping Info:")
        print(f"   Vocabulary size: {len(char_to_idx)}")
        print(f"   Character to index mappings: {len(char_to_idx)}")
        print(f"   Index to character mappings: {len(idx_to_char)}")

        print(f"\n🔤 Character Set:")
        sorted_chars = sorted(char_to_idx.items(), key=lambda x: x[1])
        for char, idx in sorted_chars:
            char_display = repr(char) if char in [" ", "\n", "\t"] else char
            print(f"   {idx:2d}: {char_display}")

        # Check for suspicious mappings
        print(f"\n🚨 Potential Issues:")

        # Check if common characters are missing
        expected_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ&:- ")
        actual_chars = set(char_to_idx.keys())
        missing_chars = expected_chars - actual_chars
        extra_chars = actual_chars - expected_chars

        if missing_chars:
            print(f"   Missing expected characters: {sorted(missing_chars)}")
        if extra_chars:
            print(f"   Extra unexpected characters: {sorted(extra_chars)}")

        # Check for duplicate mappings
        idx_counts = Counter(char_to_idx.values())
        duplicates = [idx for idx, count in idx_counts.items() if count > 1]
        if duplicates:
            print(f"   Duplicate index mappings: {duplicates}")

    except Exception as e:
        print(f"❌ Character mapping analysis failed: {e}")
        import traceback

        traceback.print_exc()


def check_training_json():
    """Check the training JSON file for data quality."""

    print("\n📄 Analyzing Training JSON Data")
    print("=" * 60)

    json_files = [
        "madden_ocr_training_data_20250614_120830.json",
        "madden_ocr_training_data_20250614_112050.json",
    ]

    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"⚠️  JSON file not found: {json_file}")
            continue

        try:
            print(f"\n📁 Analyzing: {json_file}")
            with open(json_file, "r") as f:
                data = json.load(f)

            print(f"   Total entries: {len(data):,}")

            # Analyze text quality
            texts = [item.get("text", "") for item in data if item.get("text")]
            print(f"   Valid text entries: {len(texts):,}")

            # Check for K1tC pattern in training data
            k1tc_count = sum(1 for text in texts if "K1tC" in text.upper())
            if k1tc_count > 0:
                print(f"   🚨 FOUND K1tC in training data: {k1tc_count} samples")
                k1tc_samples = [text for text in texts if "K1tC" in text.upper()][:5]
                print(f"   Examples: {k1tc_samples}")
            else:
                print(f"   ✅ No K1tC pattern found in training data")

            # Sample some entries
            sample_texts = texts[:10] if texts else []
            print(f"   Sample texts: {sample_texts}")

        except Exception as e:
            print(f"❌ JSON analysis failed for {json_file}: {e}")


def main():
    """Main analysis function."""

    print("🔍 Training Data Quality Analysis")
    print("=" * 60)
    print("Investigating why model learned 'K1tC' pattern...")

    # Analyze training database
    analyze_training_database()

    # Analyze character mappings
    analyze_character_mappings()

    # Check training JSON files
    check_training_json()

    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n💡 LIKELY ROOT CAUSES:")
    print("   1. Training data contains mislabeled samples with 'K1tC' pattern")
    print("   2. OCR preprocessing during training created artifacts")
    print("   3. Model overfitted to incorrect training examples")
    print("   4. Character mapping issues during training")
    print("   5. Insufficient training data diversity")
    print("\n🔧 IMMEDIATE FIXES:")
    print("   1. Clean training data - remove samples with 'K1tC' pattern")
    print("   2. Retrain model with cleaned data")
    print("   3. Add more diverse training samples")
    print("   4. Increase training epochs")
    print("   5. Add validation to catch such issues early")


if __name__ == "__main__":
    main()

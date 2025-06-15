#!/usr/bin/env python3
"""
Check Character Coverage in Training Data
Analyze if any important characters are missing or underrepresented
"""

import json
from collections import Counter

import matplotlib.pyplot as plt


def analyze_character_coverage():
    print("🔍 Analyzing Character Coverage in Training Data")
    print("=" * 60)

    # Load training data
    with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data):,}")

    # Count all characters
    all_chars = Counter()
    class_chars = {}

    for sample in data:
        text = sample.get("ground_truth_text", "")
        class_name = sample.get("class_name", "unknown")

        # Count characters globally
        for char in text:
            all_chars[char] += 1

        # Count characters per class
        if class_name not in class_chars:
            class_chars[class_name] = Counter()
        for char in text:
            class_chars[class_name][char] += 1

    print(f"\n📈 Character Statistics:")
    print(f"  - Unique characters found: {len(all_chars)}")
    print(f"  - Total character instances: {sum(all_chars.values()):,}")

    # Expected Madden characters
    expected_chars = set(" &-0123456789:;ACDFGHIKLOPRSTadhlnorst")
    found_chars = set(all_chars.keys())

    print(f"\n🎯 Character Coverage Analysis:")
    print(f"  - Expected characters: {len(expected_chars)}")
    print(f"  - Found characters: {len(found_chars)}")

    # Check for missing expected characters
    missing_chars = expected_chars - found_chars
    if missing_chars:
        print(f"  ❌ Missing expected chars: {sorted(missing_chars)}")
    else:
        print(f"  ✅ All expected characters found!")

    # Check for unexpected characters
    unexpected_chars = found_chars - expected_chars
    if unexpected_chars:
        print(f"  ⚠️ Unexpected chars found: {sorted(unexpected_chars)}")
    else:
        print(f"  ✅ No unexpected characters!")

    # Analyze digit coverage specifically
    print(f"\n🔢 Digit Coverage Analysis:")
    digits = "0123456789"
    for digit in digits:
        count = all_chars.get(digit, 0)
        percentage = (count / sum(all_chars.values())) * 100
        status = "✅" if count > 50 else "⚠️" if count > 10 else "❌"
        print(f"  {status} Digit '{digit}': {count:,} times ({percentage:.2f}%)")

    # Analyze letter coverage
    print(f"\n🔤 Letter Coverage Analysis:")
    letters = "STNDRDTHGOALqtrFLAGACDFGHIKLOPRSTadhlnorst"
    for letter in sorted(set(letters)):
        count = all_chars.get(letter, 0)
        percentage = (count / sum(all_chars.values())) * 100
        status = "✅" if count > 20 else "⚠️" if count > 5 else "❌"
        print(f"  {status} Letter '{letter}': {count:,} times ({percentage:.2f}%)")

    # Class-specific analysis
    print(f"\n📊 Character Distribution by Class:")
    for class_name, char_counter in class_chars.items():
        print(f"\n  📁 {class_name}:")
        print(f"    - Samples: {sum(1 for s in data if s.get('class_name') == class_name):,}")
        print(f"    - Unique chars: {len(char_counter)}")

        # Show top characters for this class
        top_chars = char_counter.most_common(10)
        print(f"    - Top chars: {[(char, count) for char, count in top_chars]}")

        # Check for class-specific missing digits
        class_digits = set(char for char in char_counter.keys() if char.isdigit())
        missing_digits = set(digits) - class_digits
        if missing_digits:
            print(f"    ❌ Missing digits: {sorted(missing_digits)}")
        else:
            print(f"    ✅ All digits present")

    # Identify potential problems
    print(f"\n⚠️ Potential Issues:")

    # Low-frequency characters
    low_freq_chars = [char for char, count in all_chars.items() if count < 10 and char.isalnum()]
    if low_freq_chars:
        print(f"  - Low frequency chars (<10 times): {low_freq_chars}")
        print(f"    → These might be poorly learned by the model")

    # Class imbalance for important characters
    important_chars = "0123456789STNDGOAL"
    for char in important_chars:
        class_counts = {}
        for class_name, char_counter in class_chars.items():
            class_counts[class_name] = char_counter.get(char, 0)

        if any(count > 0 for count in class_counts.values()):
            max_count = max(class_counts.values())
            min_count = min(count for count in class_counts.values() if count > 0)
            if max_count > min_count * 10:  # 10x imbalance
                print(f"  - Char '{char}' imbalanced across classes: {class_counts}")

    # Save detailed analysis
    analysis = {
        "total_samples": len(data),
        "character_counts": dict(all_chars),
        "class_character_counts": {k: dict(v) for k, v in class_chars.items()},
        "missing_expected": list(missing_chars),
        "unexpected_found": list(unexpected_chars),
    }

    with open("character_coverage_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n💾 Detailed analysis saved to: character_coverage_analysis.json")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Character frequency
    plt.subplot(2, 2, 1)
    chars, counts = zip(*all_chars.most_common(20))
    plt.bar(chars, counts)
    plt.title("Top 20 Character Frequencies")
    plt.xlabel("Character")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    # Plot 2: Digit distribution
    plt.subplot(2, 2, 2)
    digit_counts = [all_chars.get(d, 0) for d in digits]
    plt.bar(list(digits), digit_counts)
    plt.title("Digit Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Count")

    # Plot 3: Class distribution
    plt.subplot(2, 2, 3)
    class_names = list(class_chars.keys())
    class_sample_counts = [
        sum(1 for s in data if s.get("class_name") == name) for name in class_names
    ]
    plt.bar(class_names, class_sample_counts)
    plt.title("Samples per Class")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=45)

    # Plot 4: Character diversity per class
    plt.subplot(2, 2, 4)
    class_char_diversity = [len(class_chars[name]) for name in class_names]
    plt.bar(class_names, class_char_diversity)
    plt.title("Character Diversity per Class")
    plt.xlabel("Class")
    plt.ylabel("Unique Characters")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("character_coverage_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Visualization saved to: character_coverage_analysis.png")


if __name__ == "__main__":
    analyze_character_coverage()

#!/usr/bin/env python3
"""
Simple analysis of fixed training data.
"""

import json
import re
from collections import Counter


def main():
    print("🔍 Analyzing Fixed Training Data")
    print("=" * 50)

    # Load fixed data
    with open("madden_ocr_training_data_FIXED.json", "r") as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data):,}")

    # Count patterns
    text_counter = Counter([s["ground_truth_text"] for s in data])
    print(f"📊 Unique patterns: {len(text_counter)}")

    # Show distribution
    print(f"\n📊 Top 20 patterns:")
    for text, count in text_counter.most_common(20):
        print(f"   '{text}': {count}")

    # Categorize patterns
    down_distance = []
    scores = []
    times = []
    special = []
    other = []

    for text, count in text_counter.items():
        text_upper = text.upper()

        if "&" in text_upper and any(x in text_upper for x in ["ST", "ND", "RD", "TH"]):
            down_distance.append((text, count))
        elif re.match(r"^\d+$", text):
            scores.append((text, count))
        elif ":" in text or ("TH " in text_upper and ":" in text):
            times.append((text, count))
        elif text_upper in ["KICKOFF", "PAT", "GOAL", "--", "2-PT"]:
            special.append((text, count))
        else:
            other.append((text, count))

    print(f"\n📝 Pattern Categories:")
    print(f"   Down & Distance: {len(down_distance)} patterns")
    print(f"   Scores: {len(scores)} patterns")
    print(f"   Times: {len(times)} patterns")
    print(f"   Special: {len(special)} patterns")
    print(f"   Other: {len(other)} patterns")

    # Check if we have too many patterns
    total_patterns = len(text_counter)
    print(f"\n⚠️  ANALYSIS:")

    if total_patterns > 100:
        print(f"   🚨 TOO MANY PATTERNS: {total_patterns}")
        print(f"   📊 Recommendation: Limit to 50-80 core patterns")
        print(f"   🎯 Focus on most common game situations")
    elif total_patterns > 80:
        print(f"   ⚠️  MANY PATTERNS: {total_patterns}")
        print(f"   📊 Consider limiting to 60-70 patterns")
    else:
        print(f"   ✅ GOOD: {total_patterns} patterns is manageable")

    # Show score range
    score_numbers = []
    for text, count in scores:
        try:
            score_numbers.append(int(text))
        except:
            pass

    if score_numbers:
        print(f"\n📊 Score range: {min(score_numbers)} to {max(score_numbers)}")
        if max(score_numbers) > 50:
            print(f"   ⚠️  High scores detected (>{max(score_numbers)})")
            print(f"   📊 Consider limiting to 0-35 range")

    print(f"\n💡 RECOMMENDATIONS:")
    if total_patterns > 80:
        print(f"   1. Create LIMITED dataset with core patterns only")
        print(f"   2. Focus on: 1ST-4TH & 1-10, scores 0-35, common times")
        print(f"   3. Limit each pattern to 30-50 samples")
        print(f"   4. Target: ~50 total patterns for better learning")
    else:
        print(f"   1. Current dataset size is acceptable")
        print(f"   2. Proceed with training")


if __name__ == "__main__":
    main()

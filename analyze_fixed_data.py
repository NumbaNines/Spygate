#!/usr/bin/env python3
"""
Analyze the fixed training data to determine if we need further pattern limiting.
"""

import json
import re
from collections import Counter


def analyze_fixed_data():
    """Analyze the fixed training data distribution and patterns."""

    print("🔍 Analyzing Fixed Training Data")
    print("=" * 60)

    # Load fixed data
    with open('madden_ocr_training_data_FIXED.json', 'r') as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data):,}")

    # Analyze distribution
    text_counter = Counter([s['ground_truth_text'] for s in data])

    print(f"📊 Unique patterns: {len(text_counter)}")
    print(f"📊 Patterns with 100 samples: {sum(1 for count in text_counter.values() if count == 100)}")
    print(f"📊 Patterns with <100 samples: {sum(1 for count in text_counter.values() if count < 100)}")

    # Show patterns with fewer samples
    print(f"\n🔍 Patterns with <100 samples:")
    for text, count in sorted(text_counter.items(), key=lambda x: x[1]):
        if count < 100:
            print(f"   '{text}': {count}")

    # Categorize patterns
    print(f"\n📝 Pattern Categories:")

    categories = {
        'down_distance': [],
        'scores': [],
        'times': [],
        'special': [],
        'other': []
    }

    for text, count in text_counter.items():
        text_upper = text.upper()

        # Down & Distance patterns
        if re.match(r'\\d+(ST|ND|RD|TH)\\s*&\\s*(\\d+|GOAL)', text_upper):\n            categories['down_distance'].append((text, count))\n        # Score patterns (just numbers)\n        elif re.match(r'^\\d+$', text):\n            categories['scores'].append((text, count))\n        # Time patterns\n        elif re.match(r'\\d+:\\d+', text) or 'TH ' in text_upper:\n            categories['times'].append((text, count))\n        # Special situations\n        elif text_upper in ['KICKOFF', 'PAT', 'GOAL', '--', '2-PT']:\n            categories['special'].append((text, count))\n        else:\n            categories['other'].append((text, count))\n    \n    for category, patterns in categories.items():\n        if patterns:\n            print(f\"\\n🏷️  {category.upper()} ({len(patterns)} patterns):\")\n            for text, count in sorted(patterns, key=lambda x: x[1], reverse=True)[:10]:\n                print(f\"   '{text}': {count}\")\n            if len(patterns) > 10:\n                print(f\"   ... and {len(patterns) - 10} more\")\n    \n    # Check for problematic patterns\n    print(f\"\\n⚠️  POTENTIAL ISSUES:\")\n    \n    # Too many unique score numbers\n    score_patterns = [text for text, _ in categories['scores']]\n    if len(score_patterns) > 50:\n        print(f\"   📊 Too many unique scores: {len(score_patterns)} (should limit to ~30)\")\n    \n    # Too many unique time patterns\n    time_patterns = [text for text, _ in categories['times']]\n    if len(time_patterns) > 30:\n        print(f\"   ⏰ Too many unique times: {len(time_patterns)} (should limit to ~20)\")\n    \n    # Check for very similar patterns\n    similar_groups = find_similar_patterns(text_counter.keys())\n    if similar_groups:\n        print(f\"   🔄 Similar patterns found: {len(similar_groups)} groups\")\n        for group in similar_groups[:3]:  # Show first 3 groups\n            print(f\"      {group}\")\n    \n    # Recommendations\n    print(f\"\\n💡 RECOMMENDATIONS:\")\n    \n    total_patterns = len(text_counter)\n    if total_patterns > 100:\n        print(f\"   🎯 LIMIT PATTERNS: {total_patterns} is too many (target: 50-80)\")\n        print(f\"   📊 Focus on most common game situations\")\n        print(f\"   🔢 Limit scores to 0-50 range\")\n        print(f\"   ⏰ Limit times to common game situations\")\n    elif total_patterns > 80:\n        print(f\"   ⚠️  BORDERLINE: {total_patterns} patterns (consider limiting to 60-70)\")\n    else:\n        print(f\"   ✅ GOOD: {total_patterns} patterns is manageable\")\n    \n    return text_counter, categories\n\ndef find_similar_patterns(patterns):\n    \"\"\"Find groups of very similar patterns.\"\"\"\n    \n    similar_groups = []\n    processed = set()\n    \n    for pattern1 in patterns:\n        if pattern1 in processed:\n            continue\n        \n        group = [pattern1]\n        for pattern2 in patterns:\n            if pattern2 != pattern1 and pattern2 not in processed:\n                # Check if patterns are very similar\n                if are_similar(pattern1, pattern2):\n                    group.append(pattern2)\n        \n        if len(group) > 1:\n            similar_groups.append(group)\n            processed.update(group)\n    \n    return similar_groups\n\ndef are_similar(text1, text2):\n    \"\"\"Check if two text patterns are very similar.\"\"\"\n    \n    # Remove case and whitespace differences\n    clean1 = text1.upper().replace(' ', '')\n    clean2 = text2.upper().replace(' ', '')\n    \n    # Check for minor differences\n    if len(clean1) == len(clean2):\n        diff_count = sum(c1 != c2 for c1, c2 in zip(clean1, clean2))\n        return diff_count <= 1  # At most 1 character difference\n    \n    return False\n\ndef create_limited_dataset():\n    \"\"\"Create a more limited dataset focusing on core patterns.\"\"\"\n    \n    print(f\"\\n🔧 Creating Limited Dataset\")\n    print(\"=\" * 60)\n    \n    with open('madden_ocr_training_data_FIXED.json', 'r') as f:\n        data = json.load(f)\n    \n    # Define core patterns we want to keep\n    core_patterns = {\n        # Down & Distance (most important)\n        '1ST & 10', '1ST & Goal', '2ND & 10', '2ND & Goal', \n        '3RD & 2', '3RD & 3', '3RD & 5', '3RD & 7', '3RD & 10', '3RD & Goal',\n        '4TH & 1', '4TH & 2', '4TH & 3', '4TH & 5', '4TH & Goal',\n        \n        # Common scores (0-35)\n        '0', '3', '6', '7', '10', '13', '14', '17', '20', '21', '24', '27', '28', '30', '31', '34', '35',\n        \n        # Special situations\n        'KICKOFF', 'PAT', '2-PT', 'GOAL', '--',\n        \n        # Common game times\n        '15:00', '12:00', '10:00', '5:00', '2:00', '1:00', '0:30', '0:15',\n        '4th 15:00', '4th 2:00', '4th 1:00', '4th 0:30'\n    }\n    \n    print(f\"📊 Core patterns defined: {len(core_patterns)}\")\n    \n    # Filter data to only include core patterns\n    limited_data = []\n    pattern_counts = {}\n    \n    for sample in data:\n        text = sample['ground_truth_text']\n        \n        # Check if this pattern should be included\n        if text in core_patterns:\n            current_count = pattern_counts.get(text, 0)\n            if current_count < 50:  # Limit to 50 samples per pattern\n                limited_data.append(sample)\n                pattern_counts[text] = current_count + 1\n    \n    print(f\"📊 Limited samples: {len(limited_data):,}\")\n    print(f\"📊 Patterns included: {len(pattern_counts)}\")\n    \n    # Show final distribution\n    print(f\"\\n📊 Final distribution:\")\n    for text, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):\n        print(f\"   '{text}': {count}\")\n    \n    # Save limited dataset\n    with open('madden_ocr_training_data_LIMITED.json', 'w') as f:\n        json.dump(limited_data, f, indent=2)\n    \n    print(f\"\\n💾 Saved limited dataset: madden_ocr_training_data_LIMITED.json\")\n    print(f\"✅ Ready for focused training on core patterns!\")\n    \n    return limited_data\n\ndef main():\n    \"\"\"Main analysis function.\"\"\"\n    \n    print(\"🔍 Fixed Training Data Analysis\")\n    print(\"=\" * 60)\n    \n    # Analyze current fixed data\n    text_counter, categories = analyze_fixed_data()\n    \n    # Ask if we should create limited dataset\n    total_patterns = len(text_counter)\n    if total_patterns > 80:\n        print(f\"\\n🎯 RECOMMENDATION: Create limited dataset\")\n        print(f\"   Current: {total_patterns} patterns\")\n        print(f\"   Target: ~40 core patterns\")\n        print(f\"   Benefits: Faster training, better accuracy, less overfitting\")\n        \n        create_limited_dataset()\n    else:\n        print(f\"\\n✅ Current dataset size is acceptable\")\n\nif __name__ == \"__main__\":\n    main()

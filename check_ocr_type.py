#!/usr/bin/env python3
"""
Check OCR Type - Analyze Madden text patterns to determine optimal OCR architecture
"""

from ultimate_madden_ocr_system import MaddenOCRDatabase


def analyze_text_patterns():
    print("🔍 Analyzing Madden OCR Text Patterns")
    print("=" * 50)

    db = MaddenOCRDatabase()

    # Get sample of validated data
    samples = db.get_all_samples(100)
    validated_samples = [s for s in samples if s["ground_truth"]]

    if not validated_samples:
        print("❌ No validated samples found!")
        return

    print(f"📊 Analyzing {len(validated_samples)} validated samples...")

    # Analyze text characteristics
    text_lengths = []
    character_counts = {}
    word_patterns = []

    for sample in validated_samples:
        text = sample["ground_truth"].strip()
        if not text:
            continue

        text_lengths.append(len(text))
        word_patterns.append(text)

        # Count characters
        for char in text:
            character_counts[char] = character_counts.get(char, 0) + 1

    # Analysis results
    print(f"\n📏 Text Length Analysis:")
    print(f"  Average length: {sum(text_lengths)/len(text_lengths):.1f} characters")
    print(f"  Max length: {max(text_lengths)} characters")
    print(f"  Min length: {min(text_lengths)} characters")

    print(f"\n🔤 Character Distribution:")
    sorted_chars = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:15]:
        print(f"  '{char}': {count} times")

    print(f"\n📝 Sample Text Patterns:")
    unique_patterns = list(set(word_patterns))[:20]
    for pattern in unique_patterns:
        print(f"  '{pattern}'")

    # Determine OCR type
    print(f"\n🎯 OCR Architecture Recommendation:")

    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)

    if max_length <= 15 and avg_length <= 8:
        print("✅ SIMPLE CNN CLASSIFIER (Recommended)")
        print("   - Short, fixed-pattern text")
        print("   - High accuracy potential")
        print("   - Fast training & inference")
        print("   - Perfect for HUD elements")
    elif max_length <= 30:
        print("✅ CNN + CTC (Character Recognition)")
        print("   - Variable length text")
        print("   - Good for mixed patterns")
        print("   - Moderate complexity")
    else:
        print("✅ CNN + LSTM + CTC (Full Sequence)")
        print("   - Long text sequences")
        print("   - Complex patterns")
        print("   - Higher complexity")

    print(f"\n🚀 Transfer Learning Strategy:")
    print("   - Use pre-trained CNN backbone (ResNet/EfficientNet)")
    print("   - Replace classifier head for Madden character set")
    print("   - Fine-tune on your validated samples")
    print("   - Expected accuracy: 95-98%")


if __name__ == "__main__":
    analyze_text_patterns()

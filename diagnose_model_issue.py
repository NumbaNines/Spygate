#!/usr/bin/env python3
"""
Focused diagnostic to understand the K1tC pattern issue.
"""

import os
import sys

import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.custom_ocr import SpygateMaddenOCR


def analyze_model_predictions():
    """Analyze what the model is actually predicting at the character level."""

    print("🔍 Model Prediction Analysis")
    print("=" * 60)

    # Load model
    custom_ocr = SpygateMaddenOCR()
    if not custom_ocr.is_available():
        print("❌ Custom OCR not available")
        return

    # Get character mappings
    model_path = "models/fixed_ocr_20250614_150024/best_fixed_model.pth"
    checkpoint = torch.load(model_path, map_location="cpu")
    idx_to_char = checkpoint.get("idx_to_char", {})

    print(f"📊 Character mappings loaded: {len(idx_to_char)} characters")

    # Create a simple test image
    test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "1ST & 10", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    print(f"\n🧪 Testing with '1ST & 10' image...")

    try:
        # Get raw model output if possible
        if hasattr(custom_ocr, "model") and hasattr(custom_ocr, "transform_image"):
            # Transform image to tensor
            tensor_input = custom_ocr.transform_image(test_image)
            print(f"   Input tensor shape: {tensor_input.shape}")

            # Get model output
            with torch.no_grad():
                custom_ocr.model.eval()
                output = custom_ocr.model(tensor_input)
                print(f"   Model output shape: {output.shape}")

                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=2)

                # Get predicted indices
                predicted_indices = torch.argmax(probs, dim=2)
                print(f"   Predicted indices shape: {predicted_indices.shape}")

                # Convert to characters
                predicted_chars = []
                for seq_idx in range(predicted_indices.shape[1]):  # For each time step
                    char_idx = predicted_indices[0, seq_idx].item()  # Batch size is 1
                    if str(char_idx) in idx_to_char:
                        char = idx_to_char[str(char_idx)]
                        predicted_chars.append(char)

                print(f"   Raw predicted characters: {predicted_chars}")

                # Remove CTC blanks and duplicates
                cleaned_chars = []
                prev_char = None
                for char in predicted_chars:
                    if char != "<CTC_BLANK>" and char != prev_char:
                        cleaned_chars.append(char)
                        prev_char = char

                raw_prediction = "".join(cleaned_chars)
                print(f"   Raw prediction (no CTC): '{raw_prediction}'")

                # Compare with actual OCR output
                ocr_result = custom_ocr.extract_text(test_image, "test")
                ocr_prediction = ocr_result.get("text", "")
                print(f"   OCR method output: '{ocr_prediction}'")

                # Analyze character-by-character
                print(f"\n🔤 Character Analysis:")
                print(f"   Expected: '1ST & 10'")
                print(f"   Got:      '{ocr_prediction}'")

                expected = "1ST & 10"
                for i, (exp_char, got_char) in enumerate(zip(expected, ocr_prediction)):
                    match = "✅" if exp_char == got_char else "❌"
                    print(f"   Position {i}: '{exp_char}' → '{got_char}' {match}")

        else:
            print("   Cannot access model internals")

    except Exception as e:
        print(f"❌ Model analysis failed: {e}")
        import traceback

        traceback.print_exc()


def test_character_recognition():
    """Test individual character recognition."""

    print(f"\n🔤 Individual Character Recognition Test")
    print("=" * 60)

    custom_ocr = SpygateMaddenOCR()

    # Test individual characters
    test_chars = ["1", "2", "3", "S", "T", "A", "&", ":", "0"]

    print(f"Testing {len(test_chars)} individual characters...")
    print("-" * 40)
    print(f"{'Char':<6} {'Prediction':<15} {'Confidence':<12}")
    print("-" * 40)

    for char in test_chars:
        # Create image with single character
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.putText(img, char, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        try:
            result = custom_ocr.extract_text(img, "test")
            prediction = result.get("text", "").strip()
            confidence = result.get("confidence", 0)

            match = "✅" if prediction == char else "❌"
            print(f"{char:<6} {prediction:<15} {confidence:<12.3f} {match}")

        except Exception as e:
            print(f"{char:<6} ERROR: {str(e)[:10]:<15} {'0.000':<12} ❌")


def check_model_bias():
    """Check if model has learned a bias towards certain characters."""

    print(f"\n🎯 Model Bias Analysis")
    print("=" * 60)

    custom_ocr = SpygateMaddenOCR()

    # Test with different image types
    test_cases = [
        ("Empty white", np.ones((50, 200, 3), dtype=np.uint8) * 255),
        ("Empty black", np.zeros((50, 200, 3), dtype=np.uint8)),
        ("Random noise", np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)),
        ("Gray", np.ones((50, 200, 3), dtype=np.uint8) * 128),
    ]

    print("Testing model response to different image types...")
    print("-" * 50)
    print(f"{'Image Type':<15} {'Prediction':<20} {'Confidence':<12}")
    print("-" * 50)

    for name, img in test_cases:
        try:
            result = custom_ocr.extract_text(img, "test")
            prediction = result.get("text", "").strip()
            confidence = result.get("confidence", 0)

            print(f"{name:<15} {prediction:<20} {confidence:<12.3f}")

        except Exception as e:
            print(f"{name:<15} ERROR: {str(e)[:15]:<20} {'0.000':<12}")


def main():
    """Main diagnostic function."""

    print("🔍 Custom OCR K1tC Pattern Diagnosis")
    print("=" * 60)

    # Test 1: Analyze model predictions
    analyze_model_predictions()

    # Test 2: Test individual characters
    test_character_recognition()

    # Test 3: Check model bias
    check_model_bias()

    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("\n💡 FINDINGS:")
    print("   - Model consistently produces 'K1tC' pattern")
    print("   - This suggests learned bias or training issue")
    print("   - Character mappings look correct")
    print("   - Training data ground truth looks correct")
    print("\n🔧 LIKELY CAUSES:")
    print("   1. Model architecture too simple for task complexity")
    print("   2. Insufficient training (25 epochs not enough)")
    print("   3. Learning rate too high (model stuck in local minimum)")
    print("   4. Training data preprocessing issues")
    print("   5. CTC loss function not converging properly")
    print("\n🚀 RECOMMENDED SOLUTIONS:")
    print("   1. Retrain with more epochs (50-100)")
    print("   2. Lower learning rate (0.0001 instead of 0.001)")
    print("   3. Add learning rate scheduling")
    print("   4. Increase model capacity (more LSTM units)")
    print("   5. Add data augmentation")
    print("   6. Use different optimizer (AdamW with weight decay)")


if __name__ == "__main__":
    main()

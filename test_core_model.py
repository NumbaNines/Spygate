#!/usr/bin/env python3
"""
Test the trained core OCR model on sample images.
"""

import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SimpleOCRModel(nn.Module):
    def __init__(self, vocab_size, max_length=15):
        super(SimpleOCRModel, self).__init__()
        self.max_length = max_length

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Classifier for each character position
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, vocab_size * max_length)
        )

        self.vocab_size = vocab_size

    def forward(self, x):
        # CNN features
        features = self.cnn(x)  # [B, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 256]

        # Classify each position
        output = self.classifier(features)  # [B, vocab_size * max_length]
        output = output.view(-1, self.max_length, self.vocab_size)  # [B, max_length, vocab_size]

        return output


def preprocess_image(image_path):
    """Enhanced preprocessing for dark HUD regions."""
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Resize to standard size
        img = cv2.resize(img, (128, 32))

        # Enhanced preprocessing for dark text
        # 1. Brightness boost
        img = cv2.convertScaleAbs(img, alpha=2.5, beta=40)

        # 2. CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

        # 3. Gamma correction
        gamma = 1.2
        img = np.power(img / 255.0, 1.0 / gamma) * 255.0
        img = img.astype(np.uint8)

        # Normalize
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def predict_text(model, image_path, char_to_idx, idx_to_char, device):
    """Predict text from image using the trained model."""

    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return "ERROR"

    # Convert to tensor
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 32, 128]

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)  # [1, max_length, vocab_size]
        pred = outputs.argmax(dim=2).squeeze(0)  # [max_length]

    # Convert to text
    predicted_text = ""
    for idx in pred:
        char = idx_to_char[idx.item()]
        if char == "<PAD>":
            break
        if char != "<UNK>":
            predicted_text += char

    return predicted_text


def test_core_model():
    print("🧪 Testing Core OCR Model")
    print("=" * 50)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        "madden_simple_core_ocr_model.pth", map_location=device, weights_only=False
    )

    # Create model
    model = SimpleOCRModel(checkpoint["vocab_size"], checkpoint["max_length"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]

    print(f"✅ Model loaded successfully")
    print(f"📊 Vocab size: {checkpoint['vocab_size']}")
    print(f"📊 Max length: {checkpoint['max_length']}")

    # Load test data
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    # Test on random samples
    print(f"\n🎯 Testing on random samples:")
    print("-" * 50)

    correct = 0
    total = 0

    # Test 20 random samples
    test_samples = random.sample(data, min(20, len(data)))

    for i, sample in enumerate(test_samples):
        ground_truth = sample["ground_truth_text"]
        image_path = sample["image_path"]

        predicted = predict_text(model, image_path, char_to_idx, idx_to_char, device)

        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1
        total += 1

        status = "✅" if is_correct else "❌"
        print(f"{i+1:2d}. {status} GT: '{ground_truth}' | Pred: '{predicted}'")

        if not is_correct:
            print(f"     Image: {image_path}")

    accuracy = correct / total
    print(f"\n📊 RESULTS:")
    print(f"   Accuracy: {accuracy:.1%} ({correct}/{total})")

    if accuracy > 0.8:
        print(f"   🎉 EXCELLENT! Model is working well")
    elif accuracy > 0.6:
        print(f"   ✅ GOOD! Model shows improvement")
    elif accuracy > 0.4:
        print(f"   ⚠️  FAIR! Better than random but needs work")
    else:
        print(f"   ❌ POOR! Model needs more training")

    # Test specific patterns
    print(f"\n🎯 Testing specific pattern types:")
    print("-" * 50)

    pattern_tests = {}
    for sample in test_samples:
        gt = sample["ground_truth_text"]
        pred = predict_text(model, sample["image_path"], char_to_idx, idx_to_char, device)

        # Categorize
        if "&" in gt:
            category = "Down & Distance"
        elif gt.isdigit():
            category = "Scores"
        elif ":" in gt:
            category = "Times"
        else:
            category = "Special"

        if category not in pattern_tests:
            pattern_tests[category] = {"correct": 0, "total": 0}

        pattern_tests[category]["total"] += 1
        if pred == gt:
            pattern_tests[category]["correct"] += 1

    for category, stats in pattern_tests.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"   {category}: {acc:.1%} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    test_core_model()

#!/usr/bin/env python3
"""
FINAL KerasOCR vs PaddleOCR Comparison
"""

import json
import random
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def test_keras():
    """Test KerasOCR."""
    try:
        import keras_ocr

        pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ KerasOCR working")
        return pipeline
    except Exception as e:
        print(f"❌ KerasOCR failed: {e}")
        return None


def test_paddle():
    """Test PaddleOCR with correct API."""
    try:
        from paddleocr import PaddleOCR

        # Use minimal initialization
        ocr = PaddleOCR(lang="en")
        print("✅ PaddleOCR working")
        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return None


def preprocess_enhanced(image_path):
    """Enhanced preprocessing for both engines."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # Aggressive resize for small HUD text
    h, w = img.shape[:2]
    scale = max(128 / h, 256 / w, 4.0)  # Large scale
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aggressive enhancement for dark HUD text
    # 1. Strong brightness boost
    gray = cv2.convertScaleAbs(gray, alpha=3.0, beta=50)

    # 2. CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 3. Noise reduction
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    # 4. Gamma correction
    gamma = 0.7
    gray = np.power(gray / 255.0, gamma) * 255.0
    gray = gray.astype(np.uint8)

    # Return both formats
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # For PaddleOCR
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # For KerasOCR

    return img_color, img_rgb


def extract_keras(pipeline, image_path):
    """Extract with KerasOCR."""
    if pipeline is None:
        return ""

    try:
        img_color, img_rgb = preprocess_enhanced(image_path)
        if img_rgb is None:
            return ""

        results = pipeline.recognize([img_rgb])
        if results and results[0]:
            # Get all detected text
            texts = []
            for text, box in results[0]:
                if text and len(text.strip()) > 0:
                    texts.append(text.strip())

            if texts:
                # Return the longest text (usually most complete)
                return max(texts, key=len)

        return ""
    except Exception as e:
        return ""


def extract_paddle(ocr, image_path):
    """Extract with PaddleOCR."""
    if ocr is None:
        return ""

    try:
        img_color, img_rgb = preprocess_enhanced(image_path)
        if img_color is None:
            return ""

        results = ocr.ocr(img_color)
        if results and results[0]:
            # Get all detected text with confidence
            texts = []
            for line in results[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    if text and confidence > 0.1:
                        texts.append((text, confidence))

            if texts:
                # Return text with highest confidence
                return max(texts, key=lambda x: x[1])[0]

        return ""
    except Exception as e:
        return ""


def main():
    """Run the final comparison."""
    print("🎯 FINAL: KerasOCR vs PaddleOCR Comparison")
    print("=" * 45)

    # Test engines
    print("\n🚀 Initializing engines...")
    keras_pipeline = test_keras()
    paddle_ocr = test_paddle()

    if keras_pipeline is None and paddle_ocr is None:
        print("❌ Both engines failed!")
        return

    # Load test data
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    # Use same test samples as custom OCR
    random.seed(42)
    test_samples = random.sample(data, 15)

    print(f"\n📊 Testing same 15 samples as failed custom OCR...")
    print("=" * 45)

    # Track results
    keras_correct = 0
    paddle_correct = 0
    keras_total = 0
    paddle_total = 0

    # Test each sample
    for i, sample in enumerate(test_samples):
        gt = sample["ground_truth_text"]
        path = sample["image_path"]

        print(f"\n{i+1:2d}. GT: '{gt}'")

        # Test KerasOCR
        if keras_pipeline:
            keras_pred = extract_keras(keras_pipeline, path)
            keras_ok = keras_pred == gt
            keras_total += 1
            if keras_ok:
                keras_correct += 1
            status = "✅" if keras_ok else "❌"
            print(f"    KerasOCR : {status} '{keras_pred}'")

        # Test PaddleOCR
        if paddle_ocr:
            paddle_pred = extract_paddle(paddle_ocr, path)
            paddle_ok = paddle_pred == gt
            paddle_total += 1
            if paddle_ok:
                paddle_correct += 1
            status = "✅" if paddle_ok else "❌"
            print(f"    PaddleOCR: {status} '{paddle_pred}'")

    # Calculate accuracies
    keras_acc = keras_correct / keras_total if keras_total > 0 else 0
    paddle_acc = paddle_correct / paddle_total if paddle_total > 0 else 0

    # Print final results
    print(f"\n🏆 FINAL COMPARISON RESULTS")
    print("=" * 45)
    print(f"📊 CUSTOM OCR (Baseline): 0.0% (0/20) - Complete failure")

    if keras_total > 0:
        print(f"📊 KERAS OCR:             {keras_acc:.1%} ({keras_correct}/{keras_total})")
    else:
        print(f"📊 KERAS OCR:             FAILED")

    if paddle_total > 0:
        print(f"📊 PADDLE OCR:            {paddle_acc:.1%} ({paddle_correct}/{paddle_total})")
    else:
        print(f"📊 PADDLE OCR:            FAILED")

    # Determine winner
    print(f"\n🥇 WINNER:")
    if keras_total > 0 and paddle_total > 0:
        if keras_acc > paddle_acc:
            improvement = keras_acc - paddle_acc
            print(f"   🎉 KERAS OCR WINS!")
            print(f"   📈 {keras_acc:.1%} vs {paddle_acc:.1%} (+{improvement:.1%})")
            print(f"   🚀 INTEGRATE KERAS OCR into SpygateAI")
        elif paddle_acc > keras_acc:
            improvement = paddle_acc - keras_acc
            print(f"   🎉 PADDLE OCR WINS!")
            print(f"   📈 {paddle_acc:.1%} vs {keras_acc:.1%} (+{improvement:.1%})")
            print(f"   🚀 INTEGRATE PADDLE OCR into SpygateAI")
        else:
            print(f"   🤝 TIE! Both achieved {keras_acc:.1%}")
            print(f"   🎯 Choose based on performance/dependencies")
    elif keras_total > 0:
        print(f"   🥇 KERAS OCR wins (PaddleOCR failed)")
        print(f"   🚀 INTEGRATE KERAS OCR into SpygateAI")
    elif paddle_total > 0:
        print(f"   🥇 PADDLE OCR wins (KerasOCR failed)")
        print(f"   🚀 INTEGRATE PADDLE OCR into SpygateAI")
    else:
        print(f"   ❌ NO WINNER - Both failed")

    # Performance analysis
    best_acc = max(keras_acc, paddle_acc)
    print(f"\n💡 EXPERT ANALYSIS:")
    if best_acc >= 0.7:
        print(f"   🎉 EXCELLENT! {best_acc:.1%} is production-ready")
        print(f"   🚀 Ready for immediate integration")
    elif best_acc >= 0.5:
        print(f"   ✅ GOOD! {best_acc:.1%} shows strong potential")
        print(f"   🎯 Consider fine-tuning for optimization")
    elif best_acc >= 0.3:
        print(f"   ⚠️  FAIR! {best_acc:.1%} is better than custom")
        print(f"   🔧 Focus on preprocessing optimization")
    elif best_acc > 0:
        print(f"   ⚠️  POOR! {best_acc:.1%} indicates challenges")
        print(f"   🔍 Investigate data quality issues")
    else:
        print(f"   ❌ FAILED! Both engines got 0% accuracy")
        print(f"   🔍 Dataset may be fundamentally problematic")

    # Save results
    results = {
        "keras_accuracy": keras_acc,
        "paddle_accuracy": paddle_acc,
        "keras_correct": keras_correct,
        "paddle_correct": paddle_correct,
        "keras_total": keras_total,
        "paddle_total": paddle_total,
        "custom_baseline": 0.0,
        "winner": (
            "keras" if keras_acc > paddle_acc else "paddle" if paddle_acc > keras_acc else "tie"
        ),
    }

    with open("final_keras_paddle_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: final_keras_paddle_results.json")


if __name__ == "__main__":
    main()

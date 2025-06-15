#!/usr/bin/env python3
"""
PaddleOCR Fixed Baseline Test - Expert OCR Solution
Testing PaddleOCR with updated API on our test samples.
"""

import json
import random

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PaddleOCRFixed:
    def __init__(self):
        print("🚀 Initializing PaddleOCR (Fixed API)...")

        # Initialize PaddleOCR with updated API
        self.paddle_ocr = PaddleOCR(
            lang="en",  # English language
            show_log=False,  # Suppress logs
            use_space_char=True,  # Enable space character recognition
            drop_score=0.3,  # Lower threshold for better recall
        )
        print("✅ PaddleOCR initialized successfully")

    def preprocess_image_for_ocr(self, image_path):
        """Optimized preprocessing for PaddleOCR."""
        try:
            # Load as color image (PaddleOCR prefers color)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # Get original dimensions
            h, w = img.shape[:2]

            # Ensure minimum size for OCR
            min_height = 64
            min_width = 128

            if h < min_height or w < min_width:
                scale = max(min_height / h, min_width / w, 2.0)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale for processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing for dark HUD text
            # 1. Strong brightness and contrast boost
            gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=40)

            # 2. CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

            # 3. Bilateral filter for noise reduction
            gray = cv2.bilateralFilter(gray, 5, 50, 50)

            # 4. Gamma correction
            gamma = 0.8
            gray = np.power(gray / 255.0, gamma) * 255.0
            gray = gray.astype(np.uint8)

            # Convert back to BGR for PaddleOCR
            img_processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            return img_processed

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

    def extract_text(self, image_path):
        """Extract text using PaddleOCR."""
        img = self.preprocess_image_for_ocr(image_path)
        if img is None:
            return ""

        try:
            # Run OCR
            results = self.paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                # Collect all detected text with confidence scores
                detected_texts = []
                for line in results[0]:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        text = line[1][0].strip()
                        confidence = line[1][1]
                        if text and confidence > 0.1:  # Filter very low confidence
                            detected_texts.append((text, confidence))

                if detected_texts:
                    # Return text with highest confidence
                    best_text = max(detected_texts, key=lambda x: x[1])[0]
                    return best_text

            return ""

        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return ""

    def get_test_samples(self):
        """Get the same test samples as our custom OCR test."""
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)

        # Use same random seed for consistency
        random.seed(42)
        test_samples = random.sample(data, 20)

        return test_samples

    def run_test(self):
        """Run PaddleOCR test on our samples."""
        print("🎯 PaddleOCR Test - Same Samples as Failed Custom Model")
        print("=" * 65)

        test_samples = self.get_test_samples()

        print(f"📊 Testing on same 20 samples")
        print(f"🔄 Using optimized preprocessing for dark HUD text")

        # Track results
        correct = 0
        total = 0
        predictions = []

        # Test each sample
        for i, sample in enumerate(test_samples):
            ground_truth = sample["ground_truth_text"]
            image_path = sample["image_path"]

            prediction = self.extract_text(image_path)
            is_correct = prediction == ground_truth

            if is_correct:
                correct += 1
            total += 1

            predictions.append(
                {
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": is_correct,
                    "image_path": image_path,
                }
            )

            status = "✅" if is_correct else "❌"
            print(f"{i+1:2d}. {status} GT: '{ground_truth}' | PaddleOCR: '{prediction}'")

        accuracy = correct / total

        # Print results
        print(f"\n🏆 PADDLEOCR RESULTS")
        print("=" * 65)
        print(f"📊 CUSTOM OCR (Previous): 0.0% (0/20) - Complete failure")
        print(f"📊 PADDLEOCR (Current):   {accuracy:.1%} ({correct}/{total})")
        print(f"📈 Improvement: +{accuracy:.1%}")

        # Analyze by pattern type
        pattern_analysis = {
            "down_distance": {"correct": 0, "total": 0},
            "scores": {"correct": 0, "total": 0},
            "times": {"correct": 0, "total": 0},
            "special": {"correct": 0, "total": 0},
        }

        for pred in predictions:
            gt = pred["ground_truth"]
            correct = pred["correct"]

            if "&" in gt:
                category = "down_distance"
            elif gt.isdigit():
                category = "scores"
            elif ":" in gt:
                category = "times"
            else:
                category = "special"

            pattern_analysis[category]["total"] += 1
            if correct:
                pattern_analysis[category]["correct"] += 1

        print(f"\n📊 Performance by Pattern Type:")
        for category, stats in pattern_analysis.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                print(
                    f"   {category.replace('_', ' ').title()}: {acc:.1%} ({stats['correct']}/{stats['total']})"
                )

        # Expert recommendation
        print(f"\n💡 EXPERT ANALYSIS:")
        if accuracy >= 0.8:
            print(f"   🎉 EXCELLENT! PaddleOCR is production-ready")
            print(f"   🚀 Integrate immediately into SpygateAI")
            print(f"   🎯 Consider fine-tuning for even better results")
        elif accuracy >= 0.6:
            print(f"   ✅ VERY GOOD! Strong baseline performance")
            print(f"   🔧 Fine-tune on Madden dataset for optimization")
            print(f"   📊 Ready for production with monitoring")
        elif accuracy >= 0.4:
            print(f"   ⚠️  GOOD! Significant improvement over custom")
            print(f"   🎯 Focus on preprocessing optimization")
            print(f"   🔧 Consider domain-specific fine-tuning")
        elif accuracy >= 0.2:
            print(f"   ⚠️  FAIR! Better than custom but needs work")
            print(f"   🔍 Analyze failed cases for patterns")
            print(f"   🎯 Optimize preprocessing pipeline")
        else:
            print(f"   ❌ POOR! Dataset may be fundamentally challenging")
            print(f"   🔍 Need to examine data quality issues")
            print(f"   🎯 Consider alternative approaches")

        # Save results
        results_data = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
            "pattern_analysis": pattern_analysis,
        }

        with open("paddle_ocr_test_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n💾 Results saved to: paddle_ocr_test_results.json")

        return accuracy, predictions


def main():
    """Run the PaddleOCR test."""
    tester = PaddleOCRFixed()
    accuracy, predictions = tester.run_test()

    print(f"\n🎯 NEXT STEPS:")
    if accuracy > 0.6:
        print(f"   1. ✅ Replace custom OCR with PaddleOCR in SpygateAI")
        print(f"   2. 🎯 Fine-tune PaddleOCR on full Madden dataset")
        print(f"   3. 🚀 Deploy to production")
    elif accuracy > 0.3:
        print(f"   1. 🔧 Optimize PaddleOCR preprocessing")
        print(f"   2. 📊 Collect more training data")
        print(f"   3. 🎯 Consider ensemble methods")
    else:
        print(f"   1. 🔍 Investigate data quality issues")
        print(f"   2. 🎯 Try alternative OCR approaches")
        print(f"   3. 📊 Consider different preprocessing strategies")


if __name__ == "__main__":
    main()

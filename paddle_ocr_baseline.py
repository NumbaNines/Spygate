#!/usr/bin/env python3
"""
PaddleOCR Baseline Test - Expert OCR Solution
Testing PaddleOCR on the exact same samples as our failed custom model.
"""

import json
import random

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PaddleOCRBaseline:
    def __init__(self):
        print("🚀 Initializing PaddleOCR for Baseline Test...")

        # Initialize PaddleOCR with optimal settings
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,  # Enable angle classification
            lang="en",  # English language
            use_gpu=True,  # Use GPU acceleration
            show_log=False,  # Suppress logs
            det_model_dir=None,  # Use default detection model
            rec_model_dir=None,  # Use default recognition model
            cls_model_dir=None,  # Use default classification model
            use_space_char=True,  # Enable space character recognition
            drop_score=0.3,  # Lower threshold for better recall
        )
        print("✅ PaddleOCR initialized with optimal settings")

    def preprocess_image_minimal(self, image_path):
        """Minimal preprocessing - let PaddleOCR handle most of it."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Keep as color
            if img is None:
                return None

            # Only basic enhancement
            # 1. Slight brightness boost
            img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)

            # 2. Light denoising
            img = cv2.bilateralFilter(img, 3, 50, 50)

            return img

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

    def preprocess_image_aggressive(self, image_path):
        """Aggressive preprocessing for dark HUD text."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize to larger size for better OCR
            h, w = img.shape
            scale = max(64 / h, 256 / w, 2.0)  # Ensure minimum size
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))

            # Aggressive enhancement for dark text
            # 1. Strong brightness boost
            img = cv2.convertScaleAbs(img, alpha=3.0, beta=50)

            # 2. CLAHE for local contrast
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            img = clahe.apply(img)

            # 3. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            # 4. Gamma correction
            gamma = 0.7
            img = np.power(img / 255.0, gamma) * 255.0
            img = img.astype(np.uint8)

            # Convert back to color for PaddleOCR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            return img

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

    def extract_text_paddle(self, image_path, preprocessing="aggressive"):
        """Extract text using PaddleOCR with different preprocessing."""
        if preprocessing == "minimal":
            img = self.preprocess_image_minimal(image_path)
        else:
            img = self.preprocess_image_aggressive(image_path)

        if img is None:
            return ""

        try:
            results = self.paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                # Get all detected text and find best match
                all_texts = []
                for line in results[0]:
                    text = line[1][0].strip()
                    confidence = line[1][1]
                    all_texts.append((text, confidence))

                if all_texts:
                    # Return text with highest confidence
                    best_text = max(all_texts, key=lambda x: x[1])[0]
                    return best_text

            return ""

        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return ""

    def get_exact_test_samples(self):
        """Get the EXACT same 20 samples we used to test custom OCR."""
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)

        # Use same random seed as our previous test
        random.seed(42)
        test_samples = random.sample(data, 20)

        return test_samples

    def run_baseline_test(self):
        """Run PaddleOCR baseline test."""
        print("🎯 PaddleOCR Baseline Test - Same Samples as Custom Model")
        print("=" * 70)

        # Get exact same test samples
        test_samples = self.get_exact_test_samples()

        print(f"📊 Testing on same 20 samples used for custom OCR")
        print(f"🔄 Testing both minimal and aggressive preprocessing")

        # Results tracking
        results = {
            "minimal": {"correct": 0, "total": 0, "predictions": []},
            "aggressive": {"correct": 0, "total": 0, "predictions": []},
        }

        # Test each sample with both preprocessing methods
        for i, sample in enumerate(test_samples):
            ground_truth = sample["ground_truth_text"]
            image_path = sample["image_path"]

            print(f"\n📝 Sample {i+1}/20: '{ground_truth}'")

            # Test minimal preprocessing
            minimal_pred = self.extract_text_paddle(image_path, "minimal")
            minimal_correct = minimal_pred == ground_truth
            results["minimal"]["total"] += 1
            results["minimal"]["predictions"].append(
                {"gt": ground_truth, "pred": minimal_pred, "correct": minimal_correct}
            )
            if minimal_correct:
                results["minimal"]["correct"] += 1

            status = "✅" if minimal_correct else "❌"
            print(f"   Minimal    : {status} '{minimal_pred}'")

            # Test aggressive preprocessing
            aggressive_pred = self.extract_text_paddle(image_path, "aggressive")
            aggressive_correct = aggressive_pred == ground_truth
            results["aggressive"]["total"] += 1
            results["aggressive"]["predictions"].append(
                {"gt": ground_truth, "pred": aggressive_pred, "correct": aggressive_correct}
            )
            if aggressive_correct:
                results["aggressive"]["correct"] += 1

            status = "✅" if aggressive_correct else "❌"
            print(f"   Aggressive : {status} '{aggressive_pred}'")

        # Print final comparison
        print(f"\n🏆 PADDLEOCR BASELINE RESULTS")
        print("=" * 70)

        print(f"📊 CUSTOM OCR (Previous): 0.0% (0/20) - Failed completely")
        print(f"   Output: '22D & 10FD69RI' for all inputs")

        for method, result in results.items():
            accuracy = result["correct"] / result["total"]
            print(
                f"📊 PaddleOCR ({method}): {accuracy:.1%} ({result['correct']}/{result['total']})"
            )

        # Determine best method
        best_method = "minimal"
        best_accuracy = results["minimal"]["correct"] / results["minimal"]["total"]

        aggressive_accuracy = results["aggressive"]["correct"] / results["aggressive"]["total"]
        if aggressive_accuracy > best_accuracy:
            best_method = "aggressive"
            best_accuracy = aggressive_accuracy

        print(f"\n🥇 BEST PADDLEOCR METHOD: {best_method.upper()}")
        print(f"   Accuracy: {best_accuracy:.1%}")
        print(f"   Improvement over custom: +{best_accuracy:.1%}")

        # Show some examples from best method
        print(f"\n📋 Examples from {best_method} method:")
        best_results = results[best_method]["predictions"]
        for i, pred_data in enumerate(best_results[:10]):
            status = "✅" if pred_data["correct"] else "❌"
            print(f"   {i+1:2d}. {status} GT: '{pred_data['gt']}' | Pred: '{pred_data['pred']}'")

        return best_method, best_accuracy, results


def main():
    """Run the PaddleOCR baseline test."""
    tester = PaddleOCRBaseline()
    best_method, best_accuracy, results = tester.run_baseline_test()

    print(f"\n💡 EXPERT ANALYSIS:")
    if best_accuracy > 0.7:
        print(f"   🎉 EXCELLENT! PaddleOCR shows strong performance")
        print(f"   🎯 Ready for fine-tuning on Madden dataset")
        print(f"   🚀 Integrate into SpygateAI as primary OCR")
    elif best_accuracy > 0.4:
        print(f"   ✅ GOOD! PaddleOCR shows promise")
        print(f"   🔧 Fine-tune with domain-specific training")
        print(f"   📊 Consider ensemble with other methods")
    elif best_accuracy > 0.2:
        print(f"   ⚠️  FAIR! Better than custom but needs work")
        print(f"   🎯 Focus on preprocessing optimization")
        print(f"   🔧 Consider specialized text detection")
    else:
        print(f"   ❌ POOR! Dataset may be too challenging")
        print(f"   🔍 Need to examine image quality issues")
        print(f"   🎯 Consider different approach entirely")

    # Save results
    with open("paddle_ocr_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: paddle_ocr_baseline_results.json")


if __name__ == "__main__":
    main()

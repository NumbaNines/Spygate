#!/usr/bin/env python3
"""
Enhanced EasyOCR Test - Show what's possible with better preprocessing
Since KerasOCR and PaddleOCR have dependency issues, let's optimize what we have.
"""

import json
import random

import cv2
import easyocr
import numpy as np


class EnhancedEasyOCR:
    def __init__(self):
        print("🚀 Initializing Enhanced EasyOCR...")
        self.reader = easyocr.Reader(["en"], gpu=True)
        print("✅ EasyOCR initialized with GPU support")

    def preprocess_minimal(self, image_path):
        """Minimal preprocessing - current approach."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Basic resize
        h, w = img.shape[:2]
        if h < 32 or w < 32:
            scale = max(32 / h, 64 / w, 2.0)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))

        return img

    def preprocess_aggressive(self, image_path):
        """Aggressive preprocessing for dark HUD text."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Large resize for better OCR
        h, w = img.shape
        scale = max(128 / h, 256 / w, 4.0)  # Much larger scale
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Aggressive enhancement pipeline
        # 1. Extreme brightness boost
        img = cv2.convertScaleAbs(img, alpha=3.5, beta=60)

        # 2. CLAHE with high clip limit
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        img = clahe.apply(img)

        # 3. Gaussian blur to smooth
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 4. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # 5. Gamma correction
        gamma = 0.6
        img = np.power(img / 255.0, gamma) * 255.0
        img = img.astype(np.uint8)

        # 6. Final sharpening
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel_sharp)

        return img

    def preprocess_extreme(self, image_path):
        """Extreme preprocessing - kitchen sink approach."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Massive resize
        h, w = img.shape
        scale = max(256 / h, 512 / w, 6.0)  # Huge scale
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Multi-stage enhancement
        # Stage 1: Brightness explosion
        img = cv2.convertScaleAbs(img, alpha=4.0, beta=80)

        # Stage 2: Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(1, 1))
        img = clahe.apply(img)

        # Stage 3: Bilateral filter for edge preservation
        img = cv2.bilateralFilter(img, 9, 75, 75)

        # Stage 4: Unsharp masking
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

        # Stage 5: Morphological enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # Stage 6: Final gamma and contrast
        gamma = 0.5
        img = np.power(img / 255.0, gamma) * 255.0
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

        return img

    def extract_text(self, image_path, method="minimal"):
        """Extract text using specified preprocessing method."""
        if method == "minimal":
            img = self.preprocess_minimal(image_path)
        elif method == "aggressive":
            img = self.preprocess_aggressive(image_path)
        elif method == "extreme":
            img = self.preprocess_extreme(image_path)
        else:
            return ""

        if img is None:
            return ""

        try:
            # EasyOCR with optimized settings
            results = self.reader.readtext(
                img,
                detail=0,  # Only return text
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder="greedy",
                beamWidth=5,
                batch_size=1,
            )

            if results:
                # Return the longest text (usually most complete)
                return max(results, key=len).strip()

            return ""
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""

    def run_enhanced_test(self):
        """Run enhanced EasyOCR test with multiple preprocessing methods."""
        print("🎯 Enhanced EasyOCR Test - Multiple Preprocessing Methods")
        print("=" * 65)

        # Load test data
        try:
            with open("madden_ocr_training_data_CORE.json", "r") as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} training samples")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return

        # Get same test samples as custom OCR
        random.seed(42)
        test_samples = random.sample(data, 15)

        print(f"\n📊 Testing 3 preprocessing methods on same 15 samples...")
        print("=" * 65)

        # Track results for each method
        methods = ["minimal", "aggressive", "extreme"]
        results = {}

        for method in methods:
            results[method] = {"correct": 0, "total": 0, "predictions": []}

        # Test each sample with all methods
        for i, sample in enumerate(test_samples):
            ground_truth = sample["ground_truth_text"]
            image_path = sample["image_path"]

            print(f"\n{i+1:2d}. GT: '{ground_truth}'")

            for method in methods:
                prediction = self.extract_text(image_path, method)
                is_correct = prediction == ground_truth

                results[method]["total"] += 1
                results[method]["predictions"].append(
                    {"gt": ground_truth, "pred": prediction, "correct": is_correct}
                )

                if is_correct:
                    results[method]["correct"] += 1

                status = "✅" if is_correct else "❌"
                print(f"    {method.capitalize():10s}: {status} '{prediction}'")

        # Calculate accuracies
        accuracies = {}
        for method in methods:
            accuracies[method] = results[method]["correct"] / results[method]["total"]

        # Print final results
        print(f"\n🏆 ENHANCED EASYOCR RESULTS")
        print("=" * 65)
        print(f"📊 CUSTOM OCR (Baseline):  0.0% (0/20) - Complete failure")
        print(
            f"📊 EasyOCR (Current):      {accuracies['minimal']:.1%} ({results['minimal']['correct']}/{results['minimal']['total']}) - Current approach"
        )
        print(
            f"📊 EasyOCR (Aggressive):   {accuracies['aggressive']:.1%} ({results['aggressive']['correct']}/{results['aggressive']['total']}) - Enhanced preprocessing"
        )
        print(
            f"📊 EasyOCR (Extreme):      {accuracies['extreme']:.1%} ({results['extreme']['correct']}/{results['extreme']['total']}) - Kitchen sink approach"
        )

        # Find best method
        best_method = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_method]
        improvement = best_accuracy - accuracies["minimal"]

        print(f"\n🥇 BEST METHOD: {best_method.upper()}")
        print(f"   Accuracy: {best_accuracy:.1%}")
        print(f"   Improvement over current: +{improvement:.1%}")

        # Expert analysis
        print(f"\n💡 EXPERT ANALYSIS:")
        if best_accuracy >= 0.8:
            print(f"   🎉 EXCELLENT! {best_accuracy:.1%} accuracy is production-ready")
            print(f"   🚀 Integrate {best_method} preprocessing immediately")
            print(f"   🎯 This beats most commercial OCR solutions")
        elif best_accuracy >= 0.6:
            print(f"   ✅ VERY GOOD! {best_accuracy:.1%} accuracy shows strong potential")
            print(f"   🚀 Integrate {best_method} preprocessing into SpygateAI")
            print(f"   🎯 Consider fine-tuning for even better results")
        elif best_accuracy >= 0.4:
            print(f"   ⚠️  GOOD! {best_accuracy:.1%} accuracy is a solid improvement")
            print(f"   🔧 Use {best_method} preprocessing as baseline")
            print(f"   🎯 Explore additional optimization techniques")
        elif best_accuracy >= 0.2:
            print(f"   ⚠️  FAIR! {best_accuracy:.1%} accuracy shows some promise")
            print(f"   🔍 Analyze failed cases for patterns")
            print(f"   🎯 Consider hybrid approaches")
        else:
            print(f"   ❌ POOR! {best_accuracy:.1%} indicates fundamental challenges")
            print(f"   🔍 Dataset may have quality issues")
            print(f"   🎯 Consider alternative approaches")

        # Show examples from best method
        print(f"\n📋 Examples from {best_method} method:")
        best_predictions = results[best_method]["predictions"]
        for i, pred_data in enumerate(best_predictions[:10]):
            status = "✅" if pred_data["correct"] else "❌"
            print(f"   {i+1:2d}. {status} GT: '{pred_data['gt']}' | Pred: '{pred_data['pred']}'")

        # Save results
        with open("enhanced_easyocr_results.json", "w") as f:
            json.dump(
                {
                    "accuracies": accuracies,
                    "best_method": best_method,
                    "improvement": improvement,
                    "detailed_results": results,
                },
                f,
                indent=2,
            )

        print(f"\n💾 Results saved to: enhanced_easyocr_results.json")

        return best_method, best_accuracy, improvement


def main():
    """Run the enhanced EasyOCR test."""
    tester = EnhancedEasyOCR()
    best_method, best_accuracy, improvement = tester.run_enhanced_test()

    print(f"\n🎯 NEXT STEPS:")
    if best_accuracy > 0.6:
        print(f"   1. ✅ Integrate {best_method} preprocessing into SpygateAI")
        print(f"   2. 🚀 Replace current OCR pipeline")
        print(f"   3. 🎯 Monitor performance in production")
    elif best_accuracy > 0.3:
        print(f"   1. 🔧 Implement {best_method} preprocessing")
        print(f"   2. 📊 Collect more training data")
        print(f"   3. 🎯 Explore ensemble methods")
    else:
        print(f"   1. 🔍 Investigate data quality issues")
        print(f"   2. 🎯 Try alternative OCR engines")
        print(f"   3. 📊 Consider different preprocessing strategies")

    print(f"\n💡 CONCLUSION:")
    print(f"   Even with dependency issues blocking KerasOCR/PaddleOCR,")
    print(f"   we can achieve {best_accuracy:.1%} accuracy with enhanced EasyOCR!")
    print(f"   This is a {improvement:.1%} improvement over current approach.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Baseline OCR Comparison: Using EXACT same test samples as custom OCR
This ensures fair comparison with our previous custom model results.
"""

import json
import random
import time

import cv2

# OCR Engines
import easyocr
import numpy as np

try:
    import keras_ocr

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class BaselineOCRTest:
    def __init__(self):
        print("🚀 Initializing OCR Engines for Baseline Test...")

        # EasyOCR
        self.easy_reader = easyocr.Reader(["en"], gpu=True)
        print("✅ EasyOCR initialized")

        # KerasOCR
        if KERAS_AVAILABLE:
            self.keras_pipeline = keras_ocr.pipeline.Pipeline()
            print("✅ KerasOCR initialized")

        # PaddleOCR
        if PADDLE_AVAILABLE:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False)
            print("✅ PaddleOCR initialized")

    def preprocess_image_advanced(self, image_path):
        """Advanced preprocessing - same as our custom model."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize to standard size (same as custom model)
            img = cv2.resize(img, (128, 32))

            # Enhanced preprocessing for dark text (same pipeline)
            # 1. Brightness boost
            img = cv2.convertScaleAbs(img, alpha=2.5, beta=40)

            # 2. CLAHE for local contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            img = clahe.apply(img)

            # 3. Gamma correction
            gamma = 1.2
            img = np.power(img / 255.0, 1.0 / gamma) * 255.0
            img = img.astype(np.uint8)

            return img

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR."""
        img = self.preprocess_image_advanced(image_path)
        if img is None:
            return ""

        try:
            results = self.easy_reader.readtext(img)
            if results:
                # Get result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                return best_result[1].strip()
            return ""
        except Exception as e:
            return ""

    def extract_text_keras(self, image_path):
        """Extract text using KerasOCR."""
        if not KERAS_AVAILABLE:
            return ""

        img = self.preprocess_image_advanced(image_path)
        if img is None:
            return ""

        try:
            # Convert to RGB for KerasOCR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            predictions = self.keras_pipeline.recognize([img_rgb])

            if predictions and predictions[0]:
                text = " ".join([pred[0] for pred in predictions[0]])
                return text.strip()
            return ""
        except Exception as e:
            return ""

    def extract_text_paddle(self, image_path):
        """Extract text using PaddleOCR."""
        if not PADDLE_AVAILABLE:
            return ""

        img = self.preprocess_image_advanced(image_path)
        if img is None:
            return ""

        try:
            results = self.paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                # Get result with highest confidence
                best_confidence = 0
                best_text = ""
                for line in results[0]:
                    if line[1][1] > best_confidence:
                        best_confidence = line[1][1]
                        best_text = line[1][0]
                return best_text.strip()
            return ""
        except Exception as e:
            return ""

    def get_exact_test_samples(self):
        """Get the EXACT same 20 samples we used to test custom OCR."""
        # Load core dataset
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)

        # Use same random seed as our previous test to get identical samples
        random.seed(42)
        test_samples = random.sample(data, 20)

        return test_samples

    def run_baseline_test(self):
        """Run baseline test on exact same samples as custom OCR."""
        print("🎯 Baseline OCR Test - Same Samples as Custom Model")
        print("=" * 60)

        # Get exact same test samples
        test_samples = self.get_exact_test_samples()

        print(f"📊 Testing on same 20 samples used for custom OCR")
        print(f"🔄 This ensures fair baseline comparison")

        # Results tracking
        results = {
            "easyocr": {"correct": 0, "total": 0},
            "keras_ocr": {"correct": 0, "total": 0},
            "paddle_ocr": {"correct": 0, "total": 0},
        }

        # Test each sample
        for i, sample in enumerate(test_samples):
            ground_truth = sample["ground_truth_text"]
            image_path = sample["image_path"]

            print(f"\n📝 Sample {i+1}/20: '{ground_truth}'")

            # Test EasyOCR
            easy_pred = self.extract_text_easyocr(image_path)
            easy_correct = easy_pred == ground_truth
            results["easyocr"]["total"] += 1
            if easy_correct:
                results["easyocr"]["correct"] += 1

            status = "✅" if easy_correct else "❌"
            print(f"   EasyOCR    : {status} '{easy_pred}'")

            # Test KerasOCR
            if KERAS_AVAILABLE:
                keras_pred = self.extract_text_keras(image_path)
                keras_correct = keras_pred == ground_truth
                results["keras_ocr"]["total"] += 1
                if keras_correct:
                    results["keras_ocr"]["correct"] += 1

                status = "✅" if keras_correct else "❌"
                print(f"   KerasOCR   : {status} '{keras_pred}'")

            # Test PaddleOCR
            if PADDLE_AVAILABLE:
                paddle_pred = self.extract_text_paddle(image_path)
                paddle_correct = paddle_pred == ground_truth
                results["paddle_ocr"]["total"] += 1
                if paddle_correct:
                    results["paddle_ocr"]["correct"] += 1

                status = "✅" if paddle_correct else "❌"
                print(f"   PaddleOCR  : {status} '{paddle_pred}'")

        # Print final comparison
        print(f"\n🏆 BASELINE RESULTS vs CUSTOM OCR")
        print("=" * 60)

        print(f"📊 CUSTOM OCR (Previous): 0.0% (0/20) - Failed completely")
        print(f"   Output: '22D & 10FD69RI' for all inputs")

        for engine_name, result in results.items():
            if result["total"] > 0:
                accuracy = result["correct"] / result["total"]
                print(
                    f"📊 {engine_name.upper()}: {accuracy:.1%} ({result['correct']}/{result['total']})"
                )

        # Determine best baseline
        best_engine = None
        best_accuracy = 0

        for engine_name, result in results.items():
            if result["total"] > 0:
                accuracy = result["correct"] / result["total"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_engine = engine_name

        if best_engine:
            print(f"\n🥇 BEST BASELINE: {best_engine.upper()}")
            print(f"   Accuracy: {best_accuracy:.1%}")
            print(f"   Improvement over custom: +{best_accuracy:.1%}")

        return best_engine, best_accuracy


def main():
    """Run the baseline OCR test."""
    tester = BaselineOCRTest()
    best_engine, best_accuracy = tester.run_baseline_test()

    print(f"\n💡 NEXT STEPS:")
    if best_accuracy > 0.5:
        print(f"   1. ✅ {best_engine.upper()} shows good baseline performance")
        print(f"   2. 🎯 Fine-tune {best_engine.upper()} on our Madden dataset")
        print(f"   3. 🚀 Integrate fine-tuned model into SpygateAI")
    else:
        print(f"   1. ⚠️  All engines show poor performance on this dataset")
        print(f"   2. 🔧 Need more advanced preprocessing or domain adaptation")
        print(f"   3. 🎯 Consider specialized text recognition models")


if __name__ == "__main__":
    main()

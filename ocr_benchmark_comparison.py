#!/usr/bin/env python3
"""
Expert OCR Benchmark: KerasOCR vs PaddleOCR vs EasyOCR
Comprehensive comparison to determine the best OCR engine for Madden HUD text.
"""

import json
import os
import random
import time
from collections import defaultdict

import cv2

# OCR Engines
import easyocr
import numpy as np
from PIL import Image

try:
    import keras_ocr

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  KerasOCR not available")

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️  PaddleOCR not available")


class OCRBenchmark:
    def __init__(self):
        self.results = {
            "easyocr": {"correct": 0, "total": 0, "times": [], "predictions": []},
            "keras_ocr": {"correct": 0, "total": 0, "times": [], "predictions": []},
            "paddle_ocr": {"correct": 0, "total": 0, "times": [], "predictions": []},
        }

        # Initialize OCR engines
        print("🚀 Initializing OCR Engines...")

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
        """Advanced preprocessing for dark HUD text."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize maintaining aspect ratio
            h, w = img.shape
            if h < 32 or w < 64:
                scale = max(32 / h, 64 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))

            # Advanced preprocessing pipeline
            # 1. Bilateral filter for noise reduction
            img = cv2.bilateralFilter(img, 5, 50, 50)

            # 2. CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 2))
            img = clahe.apply(img)

            # 3. Morphological operations to clean text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            # 4. Gamma correction for better contrast
            gamma = 0.8
            img = np.power(img / 255.0, gamma) * 255.0
            img = img.astype(np.uint8)

            # 5. Final brightness/contrast adjustment
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

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
            start_time = time.time()
            results = self.easy_reader.readtext(img)
            end_time = time.time()

            # Extract text with highest confidence
            if results:
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
            else:
                text = ""

            return text, end_time - start_time
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0

    def extract_text_keras(self, image_path):
        """Extract text using KerasOCR."""
        if not KERAS_AVAILABLE:
            return "", 0

        img = self.preprocess_image_advanced(image_path)
        if img is None:
            return "", 0

        try:
            # Convert to RGB for KerasOCR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            start_time = time.time()
            predictions = self.keras_pipeline.recognize([img_rgb])
            end_time = time.time()

            # Extract text
            if predictions and predictions[0]:
                text = " ".join([pred[0] for pred in predictions[0]])
            else:
                text = ""

            return text.strip(), end_time - start_time
        except Exception as e:
            print(f"KerasOCR error: {e}")
            return "", 0

    def extract_text_paddle(self, image_path):
        """Extract text using PaddleOCR."""
        if not PADDLE_AVAILABLE:
            return "", 0

        img = self.preprocess_image_advanced(image_path)
        if img is None:
            return "", 0

        try:
            start_time = time.time()
            results = self.paddle_ocr.ocr(img, cls=True)
            end_time = time.time()

            # Extract text with highest confidence
            text = ""
            if results and results[0]:
                best_confidence = 0
                for line in results[0]:
                    if line[1][1] > best_confidence:
                        best_confidence = line[1][1]
                        text = line[1][0]

            return text.strip(), end_time - start_time
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return "", 0

    def normalize_text(self, text):
        """Normalize text for comparison."""
        # Remove extra spaces and convert to uppercase
        return " ".join(text.upper().split())

    def calculate_similarity(self, pred, gt):
        """Calculate similarity between prediction and ground truth."""
        pred_norm = self.normalize_text(pred)
        gt_norm = self.normalize_text(gt)

        # Exact match
        if pred_norm == gt_norm:
            return 1.0

        # Character-level similarity
        if len(gt_norm) == 0:
            return 0.0

        # Simple character overlap
        correct_chars = sum(1 for p, g in zip(pred_norm, gt_norm) if p == g)
        return correct_chars / max(len(pred_norm), len(gt_norm))

    def run_benchmark(self, test_samples=50):
        """Run comprehensive OCR benchmark."""
        print(f"🎯 Running OCR Benchmark")
        print("=" * 60)

        # Load test data
        with open("madden_ocr_training_data_CORE.json", "r") as f:
            data = json.load(f)

        # Select random test samples
        test_data = random.sample(data, min(test_samples, len(data)))
        print(f"📊 Testing on {len(test_data)} samples")

        engines = []
        if True:  # EasyOCR always available
            engines.append(("easyocr", self.extract_text_easyocr))
        if KERAS_AVAILABLE:
            engines.append(("keras_ocr", self.extract_text_keras))
        if PADDLE_AVAILABLE:
            engines.append(("paddle_ocr", self.extract_text_paddle))

        print(f"🔧 Testing engines: {[name for name, _ in engines]}")

        # Test each sample
        for i, sample in enumerate(test_data):
            ground_truth = sample["ground_truth_text"]
            image_path = sample["image_path"]

            print(f"\n📝 Sample {i+1}/{len(test_data)}: '{ground_truth}'")

            for engine_name, extract_func in engines:
                try:
                    prediction, processing_time = extract_func(image_path)
                    similarity = self.calculate_similarity(prediction, ground_truth)

                    # Store results
                    self.results[engine_name]["total"] += 1
                    self.results[engine_name]["times"].append(processing_time)
                    self.results[engine_name]["predictions"].append(
                        {
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "similarity": similarity,
                            "time": processing_time,
                        }
                    )

                    if similarity >= 0.9:  # 90% similarity threshold
                        self.results[engine_name]["correct"] += 1

                    status = "✅" if similarity >= 0.9 else "❌"
                    print(
                        f"   {engine_name:12}: {status} '{prediction}' (sim: {similarity:.2f}, {processing_time:.3f}s)"
                    )

                except Exception as e:
                    print(f"   {engine_name:12}: ❌ ERROR: {e}")

        # Print final results
        self.print_results()

        # Return best engine
        return self.get_best_engine()

    def print_results(self):
        """Print comprehensive benchmark results."""
        print(f"\n🏆 BENCHMARK RESULTS")
        print("=" * 60)

        for engine_name, results in self.results.items():
            if results["total"] == 0:
                continue

            accuracy = results["correct"] / results["total"]
            avg_time = np.mean(results["times"]) if results["times"] else 0

            print(f"\n📊 {engine_name.upper()}:")
            print(f"   Accuracy: {accuracy:.1%} ({results['correct']}/{results['total']})")
            print(f"   Avg Time: {avg_time:.3f}s")
            print(f"   Total Time: {sum(results['times']):.2f}s")

            # Show some examples
            print(f"   Examples:")
            for pred_data in results["predictions"][:3]:
                status = "✅" if pred_data["similarity"] >= 0.9 else "❌"
                print(
                    f"     {status} GT: '{pred_data['ground_truth']}' | Pred: '{pred_data['prediction']}'"
                )

    def get_best_engine(self):
        """Determine the best performing engine."""
        best_engine = None
        best_score = 0

        for engine_name, results in self.results.items():
            if results["total"] == 0:
                continue

            accuracy = results["correct"] / results["total"]
            avg_time = np.mean(results["times"]) if results["times"] else float("inf")

            # Combined score: 70% accuracy, 30% speed (inverse)
            speed_score = 1 / (avg_time + 0.001)  # Avoid division by zero
            combined_score = 0.7 * accuracy + 0.3 * min(speed_score, 10) / 10

            if combined_score > best_score:
                best_score = combined_score
                best_engine = engine_name

        print(f"\n🥇 WINNER: {best_engine.upper()}")
        print(f"   Combined Score: {best_score:.3f}")

        return best_engine

    def save_results(self, filename="ocr_benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {filename}")


def main():
    """Run the OCR benchmark comparison."""
    benchmark = OCRBenchmark()

    # Run benchmark
    best_engine = benchmark.run_benchmark(test_samples=30)

    # Save results
    benchmark.save_results()

    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Use {best_engine.upper()} as the primary OCR engine")
    print(f"   Consider fine-tuning {best_engine.upper()} for even better results")


if __name__ == "__main__":
    main()

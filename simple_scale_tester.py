#!/usr/bin/env python3
"""
Simple Scale Factor Tester for PaddleOCR
Test different scale factors on 25 grayscale samples to find optimal scaling
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    print("⚠️  PaddleOCR not available. Install with: pip install paddleocr")
    PADDLE_AVAILABLE = False


class SimpleScaleTester:
    def __init__(self):
        """Initialize the simple scale tester"""
        self.samples_dir = Path("preprocessing_test_samples")
        self.results_dir = Path("scale_test_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize PaddleOCR with CPU to avoid CUDA issues
        if PADDLE_AVAILABLE:
            print("🔧 Initializing PaddleOCR (CPU mode)...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
        else:
            self.ocr = None

        # Test scale factors (reasonable range)
        self.scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        # Use CUBIC interpolation (good balance of quality and speed)
        self.interpolation = cv2.INTER_CUBIC

        self.results = []

    def apply_scaling(self, image, scale_factor):
        """Apply scaling with CUBIC interpolation"""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        scaled = cv2.resize(image, (new_width, new_height), interpolation=self.interpolation)
        return scaled

    def extract_text_with_paddle(self, image):
        """Extract text using PaddleOCR and return results with confidence"""
        if not self.ocr:
            return [], 0.0, 0

        try:
            # PaddleOCR expects RGB, but works with grayscale too
            results = self.ocr.ocr(image, cls=True)

            if not results or not results[0]:
                return [], 0.0, 0

            texts = []
            confidences = []

            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    texts.append(text)
                    confidences.append(confidence)

            avg_confidence = np.mean(confidences) if confidences else 0.0
            text_count = len(texts)

            return texts, avg_confidence, text_count

        except Exception as e:
            print(f"    ❌ PaddleOCR error: {e}")
            return [], 0.0, 0

    def test_all_images(self):
        """Test all 25 grayscale images with different scale factors"""
        print("🔍 SIMPLE SCALE FACTOR TESTING")
        print("=" * 50)

        if not PADDLE_AVAILABLE:
            print("❌ PaddleOCR not available. Cannot proceed.")
            return

        # Get all grayscale sample images
        sample_images = list(self.samples_dir.glob("sample_*_grayscale.png"))

        if not sample_images:
            print(f"❌ No grayscale samples found in {self.samples_dir}")
            return

        print(f"📸 Found {len(sample_images)} grayscale samples")
        print(f"🔧 Testing {len(self.scale_factors)} scale factors: {self.scale_factors}")
        print(
            f"📊 Total tests: {len(sample_images)} images × {len(self.scale_factors)} scales = {len(sample_images) * len(self.scale_factors)}"
        )
        print()

        # Test each scale factor on all images
        for scale_factor in self.scale_factors:
            print(f"🔍 Testing Scale Factor: {scale_factor}x")
            print("-" * 40)

            scale_results = []

            for i, image_path in enumerate(sorted(sample_images), 1):
                print(f"  [{i:2d}/25] {image_path.name}")

                try:
                    # Load grayscale image
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"    ❌ Failed to load image")
                        continue

                    # Apply scaling
                    scaled_image = self.apply_scaling(image, scale_factor)

                    # Extract text with PaddleOCR
                    start_time = time.time()
                    texts, avg_confidence, text_count = self.extract_text_with_paddle(scaled_image)
                    processing_time = time.time() - start_time

                    result = {
                        "image_name": image_path.name,
                        "scale_factor": scale_factor,
                        "avg_confidence": avg_confidence,
                        "text_count": text_count,
                        "processing_time": processing_time,
                        "texts": texts,
                        "original_size": f"{image.shape[1]}x{image.shape[0]}",
                        "scaled_size": f"{scaled_image.shape[1]}x{scaled_image.shape[0]}",
                    }

                    scale_results.append(result)
                    self.results.append(result)

                    print(
                        f"    ✅ Confidence: {avg_confidence:.3f}, Texts: {text_count}, Time: {processing_time:.2f}s"
                    )

                except Exception as e:
                    print(f"    ❌ Error: {e}")

            # Summary for this scale factor
            if scale_results:
                avg_conf = np.mean([r["avg_confidence"] for r in scale_results])
                avg_texts = np.mean([r["text_count"] for r in scale_results])
                avg_time = np.mean([r["processing_time"] for r in scale_results])

                print(f"  📊 Scale {scale_factor}x Summary:")
                print(f"      Average Confidence: {avg_conf:.3f}")
                print(f"      Average Text Count: {avg_texts:.1f}")
                print(f"      Average Time: {avg_time:.2f}s")
            print()

        # Final analysis
        self.analyze_results()
        self.save_results()

    def analyze_results(self):
        """Analyze results to find the best scale factor"""
        if not self.results:
            print("❌ No results to analyze")
            return

        print("🏆 SCALE FACTOR ANALYSIS")
        print("=" * 40)

        # Group by scale factor
        scale_summary = {}
        for scale_factor in self.scale_factors:
            scale_results = [r for r in self.results if r["scale_factor"] == scale_factor]

            if scale_results:
                avg_confidence = np.mean([r["avg_confidence"] for r in scale_results])
                avg_text_count = np.mean([r["text_count"] for r in scale_results])
                avg_time = np.mean([r["processing_time"] for r in scale_results])

                # Calculate success rate (images with confidence > 0.5)
                success_count = len([r for r in scale_results if r["avg_confidence"] > 0.5])
                success_rate = success_count / len(scale_results) * 100

                scale_summary[scale_factor] = {
                    "avg_confidence": avg_confidence,
                    "avg_text_count": avg_text_count,
                    "avg_time": avg_time,
                    "success_rate": success_rate,
                    "total_images": len(scale_results),
                }

        # Display results
        print("Scale Factor | Avg Confidence | Avg Texts | Success Rate | Avg Time")
        print("-" * 65)

        for scale_factor in self.scale_factors:
            if scale_factor in scale_summary:
                s = scale_summary[scale_factor]
                print(
                    f"    {scale_factor}x      |     {s['avg_confidence']:.3f}     |   {s['avg_text_count']:.1f}    |    {s['success_rate']:.1f}%     | {s['avg_time']:.2f}s"
                )

        # Find best scale factor
        best_scale = max(
            scale_summary.keys(),
            key=lambda x: scale_summary[x]["avg_confidence"] * 0.7
            + scale_summary[x]["avg_text_count"] * 0.2
            + scale_summary[x]["success_rate"] * 0.1,
        )

        print()
        print(f"🎯 RECOMMENDED SCALE FACTOR: {best_scale}x")
        best_stats = scale_summary[best_scale]
        print(f"   Average Confidence: {best_stats['avg_confidence']:.3f}")
        print(f"   Average Text Count: {best_stats['avg_text_count']:.1f}")
        print(f"   Success Rate: {best_stats['success_rate']:.1f}%")
        print(f"   Average Processing Time: {best_stats['avg_time']:.2f}s")

        return best_scale, scale_summary

    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"scale_test_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main execution function"""
    print("🚀 Starting Simple Scale Factor Testing")
    print("Testing different scale factors on 25 grayscale samples")
    print("to find the optimal scale for PaddleOCR text extraction.")
    print()

    tester = SimpleScaleTester()
    tester.test_all_images()

    print("\n✅ SCALE FACTOR TESTING COMPLETE!")
    print("📁 Check the 'scale_test_results' folder for detailed results.")


if __name__ == "__main__":
    main()

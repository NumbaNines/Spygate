#!/usr/bin/env python3
"""
CLAHE Optimizer for PaddleOCR
Test different CLAHE parameters on 25 scaled grayscale samples
Compare PaddleOCR confidence scores and text detection counts
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


class CLAHEOptimizer:
    def __init__(self):
        """Initialize the CLAHE optimizer"""
        self.samples_dir = Path("preprocessing_test_samples")
        self.results_dir = Path("clahe_optimization_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize PaddleOCR with CPU to avoid CUDA issues
        if PADDLE_AVAILABLE:
            print("🔧 Initializing PaddleOCR (CPU mode)...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
        else:
            self.ocr = None

        # CLAHE test parameters
        self.clip_limits = [1.0, 2.0, 3.0, 4.0]
        self.grid_sizes = [(4, 4), (8, 8), (16, 16)]

        self.results = []

    def apply_clahe(self, image, clip_limit, grid_size):
        """Apply CLAHE with specified parameters"""
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        # Apply CLAHE
        clahe_image = clahe.apply(image)

        return clahe_image

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

    def test_single_image(self, image_path, clip_limit, grid_size):
        """Test a single image with specific CLAHE parameters"""
        try:
            # Load scaled grayscale image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            # Apply CLAHE
            clahe_image = self.apply_clahe(image, clip_limit, grid_size)

            # Extract text with PaddleOCR
            start_time = time.time()
            texts, avg_confidence, text_count = self.extract_text_with_paddle(clahe_image)
            processing_time = time.time() - start_time

            result = {
                "image_name": image_path.name,
                "clip_limit": clip_limit,
                "grid_size": f"{grid_size[0]}x{grid_size[1]}",
                "grid_size_tuple": grid_size,
                "avg_confidence": avg_confidence,
                "text_count": text_count,
                "processing_time": processing_time,
                "texts": texts,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            print(f"    ❌ Error testing {image_path.name}: {e}")
            return None

    def run_optimization(self):
        """Run the complete CLAHE optimization"""
        print("🔍 CLAHE OPTIMIZATION FOR PADDLEOCR")
        print("=" * 50)

        if not PADDLE_AVAILABLE:
            print("❌ PaddleOCR not available. Cannot proceed.")
            return

        # Get all scaled grayscale sample images
        sample_images = list(self.samples_dir.glob("sample_*_grayscale.png"))

        if not sample_images:
            print(f"❌ No grayscale samples found in {self.samples_dir}")
            return

        print(f"📸 Found {len(sample_images)} scaled grayscale samples")
        print(f"🔧 Testing {len(self.clip_limits)} clip limits: {self.clip_limits}")
        print(f"📐 Testing {len(self.grid_sizes)} grid sizes: {self.grid_sizes}")
        print(
            f"📊 Total combinations: {len(self.clip_limits)} × {len(self.grid_sizes)} = {len(self.clip_limits) * len(self.grid_sizes)}"
        )
        print(
            f"🎯 Total tests: {len(sample_images)} images × {len(self.clip_limits) * len(self.grid_sizes)} combinations = {len(sample_images) * len(self.clip_limits) * len(self.grid_sizes)}"
        )
        print()

        total_tests = len(sample_images) * len(self.clip_limits) * len(self.grid_sizes)
        current_test = 0

        # Test each combination
        for clip_limit in self.clip_limits:
            for grid_size in self.grid_sizes:
                print(
                    f"🔍 Testing CLAHE: Clip Limit={clip_limit}, Grid Size={grid_size[0]}x{grid_size[1]}"
                )
                print("-" * 60)

                combination_results = []

                for i, image_path in enumerate(sorted(sample_images), 1):
                    current_test += 1
                    progress = (current_test / total_tests) * 100

                    print(
                        f"  [{current_test:3d}/{total_tests}] ({progress:5.1f}%) {image_path.name}"
                    )

                    result = self.test_single_image(image_path, clip_limit, grid_size)

                    if result:
                        combination_results.append(result)
                        self.results.append(result)
                        print(
                            f"    ✅ Confidence: {result['avg_confidence']:.3f}, "
                            f"Texts: {result['text_count']}, "
                            f"Time: {result['processing_time']:.2f}s"
                        )
                    else:
                        print(f"    ❌ Failed")

                # Summary for this combination
                if combination_results:
                    avg_conf = np.mean([r["avg_confidence"] for r in combination_results])
                    avg_texts = np.mean([r["text_count"] for r in combination_results])
                    avg_time = np.mean([r["processing_time"] for r in combination_results])

                    print(f"  📊 Combination Summary:")
                    print(f"      Average Confidence: {avg_conf:.3f}")
                    print(f"      Average Text Count: {avg_texts:.1f}")
                    print(f"      Average Time: {avg_time:.2f}s")
                print()

        print(f"✅ Completed {len(self.results)} successful tests")

        # Save results
        self.save_results()

        # Analyze results
        self.analyze_results()

    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"clahe_optimization_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Detailed results saved to: {results_file}")

    def analyze_results(self):
        """Analyze and summarize the optimization results"""
        if not self.results:
            print("❌ No results to analyze")
            return

        print("\n📊 CLAHE OPTIMIZATION ANALYSIS")
        print("=" * 50)

        # Overall statistics
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total tests completed: {len(self.results)}")

        confidences = [r["avg_confidence"] for r in self.results]
        text_counts = [r["text_count"] for r in self.results]
        times = [r["processing_time"] for r in self.results]

        print(f"   Average confidence: {np.mean(confidences):.3f}")
        print(f"   Average text count: {np.mean(text_counts):.1f}")
        print(f"   Average processing time: {np.mean(times):.3f}s")
        print()

        # Group results by combination
        combinations = {}
        for result in self.results:
            key = f"Clip_{result['clip_limit']}_Grid_{result['grid_size']}"
            if key not in combinations:
                combinations[key] = []
            combinations[key].append(result)

        # Analyze each combination
        print("🏆 RESULTS BY CLAHE COMBINATION:")
        print("Combination                | Avg Confidence | Avg Texts | Success Rate | Avg Time")
        print("-" * 80)

        combination_summary = {}
        for combo_name, combo_results in combinations.items():
            avg_confidence = np.mean([r["avg_confidence"] for r in combo_results])
            avg_text_count = np.mean([r["text_count"] for r in combo_results])
            avg_time = np.mean([r["processing_time"] for r in combo_results])

            # Calculate success rate (images with confidence > 0.5)
            success_count = len([r for r in combo_results if r["avg_confidence"] > 0.5])
            success_rate = success_count / len(combo_results) * 100

            combination_summary[combo_name] = {
                "avg_confidence": avg_confidence,
                "avg_text_count": avg_text_count,
                "avg_time": avg_time,
                "success_rate": success_rate,
                "clip_limit": combo_results[0]["clip_limit"],
                "grid_size": combo_results[0]["grid_size"],
            }

            print(
                f"{combo_name:26} |     {avg_confidence:.3f}     |   {avg_text_count:.1f}    |    {success_rate:.1f}%     | {avg_time:.2f}s"
            )

        print()

        # Find best combination
        best_combo = max(
            combination_summary.keys(),
            key=lambda x: combination_summary[x]["avg_confidence"] * 0.7
            + combination_summary[x]["avg_text_count"] * 0.2
            + combination_summary[x]["success_rate"] * 0.1,
        )

        print(f"🎯 RECOMMENDED CLAHE SETTINGS: {best_combo}")
        best_stats = combination_summary[best_combo]
        print(f"   Clip Limit: {best_stats['clip_limit']}")
        print(f"   Grid Size: {best_stats['grid_size']}")
        print(f"   Average Confidence: {best_stats['avg_confidence']:.3f}")
        print(f"   Average Text Count: {best_stats['avg_text_count']:.1f}")
        print(f"   Success Rate: {best_stats['success_rate']:.1f}%")
        print(f"   Average Processing Time: {best_stats['avg_time']:.2f}s")

        # Save summary
        summary_file = self.results_dir / "clahe_optimization_summary.txt"
        with open(summary_file, "w") as f:
            f.write("CLAHE Optimization Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total tests: {len(self.results)}\n")
            f.write(f"Average confidence: {np.mean(confidences):.3f}\n")
            f.write(f"Average text count: {np.mean(text_counts):.1f}\n\n")
            f.write(f"Best combination: {best_combo}\n")
            f.write(f"Clip Limit: {best_stats['clip_limit']}\n")
            f.write(f"Grid Size: {best_stats['grid_size']}\n")
            f.write(f"Confidence: {best_stats['avg_confidence']:.3f}\n")
            f.write(f"Text Count: {best_stats['avg_text_count']:.1f}\n")
            f.write(f"Success Rate: {best_stats['success_rate']:.1f}%\n")

        print(f"\n💾 Summary saved to: {summary_file}")

        return best_combo, combination_summary


def main():
    """Main execution function"""
    print("🚀 Starting CLAHE Optimization for PaddleOCR")
    print("This will test different CLAHE parameters on all 25 scaled grayscale samples")
    print("to find optimal contrast enhancement settings.")
    print()

    optimizer = CLAHEOptimizer()
    optimizer.run_optimization()

    print("\n✅ CLAHE OPTIMIZATION COMPLETE!")
    print("📁 Check the 'clahe_optimization_results' folder for:")
    print("   - Detailed JSON results")
    print("   - Summary analysis")
    print("   - Recommended CLAHE settings")


if __name__ == "__main__":
    main()

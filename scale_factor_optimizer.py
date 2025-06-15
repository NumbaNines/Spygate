#!/usr/bin/env python3
"""
Scale Factor Optimization for PaddleOCR
Test different scale factors and interpolation methods on 25 grayscale samples
Compare PaddleOCR confidence scores and text detection performance
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    print("⚠️  PaddleOCR not available. Install with: pip install paddleocr")
    PADDLE_AVAILABLE = False


class ScaleFactorOptimizer:
    def __init__(self):
        """Initialize the scale factor optimizer"""
        self.samples_dir = Path("preprocessing_test_samples")
        self.results_dir = Path("scale_optimization_results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize PaddleOCR
        if PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        else:
            self.ocr = None

        # Test parameters
        self.scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.interpolation_methods = {
            "NEAREST": cv2.INTER_NEAREST,
            "LINEAR": cv2.INTER_LINEAR,
            "CUBIC": cv2.INTER_CUBIC,
            "LANCZOS4": cv2.INTER_LANCZOS4,
            "AREA": cv2.INTER_AREA,
        }

        self.results = []

    def apply_scaling(self, image, scale_factor, interpolation):
        """Apply scaling with specified interpolation method"""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        scaled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
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

    def test_single_image(self, image_path, scale_factor, interpolation_name, interpolation_method):
        """Test a single image with specific scale factor and interpolation"""
        try:
            # Load grayscale image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            # Apply scaling
            scaled_image = self.apply_scaling(image, scale_factor, interpolation_method)

            # Extract text with PaddleOCR
            start_time = time.time()
            texts, avg_confidence, text_count = self.extract_text_with_paddle(scaled_image)
            processing_time = time.time() - start_time

            # Calculate image size increase
            original_size = image.shape[0] * image.shape[1]
            scaled_size = scaled_image.shape[0] * scaled_image.shape[1]
            size_ratio = scaled_size / original_size

            result = {
                "image_name": image_path.name,
                "scale_factor": scale_factor,
                "interpolation": interpolation_name,
                "avg_confidence": avg_confidence,
                "text_count": text_count,
                "processing_time": processing_time,
                "original_size": original_size,
                "scaled_size": scaled_size,
                "size_ratio": size_ratio,
                "texts": texts,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            print(f"    ❌ Error testing {image_path.name}: {e}")
            return None

    def run_optimization(self):
        """Run the complete scale factor optimization"""
        print("🔍 SCALE FACTOR OPTIMIZATION FOR PADDLEOCR")
        print("=" * 60)

        if not PADDLE_AVAILABLE:
            print("❌ PaddleOCR not available. Cannot proceed.")
            return

        # Get all grayscale sample images
        sample_images = list(self.samples_dir.glob("sample_*_grayscale.png"))

        if not sample_images:
            print(f"❌ No grayscale samples found in {self.samples_dir}")
            return

        print(f"📸 Found {len(sample_images)} grayscale samples")
        print(f"🔧 Testing {len(self.scale_factors)} scale factors")
        print(f"🎨 Testing {len(self.interpolation_methods)} interpolation methods")
        print(
            f"📊 Total tests: {len(sample_images) * len(self.scale_factors) * len(self.interpolation_methods)}"
        )
        print()

        total_tests = len(sample_images) * len(self.scale_factors) * len(self.interpolation_methods)
        current_test = 0

        # Test each combination
        for image_path in sorted(sample_images):
            print(f"🖼️  Processing: {image_path.name}")

            for scale_factor in self.scale_factors:
                for interp_name, interp_method in self.interpolation_methods.items():
                    current_test += 1
                    progress = (current_test / total_tests) * 100

                    print(
                        f"  [{current_test:3d}/{total_tests}] ({progress:5.1f}%) "
                        f"Scale: {scale_factor}x, Interpolation: {interp_name}"
                    )

                    result = self.test_single_image(
                        image_path, scale_factor, interp_name, interp_method
                    )

                    if result:
                        self.results.append(result)
                        print(
                            f"    ✅ Confidence: {result['avg_confidence']:.3f}, "
                            f"Texts: {result['text_count']}, "
                            f"Time: {result['processing_time']:.3f}s"
                        )
                    else:
                        print(f"    ❌ Failed")

        print()
        print(f"✅ Completed {len(self.results)} successful tests")

        # Save results
        self.save_results()

        # Analyze results
        self.analyze_results()

        # Create visualizations
        self.create_visualizations()

    def save_results(self):
        """Save detailed results to JSON"""
        results_file = (
            self.results_dir
            / f"scale_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Detailed results saved to: {results_file}")

    def analyze_results(self):
        """Analyze and summarize the optimization results"""
        if not self.results:
            print("❌ No results to analyze")
            return

        print("\n📊 SCALE FACTOR OPTIMIZATION ANALYSIS")
        print("=" * 50)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)

        # Overall statistics
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total tests completed: {len(df)}")
        print(f"   Average confidence: {df['avg_confidence'].mean():.3f}")
        print(f"   Average text count: {df['text_count'].mean():.1f}")
        print(f"   Average processing time: {df['processing_time'].mean():.3f}s")
        print()

        # Best results by scale factor
        print("🏆 BEST RESULTS BY SCALE FACTOR:")
        scale_summary = (
            df.groupby("scale_factor")
            .agg({"avg_confidence": "mean", "text_count": "mean", "processing_time": "mean"})
            .round(3)
        )

        for scale_factor, row in scale_summary.iterrows():
            print(
                f"   {scale_factor}x: Confidence={row['avg_confidence']:.3f}, "
                f"Texts={row['text_count']:.1f}, Time={row['processing_time']:.3f}s"
            )
        print()

        # Best results by interpolation method
        print("🎨 BEST RESULTS BY INTERPOLATION METHOD:")
        interp_summary = (
            df.groupby("interpolation")
            .agg({"avg_confidence": "mean", "text_count": "mean", "processing_time": "mean"})
            .round(3)
        )

        for interp_method, row in interp_summary.iterrows():
            print(
                f"   {interp_method}: Confidence={row['avg_confidence']:.3f}, "
                f"Texts={row['text_count']:.1f}, Time={row['processing_time']:.3f}s"
            )
        print()

        # Find optimal combinations
        print("🎯 TOP 10 OPTIMAL COMBINATIONS:")
        # Score based on confidence (70%) + text_count (20%) + speed (10%, inverted)
        df["composite_score"] = (
            df["avg_confidence"] * 0.7
            + (df["text_count"] / df["text_count"].max()) * 0.2
            + (1 - df["processing_time"] / df["processing_time"].max()) * 0.1
        )

        top_combinations = df.nlargest(10, "composite_score")[
            [
                "scale_factor",
                "interpolation",
                "avg_confidence",
                "text_count",
                "processing_time",
                "composite_score",
            ]
        ]

        for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
            print(
                f"   {i:2d}. {row['scale_factor']}x {row['interpolation']}: "
                f"Confidence={row['avg_confidence']:.3f}, "
                f"Texts={row['text_count']:.0f}, "
                f"Time={row['processing_time']:.3f}s, "
                f"Score={row['composite_score']:.3f}"
            )

        # Save summary
        summary_file = self.results_dir / "optimization_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Scale Factor Optimization Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total tests: {len(df)}\n")
            f.write(f"Average confidence: {df['avg_confidence'].mean():.3f}\n")
            f.write(f"Average text count: {df['text_count'].mean():.1f}\n\n")
            f.write("Top 5 combinations:\n")
            for i, (_, row) in enumerate(top_combinations.head().iterrows(), 1):
                f.write(
                    f"{i}. {row['scale_factor']}x {row['interpolation']}: "
                    f"Confidence={row['avg_confidence']:.3f}, "
                    f"Texts={row['text_count']:.0f}\n"
                )

        print(f"\n💾 Summary saved to: {summary_file}")

    def create_visualizations(self):
        """Create visualization plots of the results"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Scale Factor Optimization Results", fontsize=16, fontweight="bold")

        # 1. Confidence by Scale Factor
        ax1 = axes[0, 0]
        scale_conf = df.groupby("scale_factor")["avg_confidence"].mean()
        ax1.plot(scale_conf.index, scale_conf.values, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Scale Factor")
        ax1.set_ylabel("Average Confidence")
        ax1.set_title("Confidence vs Scale Factor")
        ax1.grid(True, alpha=0.3)

        # 2. Text Count by Scale Factor
        ax2 = axes[0, 1]
        scale_text = df.groupby("scale_factor")["text_count"].mean()
        ax2.plot(
            scale_text.index, scale_text.values, "o-", linewidth=2, markersize=8, color="orange"
        )
        ax2.set_xlabel("Scale Factor")
        ax2.set_ylabel("Average Text Count")
        ax2.set_title("Text Detection vs Scale Factor")
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap: Confidence by Scale Factor and Interpolation
        ax3 = axes[1, 0]
        pivot_conf = df.pivot_table(
            values="avg_confidence", index="interpolation", columns="scale_factor"
        )
        sns.heatmap(pivot_conf, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax3)
        ax3.set_title("Confidence Heatmap")

        # 4. Processing Time vs Scale Factor
        ax4 = axes[1, 1]
        scale_time = df.groupby("scale_factor")["processing_time"].mean()
        ax4.plot(scale_time.index, scale_time.values, "o-", linewidth=2, markersize=8, color="red")
        ax4.set_xlabel("Scale Factor")
        ax4.set_ylabel("Processing Time (seconds)")
        ax4.set_title("Processing Time vs Scale Factor")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_file = self.results_dir / "scale_optimization_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 Visualizations saved to: {plot_file}")

        # Create detailed interpolation comparison
        self.create_interpolation_comparison(df)

    def create_interpolation_comparison(self, df):
        """Create detailed comparison of interpolation methods"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Interpolation Method Comparison", fontsize=16, fontweight="bold")

        # 1. Box plot of confidence by interpolation
        ax1 = axes[0]
        df.boxplot(column="avg_confidence", by="interpolation", ax=ax1)
        ax1.set_title("Confidence Distribution by Interpolation")
        ax1.set_xlabel("Interpolation Method")
        ax1.set_ylabel("Confidence")

        # 2. Box plot of text count by interpolation
        ax2 = axes[1]
        df.boxplot(column="text_count", by="interpolation", ax=ax2)
        ax2.set_title("Text Count Distribution by Interpolation")
        ax2.set_xlabel("Interpolation Method")
        ax2.set_ylabel("Text Count")

        # 3. Processing time by interpolation
        ax3 = axes[2]
        df.boxplot(column="processing_time", by="interpolation", ax=ax3)
        ax3.set_title("Processing Time by Interpolation")
        ax3.set_xlabel("Interpolation Method")
        ax3.set_ylabel("Processing Time (s)")

        plt.tight_layout()

        # Save the plot
        interp_plot_file = self.results_dir / "interpolation_comparison.png"
        plt.savefig(interp_plot_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"📊 Interpolation comparison saved to: {interp_plot_file}")


def main():
    """Main execution function"""
    print("🚀 Starting Scale Factor Optimization for PaddleOCR")
    print("This will test different scale factors and interpolation methods")
    print("on all 25 grayscale samples to find optimal preprocessing parameters.")
    print()

    optimizer = ScaleFactorOptimizer()
    optimizer.run_optimization()

    print("\n✅ SCALE FACTOR OPTIMIZATION COMPLETE!")
    print("📁 Check the 'scale_optimization_results' folder for:")
    print("   - Detailed JSON results")
    print("   - Summary analysis")
    print("   - Visualization plots")
    print("   - Interpolation method comparisons")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simplified Final Pipeline vs Original Images Test
Optimized: No gamma + Adaptive threshold (block=11, C=5) + No sharpening
vs Original: Raw grayscale images
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class SimpleFinalComparison:
    """Simplified test comparing optimized pipeline vs original unprocessed images"""

    def __init__(self):
        # Optimized pipeline parameters
        self.scale_factor = 2.0
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (4, 4)
        self.blur_kernel_size = (3, 3)
        self.blur_sigma = 0.5
        self.morph_kernel_size = (3, 3)

        # Adaptive threshold parameters (optimized)
        self.block_size = 11
        self.c_value = 5

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def scale_image_lanczos4(self, image):
        """Scale image using LANCZOS4 interpolation"""
        height, width = image.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def apply_clahe_enhancement(self, image):
        """Apply CLAHE enhancement"""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        return clahe.apply(image)

    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)

    def apply_adaptive_threshold(self, image):
        """Apply adaptive thresholding with optimized parameters"""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.block_size, self.c_value
        )

    def apply_morphological_closing(self, image):
        """Apply morphological closing"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def extract_text_with_ocr(self, image):
        """Extract text using PaddleOCR"""
        try:
            results = self.ocr.ocr(image, cls=True)
            if not results or not results[0]:
                return []

            detections = []
            for line in results[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line
                    if confidence > 0.3:
                        detections.append({"text": text, "confidence": confidence})
            return detections
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return []

    def process_optimized_pipeline(self, image_path):
        """Process image with optimized pipeline (no gamma, no sharpening)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            filename = Path(image_path).stem

            # Optimized 7-stage pipeline (removed gamma correction and sharpening)
            gray_image = self.convert_to_grayscale(image)
            scaled_image = self.scale_image_lanczos4(gray_image)
            clahe_image = self.apply_clahe_enhancement(scaled_image)
            # NO gamma correction (skipped)
            blurred_image = self.apply_gaussian_blur(clahe_image)
            adaptive_image = self.apply_adaptive_threshold(blurred_image)
            final_image = self.apply_morphological_closing(adaptive_image)
            # NO sharpening (skipped)

            detections = self.extract_text_with_ocr(final_image)

            return {
                "filename": filename,
                "method": "optimized_pipeline",
                "detections": detections,
                "detection_count": len(detections),
                "avg_confidence": (
                    np.mean([d["confidence"] for d in detections]) if detections else 0.0
                ),
                "success": True,
            }
        except Exception as e:
            return {
                "filename": Path(image_path).stem,
                "method": "optimized_pipeline",
                "error": str(e),
                "success": False,
            }

    def process_original_image(self, image_path):
        """Process original unprocessed grayscale image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            filename = Path(image_path).stem

            # Only convert to grayscale - no other processing
            gray_image = self.convert_to_grayscale(image)

            detections = self.extract_text_with_ocr(gray_image)

            return {
                "filename": filename,
                "method": "original_unprocessed",
                "detections": detections,
                "detection_count": len(detections),
                "avg_confidence": (
                    np.mean([d["confidence"] for d in detections]) if detections else 0.0
                ),
                "success": True,
            }
        except Exception as e:
            return {
                "filename": Path(image_path).stem,
                "method": "original_unprocessed",
                "error": str(e),
                "success": False,
            }

    def run_comparison_test(self, input_dir):
        """Run comparison test between optimized pipeline and original images"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {input_dir}")
            return []

        print(f"🔬 FINAL PIPELINE vs ORIGINAL COMPARISON")
        print("=" * 80)
        print(f"Testing {len(image_files)} images...")
        print(f"Optimized Pipeline: No gamma + Adaptive(block=11, C=5) + No sharpening")
        print(f"Original: Raw grayscale only")
        print("=" * 80)

        all_results = []

        for i, image_file in enumerate(image_files, 1):
            print(f"\n{i:2d}/{len(image_files)} Processing: {image_file.stem}")
            print("-" * 60)

            # Test optimized pipeline
            start_time = time.time()
            optimized_result = self.process_optimized_pipeline(str(image_file))
            optimized_time = time.time() - start_time

            if optimized_result and optimized_result["success"]:
                optimized_result["process_time"] = optimized_time
                print(
                    f"  ✅ Optimized:  {optimized_result['detection_count']:2d} detections | "
                    f"Avg conf: {optimized_result['avg_confidence']:.3f} | {optimized_time:.3f}s"
                )
            else:
                print(f"  ❌ Optimized:  Error")

            # Test original unprocessed
            start_time = time.time()
            original_result = self.process_original_image(str(image_file))
            original_time = time.time() - start_time

            if original_result and original_result["success"]:
                original_result["process_time"] = original_time
                print(
                    f"  ✅ Original:   {original_result['detection_count']:2d} detections | "
                    f"Avg conf: {original_result['avg_confidence']:.3f} | {original_time:.3f}s"
                )
            else:
                print(f"  ❌ Original:   Error")

            # Compare results
            if (
                optimized_result
                and optimized_result["success"]
                and original_result
                and original_result["success"]
            ):
                detection_improvement = (
                    optimized_result["detection_count"] - original_result["detection_count"]
                )
                confidence_improvement = (
                    optimized_result["avg_confidence"] - original_result["avg_confidence"]
                )

                print(
                    f"  📊 Comparison: {detection_improvement:+2d} detections | "
                    f"{confidence_improvement:+.3f} confidence | "
                    f"{'🚀 BETTER' if detection_improvement > 0 or confidence_improvement > 0 else '📉 WORSE'}"
                )

                all_results.append(
                    {
                        "filename": image_file.stem,
                        "optimized": optimized_result,
                        "original": original_result,
                        "detection_improvement": detection_improvement,
                        "confidence_improvement": confidence_improvement,
                    }
                )

        return all_results

    def print_final_summary(self, results):
        """Print comprehensive comparison summary"""
        if not results:
            print("No results to summarize")
            return

        print(f"\n{'='*80}")
        print(f"FINAL PIPELINE vs ORIGINAL COMPARISON SUMMARY")
        print(f"{'='*80}")

        # Calculate overall statistics
        total_images = len(results)
        optimized_detections = [r["optimized"]["detection_count"] for r in results]
        original_detections = [r["original"]["detection_count"] for r in results]
        optimized_confidence = [r["optimized"]["avg_confidence"] for r in results]
        original_confidence = [r["original"]["avg_confidence"] for r in results]

        avg_optimized_detections = np.mean(optimized_detections)
        avg_original_detections = np.mean(original_detections)
        avg_optimized_confidence = np.mean(optimized_confidence)
        avg_original_confidence = np.mean(original_confidence)

        detection_improvement = avg_optimized_detections - avg_original_detections
        confidence_improvement = avg_optimized_confidence - avg_original_confidence

        print(f"📊 OVERALL RESULTS:")
        print(f"   Images tested: {total_images}")
        print(
            f"   Optimized pipeline: {avg_optimized_detections:.1f} avg detections, {avg_optimized_confidence:.3f} avg confidence"
        )
        print(
            f"   Original unprocessed: {avg_original_detections:.1f} avg detections, {avg_original_confidence:.3f} avg confidence"
        )
        print(
            f"   Improvement: {detection_improvement:+.1f} detections, {confidence_improvement:+.3f} confidence"
        )

        # Count wins/losses
        better_detection = sum(1 for r in results if r["detection_improvement"] > 0)
        better_confidence = sum(1 for r in results if r["confidence_improvement"] > 0)

        print(f"\n🏆 WIN/LOSS RECORD:")
        print(
            f"   Optimized wins (detections): {better_detection}/{total_images} ({better_detection/total_images*100:.1f}%)"
        )
        print(
            f"   Optimized wins (confidence): {better_confidence}/{total_images} ({better_confidence/total_images*100:.1f}%)"
        )

        # Show top 5 best improvements
        sorted_by_detection = sorted(
            results, key=lambda x: x["detection_improvement"], reverse=True
        )
        sorted_by_confidence = sorted(
            results, key=lambda x: x["confidence_improvement"], reverse=True
        )

        print(f"\n🚀 TOP 5 DETECTION IMPROVEMENTS:")
        for i, result in enumerate(sorted_by_detection[:5], 1):
            opt_det = result["optimized"]["detection_count"]
            orig_det = result["original"]["detection_count"]
            improvement = result["detection_improvement"]
            print(
                f"   {i}. {result['filename'][:40]:<40} {orig_det} → {opt_det} ({improvement:+d})"
            )

        print(f"\n🎯 TOP 5 CONFIDENCE IMPROVEMENTS:")
        for i, result in enumerate(sorted_by_confidence[:5], 1):
            opt_conf = result["optimized"]["avg_confidence"]
            orig_conf = result["original"]["avg_confidence"]
            improvement = result["confidence_improvement"]
            print(
                f"   {i}. {result['filename'][:40]:<40} {orig_conf:.3f} → {opt_conf:.3f} ({improvement:+.3f})"
            )

        # Overall verdict
        print(f"\n🏁 FINAL VERDICT:")
        if detection_improvement > 0 and confidence_improvement > 0:
            print(f"   🚀 OPTIMIZED PIPELINE WINS! Better in both detections and confidence")
        elif detection_improvement > 0:
            print(f"   🎯 OPTIMIZED PIPELINE WINS! Better detections, similar confidence")
        elif confidence_improvement > 0:
            print(f"   🎯 OPTIMIZED PIPELINE WINS! Better confidence, similar detections")
        else:
            print(f"   🤔 MIXED RESULTS - Need further analysis")

        print("=" * 80)


def main():
    """Main execution function"""
    tester = SimpleFinalComparison()

    print("Final Pipeline vs Original Images Test")
    print("=" * 60)
    print("Optimized Pipeline:")
    print("  1. Grayscale conversion")
    print("  2. 2x LANCZOS4 scaling")
    print("  3. CLAHE enhancement")
    print("  4. NO gamma correction")
    print("  5. Gaussian blur")
    print("  6. Adaptive threshold (block=11, C=5)")
    print("  7. Morphological closing")
    print("  8. NO sharpening")
    print()
    print("Original: Raw grayscale only")
    print("=" * 60)

    # Run comparison
    results = tester.run_comparison_test("preprocessing_test_samples")

    # Print summary
    tester.print_final_summary(results)


if __name__ == "__main__":
    main()

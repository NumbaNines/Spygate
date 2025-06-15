#!/usr/bin/env python3
"""
Gamma and Adaptive Threshold Optimizer for Madden HUD OCR
Tests gamma values: 0.0, 0.8, 0.5, 1.2, 1.5
Tests adaptive threshold: blockSize=11,21 and C=2,5
"""

import itertools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class GammaAdaptiveThresholdOptimizer:
    """Test different gamma and adaptive threshold combinations"""

    def __init__(self):
        self.scale_factor = 2.0
        self.clahe_clip_limit = 2.0
        self.clahe_grid_size = (4, 4)
        self.blur_kernel_size = (3, 3)
        self.blur_sigma = 0.5
        self.morph_kernel_size = (3, 3)

        # Test parameters
        self.gamma_values = [0.0, 0.5, 0.8, 1.2, 1.5]
        self.block_sizes = [11, 21]
        self.c_values = [2, 5]

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def scale_image_lanczos4(self, image):
        height, width = image.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def apply_clahe_enhancement(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        return clahe.apply(image)

    def apply_gamma_correction(self, image, gamma):
        if gamma == 0.0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
            "uint8"
        )
        return cv2.LUT(image, table)

    def apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)

    def apply_adaptive_threshold(self, image, block_size, c_value):
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value
        )

    def apply_morphological_closing(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def apply_sharpening_filter(self, image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def extract_text_with_ocr(self, image):
        try:
            results = self.ocr.ocr(image, cls=True)
            if not results or not results[0]:
                return []
            detections = []
            for line in results[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line
                    if confidence > 0.3:
                        detections.append({"text": text, "confidence": confidence, "bbox": bbox})
            return detections
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return []

    def process_single_combination(self, image_path, gamma, block_size, c_value):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            filename = Path(image_path).stem

            # Pipeline processing
            gray_image = self.convert_to_grayscale(image)
            scaled_image = self.scale_image_lanczos4(gray_image)
            clahe_image = self.apply_clahe_enhancement(scaled_image)

            if gamma == 0.0:
                gamma_image = clahe_image
            else:
                gamma_image = self.apply_gamma_correction(clahe_image, gamma)

            blurred_image = self.apply_gaussian_blur(gamma_image)
            adaptive_image = self.apply_adaptive_threshold(blurred_image, block_size, c_value)
            closed_image = self.apply_morphological_closing(adaptive_image)
            final_image = self.apply_sharpening_filter(closed_image)

            detections = self.extract_text_with_ocr(final_image)

            return {
                "filename": filename,
                "gamma": gamma,
                "block_size": block_size,
                "c_value": c_value,
                "detections": detections,
                "detection_count": len(detections),
                "avg_confidence": (
                    np.mean([d["confidence"] for d in detections]) if detections else 0.0
                ),
                "final_image": final_image,
                "success": True,
            }
        except Exception as e:
            return {
                "filename": Path(image_path).stem,
                "gamma": gamma,
                "block_size": block_size,
                "c_value": c_value,
                "error": str(e),
                "success": False,
            }

    def test_all_combinations(self, input_dir, output_dir, max_images=5):
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions][
            :max_images
        ]

        if not image_files:
            print(f"No image files found in {input_dir}")
            return []

        os.makedirs(output_dir, exist_ok=True)
        combinations = list(itertools.product(self.gamma_values, self.block_sizes, self.c_values))

        print(f"Testing {len(combinations)} parameter combinations on {len(image_files)} images...")
        print(f"Gamma values: {self.gamma_values}")
        print(f"Block sizes: {self.block_sizes}")
        print(f"C values: {self.c_values}")
        print("=" * 80)

        all_results = []

        for i, (gamma, block_size, c_value) in enumerate(combinations, 1):
            print(
                f"\n{i:2d}/{len(combinations)} Testing: γ={gamma}, block={block_size}, C={c_value}"
            )
            print("-" * 60)

            combination_results = []

            for j, image_file in enumerate(image_files, 1):
                start_time = time.time()
                result = self.process_single_combination(
                    str(image_file), gamma, block_size, c_value
                )
                process_time = time.time() - start_time

                if result and result["success"]:
                    result["process_time"] = process_time
                    combination_results.append(result)

                    combo_dir = os.path.join(
                        output_dir, f"gamma_{gamma}_block_{block_size}_c_{c_value}"
                    )
                    os.makedirs(combo_dir, exist_ok=True)
                    output_path = os.path.join(combo_dir, f"{result['filename']}_processed.png")
                    cv2.imwrite(output_path, result["final_image"])

                    print(
                        f"  {j}/{len(image_files)} ✅ {result['filename']:<20} | "
                        f"{result['detection_count']:2d} detections | "
                        f"Avg conf: {result['avg_confidence']:.3f} | "
                        f"{process_time:.3f}s"
                    )
                else:
                    print(f"  {j}/{len(image_files)} ❌ {Path(image_file).stem:<20} | Error")

            if combination_results:
                avg_detections = np.mean([r["detection_count"] for r in combination_results])
                avg_confidence = np.mean([r["avg_confidence"] for r in combination_results])
                avg_time = np.mean([r["process_time"] for r in combination_results])

                combo_summary = {
                    "gamma": gamma,
                    "block_size": block_size,
                    "c_value": c_value,
                    "avg_detections": avg_detections,
                    "avg_confidence": avg_confidence,
                    "avg_time": avg_time,
                    "success_rate": len(combination_results) / len(image_files),
                    "individual_results": combination_results,
                }

                all_results.append(combo_summary)
                print(
                    f"  📊 Summary: {avg_detections:.1f} detections, {avg_confidence:.3f} confidence, {avg_time:.3f}s"
                )

        return all_results


def main():
    optimizer = GammaAdaptiveThresholdOptimizer()

    print("Gamma & Adaptive Threshold Optimization for Madden HUD OCR")
    print("=" * 60)
    print(f"Testing gamma values: {optimizer.gamma_values}")
    print(f"Testing block sizes: {optimizer.block_sizes}")
    print(f"Testing C values: {optimizer.c_values}")
    print(
        f"Total combinations: {len(optimizer.gamma_values) * len(optimizer.block_sizes) * len(optimizer.c_values)}"
    )
    print("=" * 60)

    results = optimizer.test_all_combinations(
        "preprocessing_test_samples", "gamma_adaptive_optimization_results", max_images=5
    )

    # Print summary
    if results:
        print(f"\n{'='*80}")
        print(f"GAMMA & ADAPTIVE THRESHOLD OPTIMIZATION SUMMARY")
        print(f"{'='*80}")

        results_sorted = sorted(
            results, key=lambda x: (x["avg_detections"], x["avg_confidence"]), reverse=True
        )

        print(
            f"{'Rank':<4} {'Gamma':<6} {'Block':<6} {'C':<3} {'Detections':<11} {'Confidence':<11} {'Time':<8}"
        )
        print("-" * 80)

        for i, result in enumerate(results_sorted, 1):
            print(
                f"{i:<4} {result['gamma']:<6} {result['block_size']:<6} {result['c_value']:<3} "
                f"{result['avg_detections']:<11.1f} {result['avg_confidence']:<11.3f} {result['avg_time']:<8.3f}s"
            )

        best = results_sorted[0]
        print(f"\n🏆 BEST COMBINATION:")
        print(f"   Gamma: {best['gamma']}")
        print(f"   Block size: {best['block_size']}")
        print(f"   C value: {best['c_value']}")
        print(f"   Average detections: {best['avg_detections']:.1f}")
        print(f"   Average confidence: {best['avg_confidence']:.3f}")
        print(f"   Average time: {best['avg_time']:.3f}s")

        with open("gamma_adaptive_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Detailed results saved to: gamma_adaptive_optimization_results.json")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

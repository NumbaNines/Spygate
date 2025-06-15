#!/usr/bin/env python3
"""
Comprehensive Random Parameter Combination Test
Tests random combinations of:
- Gamma values: 0, 0.5, 0.8, 1.2, 1.5
- Scale sizes: 1.5, 2, 2.5
- Thresholding: Otsu vs Adaptive
- Sharpening: On vs Off
- CLAHE clip sizes: 1, 2, 3
- CLAHE grid sizes: 4x4 vs 8x8
"""

import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class ComprehensiveParameterTest:
    """Test random combinations of preprocessing parameters"""

    def __init__(self):
        # Fixed parameters (not changing)
        self.blur_kernel_size = (3, 3)
        self.blur_sigma = 0.5
        self.morph_kernel_size = (3, 3)
        self.adaptive_block_size = 11
        self.adaptive_c_value = 5

        # Variable parameters to test
        self.gamma_values = [0.0, 0.5, 0.8, 1.2, 1.5]
        self.scale_sizes = [1.5, 2.0, 2.5]
        self.threshold_methods = ["otsu", "adaptive"]
        self.sharpening_options = [True, False]
        self.clahe_clip_sizes = [1.0, 2.0, 3.0]
        self.clahe_grid_sizes = [(4, 4), (8, 8)]

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def scale_image_lanczos4(self, image, scale_factor):
        """Scale image using LANCZOS4 interpolation"""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    def apply_clahe_enhancement(self, image, clip_limit, grid_size):
        """Apply CLAHE enhancement"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

    def apply_gamma_correction(self, image, gamma):
        """Apply gamma correction"""
        if gamma == 0.0:
            return image  # Skip gamma correction

        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
            "uint8"
        )
        return cv2.LUT(image, table)

    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)

    def apply_otsu_threshold(self, image):
        """Apply Otsu thresholding"""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def apply_adaptive_threshold(self, image):
        """Apply adaptive thresholding"""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            self.adaptive_c_value,
        )

    def apply_morphological_closing(self, image):
        """Apply morphological closing"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def apply_sharpening_filter(self, image):
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

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

    def generate_random_parameters(self):
        """Generate random combination of parameters"""
        return {
            "gamma": random.choice(self.gamma_values),
            "scale": random.choice(self.scale_sizes),
            "threshold_method": random.choice(self.threshold_methods),
            "sharpening": random.choice(self.sharpening_options),
            "clahe_clip": random.choice(self.clahe_clip_sizes),
            "clahe_grid": random.choice(self.clahe_grid_sizes),
        }

    def process_image_with_parameters(self, image_path, params):
        """Process image with specific parameter combination"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            filename = Path(image_path).stem

            # Processing pipeline with variable parameters
            gray_image = self.convert_to_grayscale(image)
            scaled_image = self.scale_image_lanczos4(gray_image, params["scale"])
            clahe_image = self.apply_clahe_enhancement(
                scaled_image, params["clahe_clip"], params["clahe_grid"]
            )

            # Apply gamma correction if not 0
            if params["gamma"] != 0.0:
                gamma_image = self.apply_gamma_correction(clahe_image, params["gamma"])
            else:
                gamma_image = clahe_image

            blurred_image = self.apply_gaussian_blur(gamma_image)

            # Apply thresholding method
            if params["threshold_method"] == "otsu":
                threshold_image = self.apply_otsu_threshold(blurred_image)
            else:  # adaptive
                threshold_image = self.apply_adaptive_threshold(blurred_image)

            morph_image = self.apply_morphological_closing(threshold_image)

            # Apply sharpening if enabled
            if params["sharpening"]:
                final_image = self.apply_sharpening_filter(morph_image)
            else:
                final_image = morph_image

            detections = self.extract_text_with_ocr(final_image)

            return {
                "filename": filename,
                "parameters": params,
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
                "parameters": params,
                "error": str(e),
                "success": False,
            }

    def run_random_parameter_test(self, input_dir, num_combinations=50):
        """Run test with random parameter combinations"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_dir}")
            return []

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {input_dir}")
            return []

        # Use first 5 images to keep test manageable
        test_images = image_files[:5]

        print(f"🔬 COMPREHENSIVE RANDOM PARAMETER TEST")
        print("=" * 80)
        print(f"Testing {num_combinations} random parameter combinations")
        print(f"Using {len(test_images)} test images")
        print(f"Variable parameters:")
        print(f"  - Gamma values: {self.gamma_values}")
        print(f"  - Scale sizes: {self.scale_sizes}")
        print(f"  - Threshold methods: {self.threshold_methods}")
        print(f"  - Sharpening: {self.sharpening_options}")
        print(f"  - CLAHE clip sizes: {self.clahe_clip_sizes}")
        print(f"  - CLAHE grid sizes: {self.clahe_grid_sizes}")
        print("=" * 80)

        all_results = []

        for combo_num in range(1, num_combinations + 1):
            # Generate random parameters
            params = self.generate_random_parameters()

            print(f"\n🎲 Combination {combo_num}/{num_combinations}")
            print(
                f"Parameters: γ={params['gamma']}, scale={params['scale']}, "
                f"thresh={params['threshold_method']}, sharp={params['sharpening']}, "
                f"clip={params['clahe_clip']}, grid={params['clahe_grid']}"
            )
            print("-" * 60)

            combo_results = []
            total_detections = 0
            total_confidence = 0.0
            successful_images = 0

            for img_num, image_file in enumerate(test_images, 1):
                start_time = time.time()
                result = self.process_image_with_parameters(str(image_file), params)
                process_time = time.time() - start_time

                if result and result["success"]:
                    result["process_time"] = process_time
                    combo_results.append(result)
                    total_detections += result["detection_count"]
                    total_confidence += result["avg_confidence"]
                    successful_images += 1

                    print(
                        f"  {img_num}. {result['filename'][:30]:<30} "
                        f"{result['detection_count']:2d} det | "
                        f"{result['avg_confidence']:.3f} conf | "
                        f"{process_time:.2f}s"
                    )
                else:
                    print(f"  {img_num}. {Path(image_file).stem[:30]:<30} ERROR")

            # Calculate combination summary
            if successful_images > 0:
                avg_detections = total_detections / successful_images
                avg_confidence = total_confidence / successful_images

                print(f"  📊 Summary: {avg_detections:.1f} avg det | {avg_confidence:.3f} avg conf")

                combo_summary = {
                    "combination_number": combo_num,
                    "parameters": params,
                    "avg_detections": avg_detections,
                    "avg_confidence": avg_confidence,
                    "successful_images": successful_images,
                    "individual_results": combo_results,
                }
                all_results.append(combo_summary)
            else:
                print(f"  ❌ All images failed")

        return all_results

    def print_final_summary(self, results):
        """Print comprehensive summary of all combinations"""
        if not results:
            print("No results to summarize")
            return

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE RANDOM PARAMETER TEST SUMMARY")
        print(f"{'='*80}")

        # Sort by average confidence
        sorted_by_confidence = sorted(results, key=lambda x: x["avg_confidence"], reverse=True)
        sorted_by_detections = sorted(results, key=lambda x: x["avg_detections"], reverse=True)

        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total combinations tested: {len(results)}")

        all_confidences = [r["avg_confidence"] for r in results]
        all_detections = [r["avg_detections"] for r in results]

        print(f"   Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
        print(f"   Detection range: {min(all_detections):.1f} - {max(all_detections):.1f}")
        print(f"   Average confidence: {np.mean(all_confidences):.3f}")
        print(f"   Average detections: {np.mean(all_detections):.1f}")

        print(f"\n🏆 TOP 10 BY CONFIDENCE:")
        print(
            f"{'Rank':<4} {'Conf':<6} {'Det':<5} {'γ':<4} {'Scale':<5} {'Thresh':<8} {'Sharp':<5} {'Clip':<4} {'Grid':<6}"
        )
        print("-" * 70)
        for i, result in enumerate(sorted_by_confidence[:10], 1):
            p = result["parameters"]
            print(
                f"{i:<4} {result['avg_confidence']:<6.3f} {result['avg_detections']:<5.1f} "
                f"{p['gamma']:<4} {p['scale']:<5} {p['threshold_method']:<8} "
                f"{str(p['sharpening']):<5} {p['clahe_clip']:<4} {str(p['clahe_grid']):<6}"
            )

        print(f"\n🎯 TOP 10 BY DETECTIONS:")
        print(
            f"{'Rank':<4} {'Det':<5} {'Conf':<6} {'γ':<4} {'Scale':<5} {'Thresh':<8} {'Sharp':<5} {'Clip':<4} {'Grid':<6}"
        )
        print("-" * 70)
        for i, result in enumerate(sorted_by_detections[:10], 1):
            p = result["parameters"]
            print(
                f"{i:<4} {result['avg_detections']:<5.1f} {result['avg_confidence']:<6.3f} "
                f"{p['gamma']:<4} {p['scale']:<5} {p['threshold_method']:<8} "
                f"{str(p['sharpening']):<5} {p['clahe_clip']:<4} {str(p['clahe_grid']):<6}"
            )

        # Parameter analysis
        print(f"\n📈 PARAMETER ANALYSIS:")

        # Gamma analysis
        gamma_results = {}
        for result in results:
            gamma = result["parameters"]["gamma"]
            if gamma not in gamma_results:
                gamma_results[gamma] = []
            gamma_results[gamma].append(result["avg_confidence"])

        print(f"   Gamma performance:")
        for gamma in sorted(gamma_results.keys()):
            avg_conf = np.mean(gamma_results[gamma])
            print(
                f"     γ={gamma}: {avg_conf:.3f} avg confidence ({len(gamma_results[gamma])} samples)"
            )

        # Scale analysis
        scale_results = {}
        for result in results:
            scale = result["parameters"]["scale"]
            if scale not in scale_results:
                scale_results[scale] = []
            scale_results[scale].append(result["avg_confidence"])

        print(f"   Scale performance:")
        for scale in sorted(scale_results.keys()):
            avg_conf = np.mean(scale_results[scale])
            print(
                f"     Scale={scale}: {avg_conf:.3f} avg confidence ({len(scale_results[scale])} samples)"
            )

        # Threshold method analysis
        thresh_results = {}
        for result in results:
            thresh = result["parameters"]["threshold_method"]
            if thresh not in thresh_results:
                thresh_results[thresh] = []
            thresh_results[thresh].append(result["avg_confidence"])

        print(f"   Threshold method performance:")
        for thresh in thresh_results.keys():
            avg_conf = np.mean(thresh_results[thresh])
            print(
                f"     {thresh}: {avg_conf:.3f} avg confidence ({len(thresh_results[thresh])} samples)"
            )

        # Save detailed results
        with open("comprehensive_parameter_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📁 Detailed results saved to: comprehensive_parameter_test_results.json")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    tester = ComprehensiveParameterTest()

    print("Comprehensive Random Parameter Combination Test")
    print("=" * 60)
    print("Testing random combinations of:")
    print("  - Gamma values: 0, 0.5, 0.8, 1.2, 1.5")
    print("  - Scale sizes: 1.5, 2, 2.5")
    print("  - Thresholding: Otsu vs Adaptive")
    print("  - Sharpening: On vs Off")
    print("  - CLAHE clip sizes: 1, 2, 3")
    print("  - CLAHE grid sizes: 4x4 vs 8x8")
    print("=" * 60)

    # Run test with 50 random combinations
    results = tester.run_random_parameter_test("preprocessing_test_samples", num_combinations=50)

    # Print summary
    tester.print_final_summary(results)


if __name__ == "__main__":
    main()

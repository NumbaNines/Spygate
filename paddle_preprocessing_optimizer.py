#!/usr/bin/env python3
"""
PaddleOCR Preprocessing Optimizer
Systematically test and fine-tune each preprocessing step for optimal Madden HUD OCR
"""

import itertools
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR


class PreprocessingOptimizer:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "baseline_results": {},
            "technique_results": {},
            "combination_results": {},
            "optimal_pipeline": None,
        }

    def extract_text_with_confidence(self, image):
        """Extract text using PaddleOCR and return structured results"""
        try:
            result = self.ocr.ocr(image, cls=False)

            if not result or not result[0]:
                return {
                    "success": False,
                    "text_count": 0,
                    "texts": [],
                    "avg_confidence": 0.0,
                    "total_chars": 0,
                }

            texts = []
            confidences = []
            total_chars = 0

            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    texts.append(text)
                    confidences.append(confidence)
                    total_chars += len(text)

            return {
                "success": True,
                "text_count": len(texts),
                "texts": texts,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "total_chars": total_chars,
                "all_confidences": confidences,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text_count": 0,
                "texts": [],
                "avg_confidence": 0.0,
                "total_chars": 0,
            }

    def apply_contrast_enhancement(self, image, alpha=1.5, beta=30):
        """Apply contrast and brightness adjustment"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def apply_gaussian_blur(self, image, kernel_size=3):
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_sharpening(self, image, strength=1.0):
        """Apply unsharp masking for sharpening"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

    def apply_upscaling(self, image, scale_factor=2.0, interpolation=cv2.INTER_CUBIC):
        """Apply image upscaling"""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    def apply_gamma_correction(self, image, gamma=1.2):
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
            "uint8"
        )
        return cv2.LUT(image, table)

    def test_baseline(self, test_images):
        """Test baseline performance without preprocessing"""
        print("🔍 Testing baseline performance (no preprocessing)...")

        baseline_results = []

        for i, image_path in enumerate(test_images):
            print(f"  📸 Processing {i+1}/{len(test_images)}: {image_path.name}")

            # Load original image
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            # Extract text without preprocessing
            result = self.extract_text_with_confidence(image)
            result["image_name"] = image_path.name
            baseline_results.append(result)

        self.results["baseline_results"] = baseline_results

        # Calculate baseline metrics
        successful = [r for r in baseline_results if r["success"]]
        avg_confidence = np.mean([r["avg_confidence"] for r in successful]) if successful else 0.0
        avg_text_count = np.mean([r["text_count"] for r in successful]) if successful else 0.0

        print(
            f"✅ Baseline: {len(successful)}/{len(baseline_results)} success, "
            f"avg confidence: {avg_confidence:.3f}, avg texts: {avg_text_count:.1f}"
        )

        return baseline_results

    def test_individual_techniques(self, test_images):
        """Test each preprocessing technique individually"""
        print("\n🔧 Testing individual preprocessing techniques...")

        techniques = {
            "contrast_1.2_20": lambda img: self.apply_contrast_enhancement(img, 1.2, 20),
            "contrast_1.5_30": lambda img: self.apply_contrast_enhancement(img, 1.5, 30),
            "contrast_2.0_40": lambda img: self.apply_contrast_enhancement(img, 2.0, 40),
            "gaussian_blur_3": lambda img: self.apply_gaussian_blur(img, 3),
            "gaussian_blur_5": lambda img: self.apply_gaussian_blur(img, 5),
            "sharpening_0.5": lambda img: self.apply_sharpening(img, 0.5),
            "sharpening_1.0": lambda img: self.apply_sharpening(img, 1.0),
            "sharpening_1.5": lambda img: self.apply_sharpening(img, 1.5),
            "upscale_2x_cubic": lambda img: self.apply_upscaling(img, 2.0, cv2.INTER_CUBIC),
            "upscale_3x_cubic": lambda img: self.apply_upscaling(img, 3.0, cv2.INTER_CUBIC),
            "upscale_2x_lanczos": lambda img: self.apply_upscaling(img, 2.0, cv2.INTER_LANCZOS4),
            "gamma_0.8": lambda img: self.apply_gamma_correction(img, 0.8),
            "gamma_1.2": lambda img: self.apply_gamma_correction(img, 1.2),
            "gamma_1.5": lambda img: self.apply_gamma_correction(img, 1.5),
        }

        technique_results = {}

        for technique_name, technique_func in techniques.items():
            print(f"  🔧 Testing {technique_name}...")

            technique_results[technique_name] = []

            for image_path in test_images:
                # Load original image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                try:
                    # Apply preprocessing technique
                    processed_image = technique_func(image)

                    # Extract text
                    result = self.extract_text_with_confidence(processed_image)
                    result["image_name"] = image_path.name
                    technique_results[technique_name].append(result)

                except Exception as e:
                    print(f"    ❌ Error with {technique_name} on {image_path.name}: {e}")
                    continue

            # Calculate metrics for this technique
            successful = [r for r in technique_results[technique_name] if r["success"]]
            if successful:
                avg_confidence = np.mean([r["avg_confidence"] for r in successful])
                avg_text_count = np.mean([r["text_count"] for r in successful])
                success_rate = len(successful) / len(technique_results[technique_name])

                print(
                    f"    ✅ {technique_name}: {success_rate:.1%} success, "
                    f"confidence: {avg_confidence:.3f}, texts: {avg_text_count:.1f}"
                )
            else:
                print(f"    ❌ {technique_name}: No successful extractions")

        self.results["technique_results"] = technique_results
        return technique_results

    def save_results(self, output_path):
        """Save all results to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")


def main():
    print("🚀 PADDLEOCR PREPROCESSING OPTIMIZER")
    print("🎯 Systematically testing each preprocessing technique")
    print("=" * 70)

    # Get test images (sample from 6.12 screenshots)
    screenshots_folder = Path("6.12 screenshots")
    if not screenshots_folder.exists():
        print("❌ 6.12 screenshots folder not found!")
        return

    # Use a representative sample for testing
    all_images = list(screenshots_folder.glob("*.png"))
    test_images = all_images[::10]  # Every 10th image for faster testing

    if len(test_images) < 5:
        test_images = all_images[:10]  # Use first 10 if sample is too small

    print(f"📸 Using {len(test_images)} test images from {len(all_images)} total")

    # Initialize optimizer
    optimizer = PreprocessingOptimizer()

    # Test baseline
    baseline_results = optimizer.test_baseline(test_images)

    # Test individual techniques
    technique_results = optimizer.test_individual_techniques(test_images)

    # Save results
    output_path = (
        f"paddle_preprocessing_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    optimizer.save_results(output_path)

    print(f"\n✅ Preprocessing optimization complete!")
    print(f"📊 Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

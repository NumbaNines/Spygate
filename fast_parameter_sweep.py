#!/usr/bin/env python3
"""
FAST Parameter Sweep - Optimized for speed
- Only 3 best test images (not 10)
- 1000 combinations (not 3000)
- Faster scoring method
- Same parameter ranges and tracking
"""

import glob
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class FastParameterSweep:
    """Fast parameter sweep optimized for speed"""

    def __init__(self):
        print("🚀 FAST Parameter Sweep - Optimized for Speed")
        print("🎯 3 best images + 1000 combos + faster scoring")
        print("🔧 CPU-only processing")
        print("=" * 60)

        # Initialize PaddleOCR CPU-only
        print("Initializing PaddleOCR CPU-only...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        print("✅ PaddleOCR CPU initialized successfully!")

        # IMPROVED Parameter ranges (same as before)
        self.gamma_values = ["off", 0.8, 1.0, 1.2, 1.4]
        self.sharpening_values = [True, False]
        self.scale_values = [2.0, 2.5, 3.0, 3.5, 4.0]
        self.threshold_values = ["otsu", "adaptive_mean", "adaptive_gaussian"]
        self.clahe_clip_values = [1.0, 2.0, 3.0]
        self.clahe_grid_values = [(4, 4), (8, 8)]
        self.blur_values = ["off", (3, 3), (5, 5)]

        # Results tracking
        self.results_file = f"fast_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_combinations": 1000,
            "completed_combinations": 0,
            "best_score": 0.0,
            "best_combination": None,
            "all_results": [],
        }

        # Combination tracking
        self.used_combinations_file = (
            f"used_combinations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.used_combinations = set()
        self.load_used_combinations()

        # Only 3 BEST test images for speed
        self.test_images = self.get_best_test_images()

        print(
            f"📊 Parameter space: {len(self.gamma_values)} × {len(self.sharpening_values)} × {len(self.scale_values)} × {len(self.threshold_values)} × {len(self.clahe_clip_values)} × {len(self.clahe_grid_values)} × {len(self.blur_values)}"
        )
        print(f"📊 Total possible combinations: {self.calculate_total_combinations()}")
        print(f"📊 Previously used combinations: {len(self.used_combinations)}")
        print(f"📊 Test images loaded: {len(self.test_images)} (optimized for speed)")

    def calculate_total_combinations(self):
        """Calculate total possible combinations"""
        return (
            len(self.gamma_values)
            * len(self.sharpening_values)
            * len(self.scale_values)
            * len(self.threshold_values)
            * len(self.clahe_clip_values)
            * len(self.clahe_grid_values)
            * len(self.blur_values)
        )

    def load_used_combinations(self):
        """Load previously used combinations from all files"""
        used_files = glob.glob("used_combinations_*.json")

        for file in used_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    for combo in data.get("used_combinations", []):
                        self.used_combinations.add(tuple(combo))
                print(
                    f"✅ Loaded {len(data.get('used_combinations', []))} used combinations from {file}"
                )
            except:
                continue

        print(f"📊 Total previously used combinations: {len(self.used_combinations)}")

    def get_best_test_images(self):
        """Get only the 3 BEST test images for maximum speed"""
        # Only use the most reliable debug images that we know work well
        best_images = [
            "found_and_frame_3000.png",  # Known good down/distance
            "debug_down_distance_region.png",  # Known working debug region
            "extracted_frame.jpg",  # Known territory area
        ]

        test_images = []
        for img_path in best_images:
            if os.path.exists(img_path):
                test_images.append(Path(img_path))
                print(f"✅ Using: {img_path}")

        # If we don't have the expected files, find any available images
        if len(test_images) < 3:
            print("⚠️ Some expected images missing, finding alternatives...")
            additional_paths = [".", "debug_regions/", "preprocessing_test_samples/"]

            for path in additional_paths:
                if os.path.exists(path):
                    for ext in ["*.jpg", "*.png", "*.jpeg"]:
                        images = list(Path(path).glob(ext))
                        for img in images:
                            if img not in test_images and len(test_images) < 3:
                                test_images.append(img)
                                print(f"✅ Using: {img}")
                if len(test_images) >= 3:
                    break

        return test_images[:3]  # Maximum 3 images for speed

    def generate_random_combinations(self, num_combinations):
        """Generate random untested combinations"""
        combinations = []
        max_attempts = num_combinations * 5  # Reduce attempts for speed
        attempts = 0

        print(f"🎲 Generating {num_combinations} NEW untested combinations...")

        while len(combinations) < num_combinations and attempts < max_attempts:
            combo = [
                random.choice(self.gamma_values),
                random.choice(self.sharpening_values),
                random.choice(self.scale_values),
                random.choice(self.threshold_values),
                random.choice(self.clahe_clip_values),
                random.choice(self.clahe_grid_values),
                random.choice(self.blur_values),
            ]

            combo_tuple = tuple(combo)

            if combo_tuple not in self.used_combinations:
                combinations.append(combo)
                self.used_combinations.add(combo_tuple)

            attempts += 1

        print(f"✅ Generated {len(combinations)} NEW untested combinations")
        print(f"⚠️ Skipped {attempts - len(combinations)} previously tested combinations")

        return combinations

    def apply_preprocessing_pipeline(
        self, image, gamma, sharpening, scale, threshold, clahe_clip, clahe_grid, blur
    ):
        """Same preprocessing pipeline as before (keeping it identical)"""

        # Stage 1: Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Stage 2: Scale with LANCZOS4
        height, width = gray.shape
        new_height, new_width = int(height * scale), int(width * scale)
        scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Stage 3: CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        clahe_applied = clahe.apply(scaled)

        # Stage 4: Gamma correction
        if gamma != "off":
            gamma_corrected = np.power(clahe_applied / 255.0, gamma) * 255.0
            gamma_corrected = gamma_corrected.astype(np.uint8)
        else:
            gamma_corrected = clahe_applied

        # Stage 5: Gaussian blur
        if blur != "off":
            blurred = cv2.GaussianBlur(gamma_corrected, blur, 0)
        else:
            blurred = gamma_corrected

        # Stage 6: Thresholding
        if threshold == "otsu":
            _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold == "adaptive_mean":
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5
            )
        elif threshold == "adaptive_gaussian":
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
            )

        # Stage 7: Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Stage 8: Sharpening
        if sharpening:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(morphed, -1, kernel)
            final = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            final = morphed

        return final

    def test_combination_on_images(self, combination):
        """Test combination on only 3 images for speed"""
        gamma, sharpening, scale, threshold, clahe_clip, clahe_grid, blur = combination

        total_confidence = 0.0
        total_detections = 0
        valid_tests = 0

        for img_path in self.test_images:
            try:
                if not img_path.exists():
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Apply preprocessing
                processed = self.apply_preprocessing_pipeline(
                    image, gamma, sharpening, scale, threshold, clahe_clip, clahe_grid, blur
                )

                # Run OCR
                results = self.ocr.ocr(processed, cls=True)

                if results and results[0]:
                    for line in results[0]:
                        confidence = line[1][1]
                        if confidence >= 0.1:
                            total_confidence += confidence
                            total_detections += 1

                valid_tests += 1

            except Exception as e:
                continue

        if valid_tests == 0:
            return 0.0

        # Simple scoring: 80% confidence + 20% detection count
        avg_confidence = total_confidence / max(total_detections, 1)
        normalized_detections = min(total_detections / (valid_tests * 2), 1.0)

        composite_score = (0.8 * avg_confidence) + (0.2 * normalized_detections)

        return composite_score

    def save_results(self):
        """Save current results to file"""
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def save_used_combinations(self):
        """Save all used combinations to file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_used": len(self.used_combinations),
            "used_combinations": list(self.used_combinations),
        }

        with open(self.used_combinations_file, "w") as f:
            json.dump(data, f, indent=2)

    def run_sweep(self, num_combinations=1000):
        """Run the FAST parameter sweep"""
        print(f"\n🎯 Starting FAST {num_combinations}-Combination Parameter Sweep")
        print(f"📊 Testing on {len(self.test_images)} images (optimized)")
        print("=" * 60)

        combinations = self.generate_random_combinations(num_combinations)

        if len(combinations) == 0:
            print("⚠️ No new combinations to test!")
            return

        start_time = time.time()

        for i, combination in enumerate(combinations):
            score = self.test_combination_on_images(combination)

            result = {
                "combination_id": i + 1,
                "parameters": {
                    "gamma": combination[0],
                    "sharpening": combination[1],
                    "scale": combination[2],
                    "threshold": combination[3],
                    "clahe_clip": combination[4],
                    "clahe_grid": combination[5],
                    "blur": combination[6],
                },
                "score": round(score, 3),
                "timestamp": datetime.now().isoformat(),
            }

            self.results["all_results"].append(result)
            self.results["completed_combinations"] = i + 1

            # Check for new best
            if score > self.results["best_score"]:
                self.results["best_score"] = round(score, 3)
                self.results["best_combination"] = result["parameters"]
                print(f"🏆 NEW WINNER! Combo {i+1}/1000 - Score: {score:.3f}")
                print(
                    f"   γ={combination[0]}, scale={combination[2]}x, {combination[3]}, clahe={combination[4]}/{combination[5]}, blur={combination[6]}, sharp={combination[1]}"
                )

            # Progress update every 25 combinations (more frequent)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta_minutes = (len(combinations) - i - 1) / (rate / 60)

                print(
                    f"⏱️ Progress: {i+1}/1000 ({(i+1)/10:.1f}%) | Rate: {rate:.1f}/min | ETA: {eta_minutes:.1f}min | Best: {self.results['best_score']:.3f}"
                )

                self.save_results()
                self.save_used_combinations()

        # Final save
        self.save_results()
        self.save_used_combinations()

        total_time = (time.time() - start_time) / 60
        print(f"\n🎉 FAST 1000-Combination Sweep Complete!")
        print(f"⏱️ Total time: {total_time:.1f} minutes")
        print(f"🏆 Best score: {self.results['best_score']:.3f}")
        print(f"📊 Best combination: {self.results['best_combination']}")


def main():
    """Main function"""
    random.seed(42)  # Reproducible results

    sweeper = FastParameterSweep()
    sweeper.run_sweep(1000)  # Only 1000 combinations for speed


if __name__ == "__main__":
    main()

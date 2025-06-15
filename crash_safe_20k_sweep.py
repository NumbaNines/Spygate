#!/usr/bin/env python3
"""
Crash-Safe 20,000 Combination Parameter Sweep
- Saves best combo after EVERY test
- Can resume from interruptions
- Continuous progress tracking
- Expert improvements included
"""

import glob
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class CrashSafe20KParameterSweep:
    """Crash-safe 20K parameter sweep with continuous saving"""

    def __init__(self):
        print("🚀 CRASH-SAFE 20,000 Combination Parameter Sweep")
        print("💾 Saves best combo after EVERY test")
        print("🔄 Can resume from interruptions")
        print("🎯 15.4% coverage of 129,600 total space")
        print("=" * 70)

        # Initialize PaddleOCR CPU-only
        print("Initializing PaddleOCR CPU-only...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        print("✅ PaddleOCR CPU initialized successfully!")

        # ENHANCED Parameter ranges (Original + Expert Improvements)
        self.gamma_values = ["off", 0.8, 1.0, 1.2, 1.4]
        self.sharpening_values = [True, False]
        self.scale_values = [2.0, 2.5, 3.0, 3.5, 4.0]
        self.threshold_values = ["otsu", "adaptive_mean", "adaptive_gaussian"]
        self.clahe_clip_values = [1.0, 2.0, 3.0]
        self.clahe_grid_values = [(4, 4), (8, 8)]
        self.blur_values = ["off", (3, 3), (5, 5)]

        # NEW Expert Improvements
        self.adaptive_block_sizes = [9, 11, 13, 15]
        self.adaptive_c_constants = [3, 5, 7, 9]
        self.morph_kernel_sizes = [(1, 1), (2, 2), (3, 3)]

        # Load existing test regions
        self.test_regions = self.load_test_regions()

        # CRASH-SAFE Results files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"crash_safe_20k_results_{self.timestamp}.json"
        self.best_combo_file = f"BEST_COMBO_{self.timestamp}.json"
        self.progress_file = f"progress_20k_{self.timestamp}.json"

        # Initialize or load existing results
        self.results = self.load_or_create_results()
        self.used_combinations = set()
        self.load_used_combinations()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print(f"📊 Loaded {len(self.test_regions)} test regions")
        print(f"📊 Parameter space: {self.calculate_total_combinations()} total combinations")
        print(f"📊 Previously used combinations: {len(self.used_combinations)}")
        print(f"📊 Current best score: {self.results['best_score']}")

    def signal_handler(self, signum, frame):
        """Handle interruption gracefully"""
        print(f"\n🛑 Received signal {signum}. Saving final results...")
        self.save_all_results()
        print(f"💾 Results saved to: {self.results_file}")
        print(f"🏆 Best combo saved to: {self.best_combo_file}")
        sys.exit(0)

    def load_or_create_results(self):
        """Load existing results or create new structure"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, "r") as f:
                    results = json.load(f)
                print(
                    f"📂 Resuming from existing results: {len(results.get('all_results', []))} completed"
                )
                return results
            except:
                pass

        return {
            "timestamp": datetime.now().isoformat(),
            "total_combinations": 20000,
            "completed_combinations": 0,
            "best_score": 0.0,
            "best_combination": None,
            "all_results": [],
            "crash_safe": True,
        }

    def load_test_regions(self):
        """Load existing test regions from debug files or create simple ones"""
        regions = []

        # Try to load from debug regions
        debug_files = [
            "debug_regions/down_distance_area_region.png",
            "debug_regions/territory_triangle_area_region.png",
            "debug_regions/game_clock_area_region.png",
            "debug_regions/play_clock_area_region.png",
        ]

        for file_path in debug_files:
            if os.path.exists(file_path):
                region = cv2.imread(file_path)
                if region is not None:
                    regions.append({"roi": region, "source": os.path.basename(file_path)})
                    print(
                        f"✅ Loaded debug region: {file_path} ({region.shape[1]}x{region.shape[0]})"
                    )

        # If no debug files, create synthetic test regions
        if len(regions) == 0:
            print("🔧 Creating synthetic test regions...")

            # Create simple text-like patterns
            base_image = np.ones((30, 120, 3), dtype=np.uint8) * 255
            cv2.putText(
                base_image, "1ST & 10", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            regions.append({"roi": base_image, "source": "synthetic_text"})

            base_image2 = np.ones((25, 80, 3), dtype=np.uint8) * 255
            cv2.putText(base_image2, "A35", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            regions.append({"roi": base_image2, "source": "synthetic_territory"})

            print("✅ Created 2 synthetic test regions")

        return regions

    def calculate_total_combinations(self):
        """Calculate total possible combinations including expert improvements"""
        return (
            len(self.gamma_values)
            * len(self.sharpening_values)
            * len(self.scale_values)
            * len(self.threshold_values)
            * len(self.clahe_clip_values)
            * len(self.clahe_grid_values)
            * len(self.blur_values)
            * len(self.adaptive_block_sizes)
            * len(self.adaptive_c_constants)
            * len(self.morph_kernel_sizes)
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
            except:
                continue

        # Also load from current results
        for result in self.results.get("all_results", []):
            params = result["parameters"]
            combo = [
                params["gamma"],
                params["sharpening"],
                params["scale"],
                params["threshold"],
                params["clahe_clip"],
                params["clahe_grid"],
                params["blur"],
                params["adaptive_block_size"],
                params["adaptive_c_constant"],
                params["morph_kernel_size"],
            ]
            self.used_combinations.add(tuple(combo))

    def apply_enhanced_preprocessing_pipeline(self, image, params):
        """Apply the EXACT 8-stage pipeline with enhanced parameters"""
        (
            gamma,
            sharpening,
            scale,
            threshold,
            clahe_clip,
            clahe_grid,
            blur,
            block_size,
            c_constant,
            morph_kernel,
        ) = params

        # Stage 1: Convert to grayscale (ALWAYS FIRST)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Stage 2: Scale with LANCZOS4 (ALWAYS SECOND)
        height, width = gray.shape
        new_height, new_width = int(height * scale), int(width * scale)
        scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Stage 3: CLAHE (ALWAYS THIRD)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        clahe_applied = clahe.apply(scaled)

        # Stage 4: Gamma correction (CONDITIONAL)
        if gamma != "off":
            gamma_corrected = np.power(clahe_applied / 255.0, gamma) * 255.0
            gamma_corrected = gamma_corrected.astype(np.uint8)
        else:
            gamma_corrected = clahe_applied

        # Stage 5: Gaussian blur (CONDITIONAL)
        if blur != "off":
            blurred = cv2.GaussianBlur(gamma_corrected, blur, 0)
        else:
            blurred = gamma_corrected

        # Stage 6: Thresholding (ALWAYS APPLIED) - NOW WITH EXPERT IMPROVEMENTS
        if threshold == "otsu":
            _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold == "adaptive_mean":
            thresholded = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_constant
            )
        elif threshold == "adaptive_gaussian":
            thresholded = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_constant,
            )

        # Stage 7: Morphological closing (ALWAYS APPLIED) - NOW WITH EXPERT IMPROVEMENTS
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Stage 8: Sharpening (CONDITIONAL - ALWAYS LAST)
        if sharpening:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            sharpened = cv2.filter2D(morphed, -1, kernel)
            final = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            final = morphed

        return final

    def generate_enhanced_combinations(self, num_combinations):
        """Generate random combinations including expert improvements"""
        combinations = []
        max_attempts = num_combinations * 10
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
                random.choice(self.adaptive_block_sizes),
                random.choice(self.adaptive_c_constants),
                random.choice(self.morph_kernel_sizes),
            ]

            combo_tuple = tuple(combo)

            if combo_tuple not in self.used_combinations:
                combinations.append(combo)
                self.used_combinations.add(combo_tuple)

            attempts += 1

        print(f"✅ Generated {len(combinations)} NEW untested combinations")
        print(f"⚠️ Skipped {attempts - len(combinations)} previously tested combinations")

        return combinations

    def test_combination_on_regions(self, combination):
        """Test preprocessing combination on all test regions"""
        total_confidence = 0.0
        total_detections = 0
        valid_tests = 0

        for region_data in self.test_regions:
            try:
                roi = region_data["roi"]

                # Apply enhanced preprocessing
                processed = self.apply_enhanced_preprocessing_pipeline(roi, combination)

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

        # EXPERT IMPROVEMENT: Better scoring (80% confidence + 20% detection)
        avg_confidence = total_confidence / max(total_detections, 1)
        normalized_detections = min(total_detections / (valid_tests * 2), 1.0)

        composite_score = (0.8 * avg_confidence) + (0.2 * normalized_detections)

        return composite_score

    def save_best_combo_immediately(self):
        """Save best combination immediately after each test"""
        best_data = {
            "timestamp": datetime.now().isoformat(),
            "best_score": self.results["best_score"],
            "best_combination": self.results["best_combination"],
            "completed_combinations": self.results["completed_combinations"],
            "total_combinations": self.results["total_combinations"],
            "coverage_percentage": (self.results["completed_combinations"] / 129600) * 100,
        }

        with open(self.best_combo_file, "w") as f:
            json.dump(best_data, f, indent=2)

    def save_all_results(self):
        """Save all results to files"""
        # Save main results
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save best combo
        self.save_best_combo_immediately()

        # Save used combinations
        used_data = {
            "timestamp": datetime.now().isoformat(),
            "total_used": len(self.used_combinations),
            "used_combinations": list(self.used_combinations),
        }

        with open(f"used_combinations_{self.timestamp}.json", "w") as f:
            json.dump(used_data, f, indent=2)

    def run_sweep(self, num_combinations=20000):
        """Run the crash-safe 20,000-combination parameter sweep"""
        print(f"\n🎯 Starting Crash-Safe {num_combinations}-Combination Sweep")
        print(f"📊 Testing on {len(self.test_regions)} test regions")
        print(f"💾 Saving best combo after EVERY test")
        print("=" * 70)

        if len(self.test_regions) == 0:
            print("❌ No test regions loaded! Cannot run sweep.")
            return

        # Calculate how many new combinations we need
        already_completed = self.results["completed_combinations"]
        remaining = num_combinations - already_completed

        if remaining <= 0:
            print(
                f"✅ Sweep already complete! {already_completed}/{num_combinations} combinations tested."
            )
            return

        print(f"🔄 Resuming from combination {already_completed + 1}")

        combinations = self.generate_enhanced_combinations(remaining)

        if len(combinations) == 0:
            print("⚠️ No new combinations to test!")
            return

        start_time = time.time()

        for i, combination in enumerate(combinations):
            current_combo_num = already_completed + i + 1

            score = self.test_combination_on_regions(combination)

            result = {
                "combination_id": current_combo_num,
                "parameters": {
                    "gamma": combination[0],
                    "sharpening": combination[1],
                    "scale": combination[2],
                    "threshold": combination[3],
                    "clahe_clip": combination[4],
                    "clahe_grid": combination[5],
                    "blur": combination[6],
                    "adaptive_block_size": combination[7],
                    "adaptive_c_constant": combination[8],
                    "morph_kernel_size": combination[9],
                },
                "score": round(score, 3),
                "timestamp": datetime.now().isoformat(),
            }

            self.results["all_results"].append(result)
            self.results["completed_combinations"] = current_combo_num

            # Check for new best
            if score > self.results["best_score"]:
                self.results["best_score"] = round(score, 3)
                self.results["best_combination"] = result["parameters"]
                print(f"🏆 NEW WINNER! Combo {current_combo_num}/20000 - Score: {score:.3f}")
                print(
                    f"   γ={combination[0]}, scale={combination[2]}x, {combination[3]}, clahe={combination[4]}/{combination[5]}"
                )
                print(
                    f"   blur={combination[6]}, sharp={combination[1]}, adaptive={combination[7]}/{combination[8]}, morph={combination[9]}"
                )

            # CRASH-SAFE: Save best combo after EVERY test
            self.save_best_combo_immediately()

            # Progress update every 100 combinations
            if current_combo_num % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta_minutes = (len(combinations) - i - 1) / (rate / 60)
                coverage = (current_combo_num / 129600) * 100

                print(
                    f"⏱️ Progress: {current_combo_num}/20000 ({current_combo_num/200:.1f}%) | Coverage: {coverage:.2f}% | Rate: {rate:.1f}/min | ETA: {eta_minutes:.1f}min | Best: {self.results['best_score']:.3f}"
                )

                # Save full results every 100 combinations
                self.save_all_results()

        # Final save
        self.save_all_results()

        total_time = (time.time() - start_time) / 60
        final_coverage = (self.results["completed_combinations"] / 129600) * 100

        print(f"\n🎉 Crash-Safe 20,000-Combination Sweep Complete!")
        print(f"⏱️ Total time: {total_time:.1f} minutes")
        print(f"🏆 Best score: {self.results['best_score']:.3f}")
        print(f"📊 Best combination: {self.results['best_combination']}")
        print(f"📈 Parameter space coverage: {final_coverage:.2f}%")
        print(f"💾 Results saved to: {self.results_file}")
        print(f"🏆 Best combo saved to: {self.best_combo_file}")


def main():
    """Main function"""
    random.seed(42)  # Reproducible results

    sweeper = CrashSafe20KParameterSweep()
    sweeper.run_sweep(20000)


if __name__ == "__main__":
    main()

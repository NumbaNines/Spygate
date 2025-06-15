#!/usr/bin/env python3
"""
Test Preprocessing Parameter Combinations on Debug Regions
Tests all parameter combinations on specific debug regions (down/distance and territory)
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PreprocessingDebugRegionTester:
    """Test preprocessing parameter combinations on specific debug regions"""

    def __init__(self):
        print("🔍 PREPROCESSING Parameter Testing on Debug Regions")
        print("🎯 Testing combinations on down/distance and territory areas")
        print("=" * 70)

        # Initialize PaddleOCR CPU-only
        print("Initializing PaddleOCR CPU-only...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        print("✅ PaddleOCR CPU initialized successfully!")

        # Parameter combinations to test
        self.gamma_values = ["off", 0.5, 0.8, 1.2, 1.4]
        self.sharpening_values = [True, False]  # on/off
        self.scale_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]  # EXPANDED!
        self.threshold_values = ["otsu", "adaptive_mean", "adaptive_gaussian"]
        self.clahe_clip_values = [1.0, 2.0, 3.0]
        self.clahe_grid_values = [(4, 4), (8, 8)]
        self.blur_values = ["off", (3, 3), (5, 5)]

        # Calculate total combinations
        total_combos = (
            len(self.gamma_values)
            * len(self.sharpening_values)
            * len(self.scale_values)
            * len(self.threshold_values)
            * len(self.clahe_clip_values)
            * len(self.clahe_grid_values)
            * len(self.blur_values)
        )

        print(f"📊 Total possible combinations: {total_combos:,}")

        # Extract debug regions
        self.debug_regions = self.extract_debug_regions()

        # Results tracking
        self.results = []
        self.best_score = 0.0
        self.best_combo = None
        self.start_time = time.time()

    def extract_debug_regions(self):
        """Extract the specific debug regions we want to test"""
        regions = {}

        # 1. Down/Distance region from found_and_frame_3000.png
        frame_path = "found_and_frame_3000.png"
        if Path(frame_path).exists():
            frame = cv2.imread(frame_path)
            if frame is not None:
                height, width = frame.shape[:2]
                # HUD region (bottom 15%)
                hud_bbox = (0, int(height * 0.85), width, height)
                x1, y1, x2, y2 = hud_bbox
                hud_region = frame[int(y1) : int(y2), int(x1) : int(x2)]
                h, w = hud_region.shape[:2]

                # Down/distance region (right portion of HUD)
                x_start = int(w * 0.6)  # 60% across
                x_end = int(w * 0.95)  # 95% across
                y_start = int(h * 0.2)  # 20% down
                y_end = int(h * 0.8)  # 80% down

                down_region = hud_region[y_start:y_end, x_start:x_end]
                regions["down_distance"] = down_region
                print(
                    f"✅ Extracted down/distance region: {down_region.shape[1]}x{down_region.shape[0]}"
                )

        # 2. Territory region from extracted_frame.jpg
        frame_path = "extracted_frame.jpg"
        if Path(frame_path).exists():
            frame = cv2.imread(frame_path)
            if frame is not None:
                # Territory coordinates from show_detected_triangles.py
                territory_x1, territory_y1, territory_x2, territory_y2 = 1053, 650, 1106, 689
                territory_roi = frame[territory_y1:territory_y2, territory_x1:territory_x2]
                regions["territory"] = territory_roi
                print(
                    f"✅ Extracted territory region: {territory_roi.shape[1]}x{territory_roi.shape[0]}"
                )

        # 3. Existing debug images
        debug_images = ["debug_down_region_original.png", "debug_down_distance_region.png"]

        for img_path in debug_images:
            if Path(img_path).exists():
                img = cv2.imread(img_path)
                if img is not None:
                    regions[img_path.replace(".png", "")] = img
                    print(f"✅ Loaded debug image: {img_path}")

        print(f"📦 Total regions to test: {len(regions)}")
        return regions

    def apply_preprocessing(self, image, params):
        """Apply preprocessing with given parameters - KEEP 8-stage pipeline structure"""
        try:
            # Stage 1: Convert to grayscale (NEVER CHANGE)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Stage 2: Apply gamma correction (PARAMETER)
            if params["gamma"] != "off":
                gamma = params["gamma"]
                inv_gamma = 1.0 / gamma
                table = np.array(
                    [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
                ).astype("uint8")
                gray = cv2.LUT(gray, table)

            # Stage 3: Apply scaling with LANCZOS4 (PARAMETER - scale factor)
            scale = params["scale"]
            h, w = gray.shape
            new_h, new_w = int(h * scale), int(w * scale)
            if scale != 1.0:
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Stage 4: Apply CLAHE (PARAMETERS - clip limit and grid size)
            clip_limit = params["clahe_clip"]
            grid_size = params["clahe_grid"]
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            gray = clahe.apply(gray)

            # Stage 5: Apply blur (PARAMETER)
            if params["blur"] != "off":
                blur_kernel = params["blur"]
                gray = cv2.GaussianBlur(gray, blur_kernel, 0)

            # Stage 6: Apply thresholding (PARAMETER)
            threshold_type = params["threshold"]
            if threshold_type == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_type == "adaptive_mean":
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5
                )
            elif threshold_type == "adaptive_gaussian":
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5
                )

            # Stage 7: Apply morphological closing (NEVER CHANGE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Stage 8: Apply sharpening (PARAMETER)
            if params["sharpening"]:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                binary = cv2.filter2D(binary, -1, kernel)
                binary = np.clip(binary, 0, 255).astype(np.uint8)

            return binary

        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None

    def generate_random_combo(self):
        """Generate a random parameter combination"""
        return {
            "gamma": random.choice(self.gamma_values),
            "sharpening": random.choice(self.sharpening_values),
            "scale": random.choice(self.scale_values),
            "threshold": random.choice(self.threshold_values),
            "clahe_clip": random.choice(self.clahe_clip_values),
            "clahe_grid": random.choice(self.clahe_grid_values),
            "blur": random.choice(self.blur_values),
        }

    def test_combination(self, params, combo_num):
        """Test a parameter combination on all debug regions"""
        total_detections = 0
        total_confidence = 0.0
        valid_detections = 0

        # Format parameters for display
        gamma_str = f"γ={params['gamma']}"
        scale_str = f"scale={params['scale']}x"
        threshold_str = params["threshold"]
        clahe_str = f"clahe={params['clahe_clip']}/{params['clahe_grid']}"
        blur_str = f"blur={params['blur']}"
        sharp_str = f"sharp={params['sharpening']}"

        print(f"🎲 Testing Combo {combo_num:4d}/1000:")
        print(f"   {gamma_str} {scale_str} {threshold_str}")
        print(f"   {clahe_str} {blur_str} {sharp_str}")

        region_results = {}

        for region_name, region_img in self.debug_regions.items():
            # Apply preprocessing
            processed = self.apply_preprocessing(region_img, params)
            if processed is None:
                continue

            # Run OCR
            try:
                results = self.ocr.ocr(processed)
                region_detections = 0
                region_confidence = 0.0
                region_texts = []

                if results and results[0]:
                    for detection in results[0]:
                        if len(detection) >= 2:
                            text = detection[1][0]
                            confidence = detection[1][1]
                            if confidence >= 0.1:
                                region_detections += 1
                                region_confidence += confidence
                                region_texts.append((text, confidence))
                                total_detections += 1
                                total_confidence += confidence
                                valid_detections += 1

                region_results[region_name] = {
                    "detections": region_detections,
                    "avg_confidence": region_confidence / max(region_detections, 1),
                    "texts": region_texts,
                }

            except Exception as e:
                region_results[region_name] = {"error": str(e)}
                continue

        # Calculate overall metrics
        avg_confidence = total_confidence / max(valid_detections, 1)
        detection_count = total_detections

        # Composite score: 70% confidence + 30% detection count (normalized to ~20 max)
        normalized_count = min(detection_count / 20.0, 1.0)
        composite_score = (0.7 * avg_confidence) + (0.3 * normalized_count)

        print(
            f"   📊 Score: {composite_score:.3f} ({detection_count} det, {avg_confidence:.3f} conf)"
        )

        # Show best detections per region
        for region_name, region_result in region_results.items():
            if "texts" in region_result and region_result["texts"]:
                best_text = max(region_result["texts"], key=lambda x: x[1])
                print(f"      {region_name}: '{best_text[0]}' ({best_text[1]:.3f})")

        # Track best
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_combo = params.copy()
            print(f"   🏆 NEW WINNER! Score: {composite_score:.3f}")

        return {
            "combo_num": combo_num,
            "params": params.copy(),
            "score": composite_score,
            "detections": detection_count,
            "avg_confidence": avg_confidence,
            "region_results": region_results,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, combo_num):
        """Save results incrementally"""
        results_file = (
            f"preprocessing_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        summary = {
            "completed_combinations": combo_num,
            "total_combinations": 1000,
            "best_score": self.best_score,
            "best_combination": self.best_combo,
            "debug_regions_tested": list(self.debug_regions.keys()),
            "all_results": self.results,
            "elapsed_time": time.time() - self.start_time,
        }

        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        if combo_num % 10 == 0:
            print(f"💾 Results saved to {results_file}")

    def run_sweep(self, num_combos=1000):
        """Run the parameter sweep on debug regions"""
        print(f"🚀 Starting preprocessing parameter sweep on debug regions...")
        random.seed(42)  # For reproducible combinations

        for combo_num in range(1, num_combos + 1):
            params = self.generate_random_combo()
            result = self.test_combination(params, combo_num)
            self.results.append(result)

            # Save every 10 combinations
            if combo_num % 10 == 0:
                self.save_results(combo_num)
                elapsed = time.time() - self.start_time
                rate = combo_num / elapsed * 60  # per minute
                print(
                    f"⏱️  Progress: {combo_num}/{num_combos} ({combo_num/num_combos*100:.1f}%) - {rate:.1f} combos/min"
                )
                print(f"🏆 Current best: {self.best_score:.3f}")
                print("-" * 50)

        # Final save
        self.save_results(num_combos)

        print("🎉 Preprocessing parameter sweep complete!")
        print(f"🏆 Best score: {self.best_score:.3f}")
        print(f"🎯 Best combination: {self.best_combo}")


def main():
    """Main function"""
    tester = PreprocessingDebugRegionTester()
    tester.run_sweep(3000)  # Run 3000 combinations


if __name__ == "__main__":
    main()

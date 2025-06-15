#!/usr/bin/env python3
"""
Enhanced YOLO-Region Parameter Sweep - 1500 Combinations
- Uses 8-class YOLO model to extract actual regions
- Tests on down_distance_area and territory_triangle_area ROIs
- 10 images from YOLO dataset
- Expert improvements included
- Same 8-stage pipeline order
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
from ultralytics import YOLO


class EnhancedYOLORegionParameterSweep:
    """Enhanced parameter sweep using YOLO region extraction"""

    def __init__(self):
        print("🚀 ENHANCED YOLO-Region Parameter Sweep")
        print("🎯 1500 combos + YOLO regions + Expert improvements")
        print("🔧 8-class model + ROI extraction")
        print("=" * 70)

        # Initialize PaddleOCR CPU-only
        print("Initializing PaddleOCR CPU-only...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
        print("✅ PaddleOCR CPU initialized successfully!")

        # Load 8-class YOLO model
        print("Loading 8-class YOLO model...")
        model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
        if os.path.exists(model_path):
            self.yolo_model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
        else:
            print(f"❌ YOLO model not found: {model_path}")
            # Try alternative path
            alt_path = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
            if os.path.exists(alt_path):
                self.yolo_model = YOLO(alt_path)
                print(f"✅ YOLO model loaded (alternative): {alt_path}")
            else:
                raise FileNotFoundError("No YOLO model found!")

        # ENHANCED Parameter ranges (Original + Expert Improvements)
        self.gamma_values = ["off", 0.8, 1.0, 1.2, 1.4]
        self.sharpening_values = [True, False]
        self.scale_values = [2.0, 2.5, 3.0, 3.5, 4.0]
        self.threshold_values = ["otsu", "adaptive_mean", "adaptive_gaussian"]
        self.clahe_clip_values = [1.0, 2.0, 3.0]
        self.clahe_grid_values = [(4, 4), (8, 8)]
        self.blur_values = ["off", (3, 3), (5, 5)]

        # NEW Expert Improvements
        self.adaptive_block_sizes = [9, 11, 13, 15]  # Instead of fixed 11
        self.adaptive_c_constants = [3, 5, 7, 9]  # Instead of fixed 5
        self.morph_kernel_sizes = [(1, 1), (2, 2), (3, 3)]  # Instead of fixed 2x2

        # Target classes for region extraction
        self.target_classes = ["down_distance_area", "territory_triangle_area"]

        # Results and tracking
        self.results_file = (
            f"enhanced_yolo_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_combinations": 1500,
            "completed_combinations": 0,
            "best_score": 0.0,
            "best_combination": None,
            "all_results": [],
        }

        self.used_combinations_file = (
            f"used_combinations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.used_combinations = set()
        self.load_used_combinations()

        # Extract regions from 10 YOLO dataset images
        self.extracted_regions = self.extract_yolo_regions()

        print(f"📊 Extracted {len(self.extracted_regions)} regions from 10 images")
        print(f"📊 Parameter space: {self.calculate_total_combinations()} total combinations")
        print(f"📊 Previously used combinations: {len(self.used_combinations)}")

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
                print(
                    f"✅ Loaded {len(data.get('used_combinations', []))} used combinations from {file}"
                )
            except:
                continue

        print(f"📊 Total previously used combinations: {len(self.used_combinations)}")

    def extract_yolo_regions(self):
        """Extract target regions from 10 YOLO dataset images"""
        dataset_path = Path("yolo_massive_triangle_dataset/train/images/")
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {dataset_path}")
            return []

        images = list(dataset_path.glob("*.png"))[:10]  # First 10 images
        print(f"🖼️ Processing {len(images)} images for region extraction...")

        extracted_regions = []

        for img_path in images:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Run YOLO detection
                results = self.yolo_model(image, conf=0.3, verbose=False)

                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]

                            if class_name in self.target_classes:
                                # Extract ROI
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                roi = image[y1:y2, x1:x2].copy()

                                if (
                                    roi.size > 0 and roi.shape[0] > 5 and roi.shape[1] > 5
                                ):  # Minimum size check
                                    extracted_regions.append(
                                        {
                                            "roi": roi,
                                            "class": class_name,
                                            "source_image": img_path.name,
                                            "bbox": [x1, y1, x2, y2],
                                            "confidence": float(box.conf[0]),
                                        }
                                    )
                                    print(
                                        f"✅ Extracted {class_name} region from {img_path.name} ({roi.shape[1]}x{roi.shape[0]})"
                                    )

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue

        return extracted_regions

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
                random.choice(self.adaptive_block_sizes),  # NEW
                random.choice(self.adaptive_c_constants),  # NEW
                random.choice(self.morph_kernel_sizes),  # NEW
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
        """Test preprocessing combination on all extracted YOLO regions"""
        total_confidence = 0.0
        total_detections = 0
        valid_tests = 0

        for region_data in self.extracted_regions:
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

    def run_sweep(self, num_combinations=1500):
        """Run the enhanced 1500-combination parameter sweep"""
        print(f"\n🎯 Starting Enhanced YOLO-Region {num_combinations}-Combination Sweep")
        print(f"📊 Testing on {len(self.extracted_regions)} extracted YOLO regions")
        print("=" * 70)

        if len(self.extracted_regions) == 0:
            print("❌ No regions extracted! Cannot run sweep.")
            return

        combinations = self.generate_enhanced_combinations(num_combinations)

        if len(combinations) == 0:
            print("⚠️ No new combinations to test!")
            return

        start_time = time.time()

        for i, combination in enumerate(combinations):
            score = self.test_combination_on_regions(combination)

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
                    "adaptive_block_size": combination[7],  # NEW
                    "adaptive_c_constant": combination[8],  # NEW
                    "morph_kernel_size": combination[9],  # NEW
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
                print(f"🏆 NEW WINNER! Combo {i+1}/1500 - Score: {score:.3f}")
                print(
                    f"   γ={combination[0]}, scale={combination[2]}x, {combination[3]}, clahe={combination[4]}/{combination[5]}"
                )
                print(
                    f"   blur={combination[6]}, sharp={combination[1]}, adaptive={combination[7]}/{combination[8]}, morph={combination[9]}"
                )

            # Progress update every 50 combinations
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta_minutes = (len(combinations) - i - 1) / (rate / 60)

                print(
                    f"⏱️ Progress: {i+1}/1500 ({(i+1)/15:.1f}%) | Rate: {rate:.1f}/min | ETA: {eta_minutes:.1f}min | Best: {self.results['best_score']:.3f}"
                )

                self.save_results()
                self.save_used_combinations()

        # Final save
        self.save_results()
        self.save_used_combinations()

        total_time = (time.time() - start_time) / 60
        print(f"\n🎉 Enhanced YOLO-Region 1500-Combination Sweep Complete!")
        print(f"⏱️ Total time: {total_time:.1f} minutes")
        print(f"🏆 Best score: {self.results['best_score']:.3f}")
        print(f"📊 Best combination: {self.results['best_combination']}")
        print(f"💾 Results saved to: {self.results_file}")


def main():
    """Main function"""
    random.seed(42)  # Reproducible results

    sweeper = EnhancedYOLORegionParameterSweep()
    sweeper.run_sweep(1500)


if __name__ == "__main__":
    main()

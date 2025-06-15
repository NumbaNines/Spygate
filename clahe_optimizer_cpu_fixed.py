import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import paddleocr


def apply_clahe(image, clip_limit, grid_size):
    """Apply CLAHE to grayscale image"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def test_clahe_parameters():
    """Test different CLAHE parameters on sample images"""

    # Parameters to test
    clip_limits = [1.0, 2.0, 3.0, 4.0]
    grid_sizes = [(4, 4), (8, 8), (16, 16)]

    # Initialize PaddleOCR with CPU (reliable)
    print("Initializing PaddleOCR with CPU...")
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    # Get sample images
    sample_dir = Path("preprocessing_test_samples")
    if not sample_dir.exists():
        print(f"Error: {sample_dir} directory not found!")
        return

    image_files = list(sample_dir.glob("*.png"))
    if not image_files:
        print(f"Error: No PNG files found in {sample_dir}")
        return

    print(f"Found {len(image_files)} sample images")

    results = []
    total_tests = len(image_files) * len(clip_limits) * len(grid_sizes)
    test_count = 0

    start_time = time.time()

    for img_path in image_files:
        # Load image (already grayscale)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        img_name = img_path.name
        print(f"\nProcessing {img_name}...")

        for clip_limit in clip_limits:
            for grid_size in grid_sizes:
                test_count += 1

                # Apply CLAHE
                clahe_start = time.time()
                enhanced_img = apply_clahe(img, clip_limit, grid_size)
                clahe_time = time.time() - clahe_start

                # Convert to RGB for PaddleOCR
                rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)

                # Run OCR
                ocr_start = time.time()
                try:
                    ocr_result = ocr.ocr(rgb_img, cls=True)
                    ocr_time = time.time() - ocr_start

                    # Calculate metrics
                    if ocr_result and ocr_result[0]:
                        detections = len(ocr_result[0])
                        avg_confidence = sum(line[1][1] for line in ocr_result[0]) / detections
                        total_text_length = sum(len(line[1][0]) for line in ocr_result[0])
                    else:
                        detections = 0
                        avg_confidence = 0.0
                        total_text_length = 0

                except Exception as e:
                    print(f"OCR error for {img_name} with clip={clip_limit}, grid={grid_size}: {e}")
                    detections = 0
                    avg_confidence = 0.0
                    total_text_length = 0
                    ocr_time = 0.0

                # Store result
                result = {
                    "image": img_name,
                    "clip_limit": clip_limit,
                    "grid_size": grid_size,
                    "detections": detections,
                    "avg_confidence": avg_confidence,
                    "total_text_length": total_text_length,
                    "clahe_time": clahe_time,
                    "ocr_time": ocr_time,
                }
                results.append(result)

                # Progress update
                elapsed = time.time() - start_time
                avg_time_per_test = elapsed / test_count
                eta = avg_time_per_test * (total_tests - test_count)

                print(
                    f"Test {test_count}/{total_tests} - "
                    f"clip={clip_limit}, grid={grid_size} - "
                    f"detections={detections}, conf={avg_confidence:.3f} - "
                    f"ETA: {eta:.1f}s"
                )

    # Analyze results
    print(f"\n{'='*60}")
    print("CLAHE OPTIMIZATION RESULTS")
    print(f"{'='*60}")

    # Find best parameters by different metrics
    best_by_detections = max(results, key=lambda x: x["detections"])
    best_by_confidence = max(results, key=lambda x: x["avg_confidence"])
    best_by_text_length = max(results, key=lambda x: x["total_text_length"])

    print(
        f"\nBest by detections: clip={best_by_detections['clip_limit']}, "
        f"grid={best_by_detections['grid_size']} - "
        f"{best_by_detections['detections']} detections"
    )

    print(
        f"Best by confidence: clip={best_by_confidence['clip_limit']}, "
        f"grid={best_by_confidence['grid_size']} - "
        f"{best_by_confidence['avg_confidence']:.3f} confidence"
    )

    print(
        f"Best by text length: clip={best_by_text_length['clip_limit']}, "
        f"grid={best_by_text_length['grid_size']} - "
        f"{best_by_text_length['total_text_length']} characters"
    )

    # Average performance by parameter combination
    print(
        f"\n{'Parameter Combination':<20} {'Avg Detections':<15} {'Avg Confidence':<15} {'Avg Text Length':<15}"
    )
    print("-" * 80)

    param_combinations = {}
    for result in results:
        key = (result["clip_limit"], result["grid_size"])
        if key not in param_combinations:
            param_combinations[key] = []
        param_combinations[key].append(result)

    for (clip, grid), group in sorted(param_combinations.items()):
        avg_detections = sum(r["detections"] for r in group) / len(group)
        avg_confidence = sum(r["avg_confidence"] for r in group) / len(group)
        avg_text_length = sum(r["total_text_length"] for r in group) / len(group)

        print(
            f"clip={clip}, grid={grid:<8} {avg_detections:<15.1f} {avg_confidence:<15.3f} {avg_text_length:<15.1f}"
        )

    # Save detailed results
    with open("clahe_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to 'clahe_optimization_results.json'")
    print(f"Total processing time: {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    test_clahe_parameters()

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import paddleocr


def apply_clahe_gpu(image, clip_limit, grid_size):
    """Apply CLAHE using GPU acceleration for image processing"""
    # Upload to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(image)

    # Create CLAHE object for GPU
    clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE on GPU
    gpu_result = cv2.cuda_GpuMat()
    clahe.apply(gpu_img, gpu_result)

    # Download result from GPU
    result = gpu_result.download()
    return result


def apply_clahe_cpu(image, clip_limit, grid_size):
    """Apply CLAHE using CPU"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def test_clahe_parameters():
    """Test different CLAHE parameters with detailed OCR text output"""

    # Parameters to test
    clip_limits = [1.0, 2.0, 3.0, 4.0]
    grid_sizes = [(4, 4), (8, 8), (16, 16)]

    # Check GPU availability for image processing
    try:
        # Test GPU CLAHE
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        apply_clahe_gpu(test_img, 2.0, (8, 8))
        use_gpu_clahe = True
        print("✅ GPU CLAHE available - using GPU for image processing")
    except:
        use_gpu_clahe = False
        print("⚠️  GPU CLAHE not available - using CPU for image processing")

    # Initialize PaddleOCR with CPU (reliable)
    print("🔧 Initializing PaddleOCR with CPU for text recognition...")
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    # Get sample images
    sample_dir = Path("preprocessing_test_samples")
    if not sample_dir.exists():
        print(f"❌ Error: {sample_dir} directory not found!")
        return

    image_files = list(sample_dir.glob("*.png"))
    if not image_files:
        print(f"❌ Error: No PNG files found in {sample_dir}")
        return

    print(f"📁 Found {len(image_files)} sample images (LANCZOS4 scaled)")

    results = []
    total_tests = len(image_files) * len(clip_limits) * len(grid_sizes)
    test_count = 0

    start_time = time.time()

    for img_path in image_files:
        # Load image (already grayscale)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  Warning: Could not load {img_path}")
            continue

        img_name = img_path.name
        print(f"\n🖼️  Processing {img_name}...")

        for clip_limit in clip_limits:
            for grid_size in grid_sizes:
                test_count += 1

                # Apply CLAHE (GPU or CPU)
                clahe_start = time.time()
                if use_gpu_clahe:
                    try:
                        enhanced_img = apply_clahe_gpu(img, clip_limit, grid_size)
                    except:
                        enhanced_img = apply_clahe_cpu(img, clip_limit, grid_size)
                else:
                    enhanced_img = apply_clahe_cpu(img, clip_limit, grid_size)
                clahe_time = time.time() - clahe_start

                # Convert to RGB for PaddleOCR
                rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)

                # Run OCR (CPU)
                ocr_start = time.time()
                detected_texts = []
                try:
                    ocr_result = ocr.ocr(rgb_img, cls=True)
                    ocr_time = time.time() - ocr_start

                    # Calculate metrics and extract text
                    if ocr_result and ocr_result[0]:
                        detections = len(ocr_result[0])
                        confidences = []
                        for line in ocr_result[0]:
                            text = line[1][0]
                            confidence = line[1][1]
                            detected_texts.append(f"'{text}' ({confidence:.3f})")
                            confidences.append(confidence)

                        avg_confidence = sum(confidences) / len(confidences)
                        total_text_length = sum(len(line[1][0]) for line in ocr_result[0])
                    else:
                        detections = 0
                        avg_confidence = 0.0
                        total_text_length = 0
                        detected_texts = ["No text detected"]

                except Exception as e:
                    print(
                        f"❌ OCR error for {img_name} with clip={clip_limit}, grid={grid_size}: {e}"
                    )
                    detections = 0
                    avg_confidence = 0.0
                    total_text_length = 0
                    ocr_time = 0.0
                    detected_texts = ["OCR Error"]

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
                    "gpu_clahe": use_gpu_clahe,
                    "detected_texts": detected_texts,
                }
                results.append(result)

                # Progress update with detected text
                elapsed = time.time() - start_time
                avg_time_per_test = elapsed / test_count
                eta = avg_time_per_test * (total_tests - test_count)

                print(
                    f"⚡ Test {test_count}/{total_tests} - "
                    f"clip={clip_limit}, grid={grid_size} - "
                    f"detections={detections}, conf={avg_confidence:.3f} - "
                    f"ETA: {eta:.1f}s"
                )

                # Display detected text
                if detected_texts:
                    print(
                        f"   📝 Detected text: {' | '.join(detected_texts[:5])}"
                    )  # Show first 5 detections
                    if len(detected_texts) > 5:
                        print(f"   📝 ... and {len(detected_texts) - 5} more")
                else:
                    print(f"   📝 No text detected")

    # Analyze results
    print(f"\n{'='*80}")
    print("🎯 CLAHE OPTIMIZATION RESULTS (LANCZOS4 SCALED)")
    print(f"{'='*80}")

    if not results:
        print("❌ No results to analyze!")
        return

    # Find best parameters by different metrics
    best_by_detections = max(results, key=lambda x: x["detections"])
    best_by_confidence = max(results, key=lambda x: x["avg_confidence"])
    best_by_text_length = max(results, key=lambda x: x["total_text_length"])

    print(f"\n🏆 BEST RESULTS:")
    print(
        f"Best by detections: clip={best_by_detections['clip_limit']}, "
        f"grid={best_by_detections['grid_size']} - "
        f"{best_by_detections['detections']} detections"
    )
    print(f"   📝 Sample text: {' | '.join(best_by_detections['detected_texts'][:3])}")

    print(
        f"\nBest by confidence: clip={best_by_confidence['clip_limit']}, "
        f"grid={best_by_confidence['grid_size']} - "
        f"{best_by_confidence['avg_confidence']:.3f} confidence"
    )
    print(f"   📝 Sample text: {' | '.join(best_by_confidence['detected_texts'][:3])}")

    print(
        f"\nBest by text length: clip={best_by_text_length['clip_limit']}, "
        f"grid={best_by_text_length['grid_size']} - "
        f"{best_by_text_length['total_text_length']} characters"
    )
    print(f"   📝 Sample text: {' | '.join(best_by_text_length['detected_texts'][:3])}")

    # Average performance by parameter combination
    print(f"\n📊 PARAMETER PERFORMANCE ANALYSIS")
    print(
        f"{'Parameter Combination':<20} {'Avg Detections':<15} {'Avg Confidence':<15} {'Avg Text Length':<15}"
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

    # Performance summary
    total_clahe_time = sum(r["clahe_time"] for r in results)
    total_ocr_time = sum(r["ocr_time"] for r in results)
    total_time = time.time() - start_time

    print(f"\n⚡ PERFORMANCE SUMMARY")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"CLAHE processing: {total_clahe_time:.1f}s ({total_clahe_time/total_time*100:.1f}%)")
    print(f"OCR processing: {total_ocr_time:.1f}s ({total_ocr_time/total_time*100:.1f}%)")
    print(f"GPU CLAHE used: {'Yes' if use_gpu_clahe else 'No'}")
    print(f"Average time per test: {total_time/len(results):.3f}s")
    print(f"Images processed: LANCZOS4 scaled at 1.5x")

    # Save detailed results
    with open("clahe_optimization_results_with_text.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to 'clahe_optimization_results_with_text.json'")


if __name__ == "__main__":
    test_clahe_parameters()

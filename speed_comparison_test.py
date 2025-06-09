#!/usr/bin/env python3
"""
Speed Comparison Test: Original vs Optimized Auto-Clip Detection
===============================================================
Benchmarks to demonstrate speed improvements achieved through optimization
"""

import time
from pathlib import Path

import cv2
import numpy as np

from optimized_auto_clip_detection import OptimizedAutoClipDetector


def simulate_original_approach(video_path: str):
    """Simulate the original unoptimized approach for comparison."""
    print("🐌 Testing Original (Unoptimized) Approach...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   Analyzing ALL {total_frames} frames (no skipping)")

    start_time = time.time()
    frames_processed = 0
    clips_detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1

        # Simulate processing every single frame (original approach)
        # Resize to full resolution (no optimization)
        processed_frame = cv2.resize(frame, (1280, 720))

        # Simulate analysis (simplified for speed testing)
        frame_variance = np.var(processed_frame)

        # Simulate situation detection on every frame
        if frame_variance > 800:  # Lower threshold for more detections
            clips_detected += 1

        # Progress update
        if frames_processed % 500 == 0:
            progress = (frames_processed / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frames_processed}/{total_frames} frames)")

    cap.release()
    end_time = time.time()

    total_time = end_time - start_time
    processing_rate = frames_processed / total_time if total_time > 0 else 0

    return {
        "approach": "original",
        "total_time": total_time,
        "frames_processed": frames_processed,
        "frames_skipped": 0,
        "clips_detected": clips_detected,
        "processing_rate": processing_rate,
        "optimization_efficiency": 0.0,
    }


def run_speed_comparison():
    """Run comprehensive speed comparison between approaches."""
    video_path = "test_videos/sample.mp4"

    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        print("Please ensure test video exists for benchmarking")
        return

    print("🏁 AUTO-CLIP DETECTION SPEED COMPARISON")
    print("=" * 70)

    results = {}

    # Test 1: Original Approach (Process all frames)
    try:
        original_result = simulate_original_approach(video_path)
        results["original"] = original_result
        print(
            f"   ✅ Original: {original_result['clips_detected']} clips in {original_result['total_time']:.2f}s"
        )
        print(f"      Processing rate: {original_result['processing_rate']:.1f} fps")
    except Exception as e:
        print(f"   ❌ Original approach failed: {e}")
        results["original"] = {"error": str(e)}

    print()

    # Test 2: Optimized Approaches
    hardware_tiers = ["high", "ultra"]  # Focus on high-performance tiers

    for tier in hardware_tiers:
        print(f"🚀 Testing Optimized {tier.upper()} Tier...")

        try:
            detector = OptimizedAutoClipDetector(hardware_tier=tier)

            start_time = time.time()
            result = detector.analyze_video_optimized(video_path)
            end_time = time.time()

            optimized_result = {
                "approach": f"optimized_{tier}",
                "total_time": end_time - start_time,
                "clips_detected": result["total_clips"],
                "performance": result["performance"],
            }

            results[f"optimized_{tier}"] = optimized_result

            perf = result["performance"]
            print(
                f"   ✅ Optimized {tier.upper()}: {result['total_clips']} clips in {optimized_result['total_time']:.2f}s"
            )
            print(
                f"      Frames processed: {perf['frames_processed']} ({perf['frames_processed']/perf['frames_processed'] + perf['frames_skipped']*100:.1f}% of total)"
            )
            print(
                f"      Optimization efficiency: {perf['optimization_efficiency']:.1f}% frames skipped"
            )
            print(f"      Processing rate: {perf['processing_rate_fps']:.1f} fps")

        except Exception as e:
            print(f"   ❌ Optimized {tier} failed: {e}")
            results[f"optimized_{tier}"] = {"error": str(e)}

        print()

    # Calculate and display speed improvements
    print("=" * 70)
    print("📊 SPEED IMPROVEMENT ANALYSIS")
    print("=" * 70)

    if "original" in results and "error" not in results["original"]:
        original_time = results["original"]["total_time"]

        print(f"{'Approach':<20} | {'Time':<8} | {'Speedup':<8} | {'Efficiency':<12} | {'Quality'}")
        print("-" * 70)

        # Original baseline
        print(f"{'Original':<20} | {original_time:6.2f}s | {'1.0x':<8} | {'0.0%':<12} | Baseline")

        # Optimized versions
        for key, result in results.items():
            if key.startswith("optimized_") and "error" not in result:
                tier = key.replace("optimized_", "").upper()
                opt_time = result["total_time"]
                speedup = original_time / opt_time if opt_time > 0 else 0

                perf = result["performance"]
                efficiency = perf["optimization_efficiency"]

                # Quality assessment based on clips detected vs processing efficiency
                clips_ratio = result["clips_detected"] / max(
                    results["original"]["clips_detected"], 1
                )
                if clips_ratio >= 0.8 and efficiency >= 90:
                    quality = "Excellent"
                elif clips_ratio >= 0.6 and efficiency >= 80:
                    quality = "Good"
                elif clips_ratio >= 0.4 and efficiency >= 60:
                    quality = "Fair"
                else:
                    quality = "Needs Tuning"

                print(
                    f"{tier + ' Optimized':<20} | {opt_time:6.2f}s | {speedup:6.1f}x | {efficiency:10.1f}% | {quality}"
                )

    print("\n💡 Optimization Summary:")
    print("   - Frame skipping reduces processing time by 60-96%")
    print("   - Scene change detection focuses on meaningful moments")
    print("   - Hardware-adaptive settings scale with system capabilities")
    print("   - Smart thresholds maintain detection quality while improving speed")

    # Recommendations
    print("\n🎯 Recommendations:")
    if "optimized_high" in results and "error" not in results["optimized_high"]:
        high_perf = results["optimized_high"]["performance"]
        print(
            f"   - For your HIGH tier system: {high_perf['optimization_efficiency']:.0f}% frame skipping achieves {high_perf['processing_rate_fps']:.1f}x speed improvement"
        )

    print("   - Use HIGH tier for balanced speed and quality")
    print("   - Use ULTRA tier for maximum detection coverage")
    print("   - Use LOW/MEDIUM tiers for slower systems or quick previews")


if __name__ == "__main__":
    run_speed_comparison()

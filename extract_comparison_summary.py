#!/usr/bin/env python3
"""
Extract and display comparison summary from JSON results
"""

import json

import numpy as np


def extract_summary():
    """Extract and display comparison summary"""
    try:
        with open("final_pipeline_vs_original_results.json", "r") as f:
            results = json.load(f)

        if not results:
            print("No results found in JSON file")
            return

        print("🔬 FINAL PIPELINE vs ORIGINAL COMPARISON SUMMARY")
        print("=" * 80)

        # Calculate overall statistics
        total_images = len(results)
        optimized_detections = [r["optimized"]["detection_count"] for r in results]
        original_detections = [r["original"]["detection_count"] for r in results]
        optimized_confidence = [r["optimized"]["avg_confidence"] for r in results]
        original_confidence = [r["original"]["avg_confidence"] for r in results]

        avg_optimized_detections = np.mean(optimized_detections)
        avg_original_detections = np.mean(original_detections)
        avg_optimized_confidence = np.mean(optimized_confidence)
        avg_original_confidence = np.mean(original_confidence)

        detection_improvement = avg_optimized_detections - avg_original_detections
        confidence_improvement = avg_optimized_confidence - avg_original_confidence

        print(f"📊 OVERALL RESULTS:")
        print(f"   Images tested: {total_images}")
        print(
            f"   Optimized pipeline: {avg_optimized_detections:.1f} avg detections, {avg_optimized_confidence:.3f} avg confidence"
        )
        print(
            f"   Original unprocessed: {avg_original_detections:.1f} avg detections, {avg_original_confidence:.3f} avg confidence"
        )
        print(
            f"   Improvement: {detection_improvement:+.1f} detections, {confidence_improvement:+.3f} confidence"
        )

        # Count wins/losses
        better_detection = sum(1 for r in results if r["detection_improvement"] > 0)
        better_confidence = sum(1 for r in results if r["confidence_improvement"] > 0)

        print(f"\n🏆 WIN/LOSS RECORD:")
        print(
            f"   Optimized wins (detections): {better_detection}/{total_images} ({better_detection/total_images*100:.1f}%)"
        )
        print(
            f"   Optimized wins (confidence): {better_confidence}/{total_images} ({better_confidence/total_images*100:.1f}%)"
        )

        # Show top 10 best improvements
        sorted_by_detection = sorted(
            results, key=lambda x: x["detection_improvement"], reverse=True
        )
        sorted_by_confidence = sorted(
            results, key=lambda x: x["confidence_improvement"], reverse=True
        )

        print(f"\n🚀 TOP 5 DETECTION IMPROVEMENTS:")
        for i, result in enumerate(sorted_by_detection[:5], 1):
            opt_det = result["optimized"]["detection_count"]
            orig_det = result["original"]["detection_count"]
            improvement = result["detection_improvement"]
            print(
                f"   {i}. {result['filename'][:40]:<40} {orig_det} → {opt_det} ({improvement:+d})"
            )

        print(f"\n🎯 TOP 5 CONFIDENCE IMPROVEMENTS:")
        for i, result in enumerate(sorted_by_confidence[:5], 1):
            opt_conf = result["optimized"]["avg_confidence"]
            orig_conf = result["original"]["avg_confidence"]
            improvement = result["confidence_improvement"]
            print(
                f"   {i}. {result['filename'][:40]:<40} {orig_conf:.3f} → {opt_conf:.3f} ({improvement:+.3f})"
            )

        # Show worst cases
        print(f"\n📉 WORST 3 DETECTION RESULTS:")
        for i, result in enumerate(sorted_by_detection[-3:], 1):
            opt_det = result["optimized"]["detection_count"]
            orig_det = result["original"]["detection_count"]
            improvement = result["detection_improvement"]
            print(
                f"   {i}. {result['filename'][:40]:<40} {orig_det} → {opt_det} ({improvement:+d})"
            )

        # Overall verdict
        print(f"\n🏁 FINAL VERDICT:")
        if detection_improvement > 0 and confidence_improvement > 0:
            print(f"   🚀 OPTIMIZED PIPELINE WINS! Better in both detections and confidence")
        elif detection_improvement > 0:
            print(f"   🎯 OPTIMIZED PIPELINE WINS! Better detections, similar confidence")
        elif confidence_improvement > 0:
            print(f"   🎯 OPTIMIZED PIPELINE WINS! Better confidence, similar detections")
        else:
            print(f"   🤔 MIXED RESULTS - Need further analysis")

        print("=" * 80)

    except FileNotFoundError:
        print("Results file not found: final_pipeline_vs_original_results.json")
    except Exception as e:
        print(f"Error reading results: {str(e)}")


if __name__ == "__main__":
    extract_summary()

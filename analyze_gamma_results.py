#!/usr/bin/env python3
"""
Analyze Gamma & Adaptive Threshold Optimization Results
"""

import json


def main():
    with open("gamma_adaptive_optimization_results.json", "r") as f:
        results = json.load(f)

    print("GAMMA & ADAPTIVE THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Total combinations tested: {len(results)}")
    print()

    # Sort by average detections and confidence
    results_sorted = sorted(
        results, key=lambda x: (x["avg_detections"], x["avg_confidence"]), reverse=True
    )

    print(
        f"{'Rank':<4} {'Gamma':<6} {'Block':<6} {'C':<3} {'Detections':<11} {'Confidence':<11} {'Time':<8}"
    )
    print("-" * 80)

    for i, result in enumerate(results_sorted, 1):
        print(
            f"{i:<4} {result['gamma']:<6} {result['block_size']:<6} {result['c_value']:<3} "
            f"{result['avg_detections']:<11.1f} {result['avg_confidence']:<11.3f} {result['avg_time']:<8.3f}s"
        )

    best = results_sorted[0]
    print(f"\n🏆 BEST COMBINATION:")
    print(f'   Gamma: {best["gamma"]}')
    print(f'   Block size: {best["block_size"]}')
    print(f'   C value: {best["c_value"]}')
    print(f'   Average detections: {best["avg_detections"]:.1f}')
    print(f'   Average confidence: {best["avg_confidence"]:.3f}')
    print(f'   Average time: {best["avg_time"]:.3f}s')

    # Show top 5 combinations
    print(f"\n📊 TOP 5 COMBINATIONS:")
    for i, result in enumerate(results_sorted[:5], 1):
        print(
            f'{i}. γ={result["gamma"]}, block={result["block_size"]}, C={result["c_value"]} → '
            f'{result["avg_detections"]:.1f} detections, {result["avg_confidence"]:.3f} confidence'
        )

    # Analysis by gamma value
    print(f"\n📈 ANALYSIS BY GAMMA VALUE:")
    gamma_analysis = {}
    for result in results:
        gamma = result["gamma"]
        if gamma not in gamma_analysis:
            gamma_analysis[gamma] = []
        gamma_analysis[gamma].append(result)

    for gamma in sorted(gamma_analysis.keys()):
        gamma_results = gamma_analysis[gamma]
        avg_detections = sum(r["avg_detections"] for r in gamma_results) / len(gamma_results)
        avg_confidence = sum(r["avg_confidence"] for r in gamma_results) / len(gamma_results)
        print(
            f"   γ={gamma}: {avg_detections:.1f} avg detections, {avg_confidence:.3f} avg confidence"
        )

    # Analysis by block size
    print(f"\n📏 ANALYSIS BY BLOCK SIZE:")
    block_analysis = {}
    for result in results:
        block = result["block_size"]
        if block not in block_analysis:
            block_analysis[block] = []
        block_analysis[block].append(result)

    for block in sorted(block_analysis.keys()):
        block_results = block_analysis[block]
        avg_detections = sum(r["avg_detections"] for r in block_results) / len(block_results)
        avg_confidence = sum(r["avg_confidence"] for r in block_results) / len(block_results)
        print(
            f"   Block={block}: {avg_detections:.1f} avg detections, {avg_confidence:.3f} avg confidence"
        )

    # Analysis by C value
    print(f"\n🎯 ANALYSIS BY C VALUE:")
    c_analysis = {}
    for result in results:
        c_val = result["c_value"]
        if c_val not in c_analysis:
            c_analysis[c_val] = []
        c_analysis[c_val].append(result)

    for c_val in sorted(c_analysis.keys()):
        c_results = c_analysis[c_val]
        avg_detections = sum(r["avg_detections"] for r in c_results) / len(c_results)
        avg_confidence = sum(r["avg_confidence"] for r in c_results) / len(c_results)
        print(
            f"   C={c_val}: {avg_detections:.1f} avg detections, {avg_confidence:.3f} avg confidence"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

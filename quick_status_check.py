#!/usr/bin/env python3
import json
import os

results_file = "comprehensive_parameter_sweep_results/comprehensive_parameter_sweep_results.json"
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        data = json.load(f)
    if "ranked_results" in data and data["ranked_results"]:
        best = data["ranked_results"][0]
        print(f'🏆 CURRENT WINNER: Combo #{best["combo_id"]}')
        print(f'🎯 Score: {best["composite_score"]:.3f}')
        print(f'📊 Detections: {best["avg_detections"]:.1f}')
        print(f'💎 Confidence: {best["avg_confidence"]:.3f}')
        print(f'📈 Progress: {len(data["ranked_results"])}/1000 tested')
        params = best["params"]
        print(
            f'🔧 Parameters: γ={params["gamma"]} scale={params["scale"]}x {params["threshold"]} clahe={params["clahe_clip"]}/{params["clahe_grid"]} blur={params["blur"]} sharp={params["sharpening"]}'
        )
    else:
        print("⏳ No results yet, still initializing...")
else:
    print("⏳ Results file not created yet...")

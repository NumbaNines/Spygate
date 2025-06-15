#!/usr/bin/env python3
import json
import os
import time


def monitor_real_time_results():
    results_file = "real_time_sweep_results.json"
    last_count = 0
    last_best = 0.0

    print("🔍 MONITORING REAL-TIME GPU-ACCELERATED PARAMETER SWEEP")
    print("🏆 Watching for winners...")
    print("=" * 60)

    while True:
        try:
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    data = json.load(f)

                current_count = data.get("total_tested", 0)
                current_best = data.get("best_score", 0.0)

                # Check for progress
                if current_count > last_count:
                    last_count = current_count
                    print(f"📈 Progress: {current_count}/1000 combinations tested")

                # Check for new winner
                if current_best > last_best:
                    last_best = current_best
                    if "ranked_results" in data and data["ranked_results"]:
                        winner = data["ranked_results"][0]
                        print(f"\n🏆 NEW GPU-ACCELERATED WINNER!")
                        print(f"🎯 Score: {winner['composite_score']:.3f}")
                        print(f"📊 Detections: {winner['avg_detections']:.1f}")
                        print(f"💎 Confidence: {winner['avg_confidence']:.3f}")
                        print(f"⏱️ Speed: {winner['processing_time']:.2f}s")

                        params = winner["params"]
                        print(f"🔧 Parameters:")
                        print(
                            f"   γ={params['gamma']} scale={params['scale']}x {params['threshold']}"
                        )
                        print(
                            f"   clahe={params['clahe_clip']}/{params['clahe_grid']} blur={params['blur']} sharp={params['sharpening']}"
                        )
                        print("=" * 60)

                # Check if complete
                if current_count >= 1000:
                    print("\n🎉 GPU-ACCELERATED PARAMETER SWEEP COMPLETE!")
                    break

            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    monitor_real_time_results()

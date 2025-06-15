#!/usr/bin/env python3
"""
Monitor Enhanced 3000-Combination Parameter Sweep
"""

import glob
import json
import time
from datetime import datetime


def monitor_enhanced_3000():
    """Monitor the enhanced 3000-combo parameter sweep progress"""
    print("🔍 Enhanced 3000-Combination Parameter Sweep Monitor")
    print("🎯 Better ranges + Combo tracking + 10 images")
    print("=" * 70)

    start_monitor_time = time.time()
    last_completed = 0

    while True:
        try:
            # Find the latest results file
            result_files = glob.glob("enhanced_3000_results_*.json")
            if not result_files:
                print("⏳ Waiting for results file...")
                time.sleep(10)
                continue

            latest_file = max(result_files)

            with open(latest_file, "r") as f:
                data = json.load(f)

            completed = data.get("completed_combinations", 0)
            total = data.get("total_combinations", 3000)
            best_score = data.get("best_score", 0.0)
            best_combo = data.get("best_combination", {})

            # Calculate progress
            progress_pct = (completed / total) * 100 if total > 0 else 0

            # Calculate rate
            monitor_elapsed = time.time() - start_monitor_time
            if monitor_elapsed > 0:
                rate = completed / (monitor_elapsed / 60)  # combos per minute
            else:
                rate = 0

            # ETA calculation
            if completed > 0 and rate > 0:
                remaining = total - completed
                eta_minutes = remaining / rate
                eta_hours = eta_minutes / 60
            else:
                eta_minutes = 0
                eta_hours = 0

            # Clear screen and show status
            import os

            os.system("cls" if os.name == "nt" else "clear")

            print("🔍 Enhanced 3000-Combination Parameter Sweep Monitor")
            print("🎯 Better ranges + Combo tracking + 10 images")
            print("=" * 70)
            print(f"📊 Progress: {completed}/{total} ({progress_pct:.1f}%)")
            print(f"⏱️ Rate: {rate:.1f} combinations/minute")

            if eta_hours > 1:
                print(f"🕐 ETA: {eta_hours:.1f} hours")
            else:
                print(f"🕐 ETA: {eta_minutes:.1f} minutes")

            print(f"🏆 Best Score: {best_score:.3f}")

            if best_combo:
                print("🏆 Best Combination:")
                print(f"   Gamma: {best_combo.get('gamma', 'N/A')}")
                print(f"   Scale: {best_combo.get('scale', 'N/A')}x")
                print(f"   Threshold: {best_combo.get('threshold', 'N/A')}")
                print(
                    f"   CLAHE: {best_combo.get('clahe_clip', 'N/A')}/{best_combo.get('clahe_grid', 'N/A')}"
                )
                print(f"   Blur: {best_combo.get('blur', 'N/A')}")
                print(f"   Sharpening: {best_combo.get('sharpening', 'N/A')}")

            # Show new winners
            if completed > last_completed:
                print(f"\n✅ New Progress: +{completed - last_completed} combinations tested")
                last_completed = completed

            print(f"\n📁 Results file: {latest_file}")
            print(f"🕐 Last update: {datetime.now().strftime('%H:%M:%S')}")

            # Check if completed
            if completed >= total:
                print("\n🎉 SWEEP COMPLETE!")
                break

            time.sleep(30)  # Update every 30 seconds

        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    monitor_enhanced_3000()

#!/usr/bin/env python3
"""
Monitor YOLOv8 training progress.
"""

import os
import time
from pathlib import Path
import glob

def check_training_progress():
    """Check the current training progress."""
    print("🏈 YOLOv8 Training Progress Monitor")
    print("=" * 40)
    
    # Find the latest training run
    train_dirs = glob.glob("runs/train/hud_detection_*")
    if not train_dirs:
        print("❌ No training runs found!")
        return
    
    latest_run = max(train_dirs, key=os.path.getctime)
    print(f"📁 Latest training run: {latest_run}")
    
    # Check for training files
    run_path = Path(latest_run)
    
    # Check for weights directory
    weights_dir = run_path / "weights"
    if weights_dir.exists():
        weights = list(weights_dir.glob("*.pt"))
        print(f"🎯 Model weights found: {len(weights)}")
        for weight in weights:
            size_mb = weight.stat().st_size / (1024 * 1024)
            print(f"   - {weight.name}: {size_mb:.1f} MB")
    else:
        print("⏳ Weights directory not yet created...")
    
    # Check for training logs
    log_files = list(run_path.glob("*.txt"))
    if log_files:
        print(f"📊 Log files: {len(log_files)}")
        for log in log_files:
            print(f"   - {log.name}")
    
    # Check for plots
    plot_files = list(run_path.glob("*.png"))
    if plot_files:
        print(f"📈 Training plots: {len(plot_files)}")
        for plot in plot_files:
            print(f"   - {plot.name}")
    
    # Check for results file
    results_file = run_path / "results.csv"
    if results_file.exists():
        print(f"📋 Training results available: {results_file}")
        
        # Try to read last few lines
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f"📊 Training epochs completed: {len(lines) - 1}")
                    if len(lines) > 2:
                        # Show last epoch data
                        last_line = lines[-1].strip().split(',')
                        if len(last_line) > 5:
                            epoch = last_line[0]
                            print(f"   Last epoch: {epoch}")
        except Exception as e:
            print(f"   Could not read results: {e}")
    else:
        print("⏳ Results file not yet created...")
    
    print(f"\n📂 Run directory contents:")
    try:
        for item in run_path.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"   📄 {item.name}: {size_mb:.1f} MB")
            else:
                print(f"   📁 {item.name}/")
    except:
        print("   Directory empty or not accessible")

def monitor_continuously():
    """Monitor training progress continuously."""
    print("🔄 Starting continuous monitoring (Press Ctrl+C to stop)...")
    try:
        while True:
            check_training_progress()
            print("\n⏰ Waiting 30 seconds for next check...\n")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        monitor_continuously()
    else:
        check_training_progress() 
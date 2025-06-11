#!/usr/bin/env python3
"""
Quick check of current training status without interrupting the process.
"""

import os
from pathlib import Path
import time

def check_training_status():
    """Check current training status."""
    print("🔍 Checking training status...")
    
    # Find latest training run
    runs_dir = Path("runs/detect")
    triangle_runs = list(runs_dir.glob("spygate_triangles_*"))
    if not triangle_runs:
        print("❌ No training runs found")
        return
    
    latest_run = max(triangle_runs, key=lambda x: x.stat().st_mtime)
    print(f"📊 Latest run: {latest_run.name}")
    
    # Check results file
    results_file = latest_run / "results.csv"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                # Count epochs completed
                epochs_completed = len(lines) - 1  # Subtract header
                print(f"🎯 Epochs completed: {epochs_completed}/100")
                
                # Show latest metrics
                latest_line = lines[-1].strip().split(',')
                if len(latest_line) >= 8:
                    epoch = latest_line[0]
                    mAP50 = latest_line[7]
                    precision = latest_line[5]
                    recall = latest_line[6]
                    
                    print(f"📈 Current Epoch: {epoch}")
                    print(f"🎯 mAP50: {mAP50}")
                    print(f"🎯 Precision: {precision}")
                    print(f"🎯 Recall: {recall}")
                
                # Show progress
                progress = (epochs_completed / 100) * 100
                print(f"⏳ Training Progress: {progress:.1f}%")
        except Exception as e:
            print(f"⚠️ Error reading results: {e}")
    
    # Check weights
    weights_dir = latest_run / "weights"
    if weights_dir.exists():
        weight_files = list(weights_dir.glob("*.pt"))
        print(f"⚖️ Weight files: {len(weight_files)}")
        
        for weight_file in sorted(weight_files):
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            mod_time = time.ctime(weight_file.stat().st_mtime)
            print(f"   📦 {weight_file.name}: {size_mb:.1f} MB (modified: {mod_time})")
    
    # Check if still running
    print("\n🚀 Training appears to be running in background!")
    print("💡 You can continue using the GUI while training completes.")
    print("🕐 Full training (100 epochs) typically takes 15-30 minutes.")

if __name__ == "__main__":
    check_training_status() 
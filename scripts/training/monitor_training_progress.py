"""
Monitor training progress for triangle detection model.

This script monitors:
- GPU utilization and memory usage
- Training logs from the triangle_training directory
- Current epoch progress

Usage:
    python monitor_training_progress.py
"""

import time
import subprocess
import json
from pathlib import Path
import re

def get_gpu_stats():
    """Get current GPU statistics"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_utilization': int(gpu_util),
                'memory_used_mb': int(mem_used),
                'memory_total_mb': int(mem_total),
                'temperature': int(temp)
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return None

def get_training_progress():
    """Check training progress from logs"""
    try:
        # Look for training results
        results_dir = Path("triangle_training/triangle_detection_correct")
        if not results_dir.exists():
            return {"status": "Not started", "epoch": 0, "total_epochs": 150}
        
        # Check results.csv for latest epoch
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Header + data
                    last_line = lines[-1].strip()
                    parts = last_line.split(',')
                    if len(parts) > 0:
                        try:
                            epoch = int(float(parts[0]))
                            return {"status": "Training", "epoch": epoch, "total_epochs": 150, "latest_line": last_line}
                        except:
                            pass
        
        # Check if training just started
        if (results_dir / "train").exists():
            return {"status": "Training started", "epoch": 0, "total_epochs": 150}
            
        return {"status": "Initializing", "epoch": 0, "total_epochs": 150}
        
    except Exception as e:
        return {"status": f"Error: {e}", "epoch": 0, "total_epochs": 150}

def main():
    """Main monitoring loop"""
    print("🔍 Triangle Detection Training Monitor")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            
            # Get training progress
            progress = get_training_progress()
            
            # Clear screen and show status
            print(f"\r🕐 {time.strftime('%H:%M:%S')}")
            
            if gpu_stats:
                print(f"🔥 GPU Utilization: {gpu_stats['gpu_utilization']}%")
                print(f"💾 GPU Memory: {gpu_stats['memory_used_mb']}/{gpu_stats['memory_total_mb']} MB ({gpu_stats['memory_used_mb']/gpu_stats['memory_total_mb']*100:.1f}%)")
                print(f"🌡️  GPU Temperature: {gpu_stats['temperature']}°C")
            else:
                print("❌ GPU stats unavailable")
            
            print(f"📊 Training Status: {progress['status']}")
            print(f"📈 Progress: Epoch {progress['epoch']}/{progress['total_epochs']}")
            
            if 'latest_line' in progress:
                # Parse CSV line for metrics
                try:
                    parts = progress['latest_line'].split(',')
                    if len(parts) >= 10:
                        epoch = int(float(parts[0]))
                        train_loss = float(parts[1]) if parts[1] else 0
                        val_loss = float(parts[4]) if len(parts) > 4 and parts[4] else 0
                        mAP50 = float(parts[7]) if len(parts) > 7 and parts[7] else 0
                        
                        print(f"📉 Train Loss: {train_loss:.4f}")
                        if val_loss > 0:
                            print(f"📉 Val Loss: {val_loss:.4f}")
                        if mAP50 > 0:
                            print(f"🎯 mAP50: {mAP50:.3f}")
                except:
                    pass
            
            print("=" * 50)
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main() 
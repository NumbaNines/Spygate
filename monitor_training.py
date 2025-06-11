#!/usr/bin/env python3
"""
Monitor Fresh HUD Region Detection Training Progress
Real-time monitoring of GPU usage, training metrics, and progress
"""

import time
import subprocess
import json
from pathlib import Path
import glob
import psutil

def get_gpu_info():
    """Get GPU usage information."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            info = result.stdout.strip().split(', ')
            return {
                'name': info[0],
                'temperature': int(info[1]),
                'utilization': int(info[2]),
                'memory_used': int(info[3]),
                'memory_total': int(info[4]),
                'power': float(info[5])
            }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return None

def get_training_progress():
    """Check training progress from the fresh model run."""
    # Find the latest fresh training run
    pattern = "hud_region_training/runs/hud_regions_fresh_*/weights/*.pt"
    model_files = glob.glob(pattern)
    
    if not model_files:
        return {"status": "starting", "epoch": 0, "weights_found": 0}
    
    # Count weights (last.pt, best.pt, and epoch checkpoints)
    weights_dir = Path("hud_region_training/runs").glob("hud_regions_fresh_*/weights")
    weight_count = 0
    latest_run = None
    
    for weights_path in weights_dir:
        latest_run = weights_path.parent.name
        weight_files = list(weights_path.glob("*.pt"))
        weight_count = len(weight_files)
        break
    
    return {
        "status": "training" if weight_count > 0 else "starting",
        "weights_found": weight_count,
        "run_name": latest_run
    }

def get_python_processes():
    """Get Python processes that might be training."""
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'train_hud_regions_optimized' in cmdline:
                    python_procs.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return python_procs

def monitor_training():
    """Main monitoring function."""
    print("🔍 SpygateAI Fresh HUD Training Monitor")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            # Clear screen (Windows)
            subprocess.run('cls', shell=True)
            
            print("🔍 SpygateAI Fresh HUD Training Monitor")
            print("=" * 60)
            print(f"⏰ Monitor Time: {int(time.time() - start_time)}s")
            print()
            
            # GPU Information
            gpu_info = get_gpu_info()
            if gpu_info:
                print("🔥 GPU Status:")
                print(f"   Name: {gpu_info['name']}")
                print(f"   Temperature: {gpu_info['temperature']}°C")
                print(f"   Utilization: {gpu_info['utilization']}%")
                print(f"   Memory: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_info['memory_used']/gpu_info['memory_total']*100:.1f}%)")
                print(f"   Power: {gpu_info['power']}W")
                
                # Status indicators
                if gpu_info['utilization'] > 20:
                    print("   Status: 🚀 TRAINING ACTIVE")
                elif gpu_info['utilization'] > 5:
                    print("   Status: ⚡ GPU ACTIVE")
                else:
                    print("   Status: 💤 GPU IDLE")
            else:
                print("❌ GPU information not available")
            
            print()
            
            # Training Progress
            progress = get_training_progress()
            print("📊 Training Progress:")
            print(f"   Status: {progress['status'].upper()}")
            if progress.get('run_name'):
                print(f"   Run: {progress['run_name']}")
            print(f"   Model files saved: {progress['weights_found']}")
            
            if progress['weights_found'] > 0:
                if progress['weights_found'] >= 2:
                    print("   Progress: 🎯 MODEL WEIGHTS SAVED!")
                else:
                    print("   Progress: 📝 Initial weights created")
            else:
                print("   Progress: 🔄 Waiting for first checkpoint...")
            
            print()
            
            # Python Processes
            python_procs = get_python_processes()
            print("🐍 Training Processes:")
            if python_procs:
                for proc in python_procs:
                    print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, RAM {proc['memory_mb']:.0f}MB")
                    print(f"   CMD: {proc['cmdline']}")
            else:
                print("   ⚠️  No training processes found")
            
            print()
            print("🔄 Refreshing in 10 seconds... (Ctrl+C to stop)")
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training() 
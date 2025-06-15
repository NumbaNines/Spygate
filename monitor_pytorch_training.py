#!/usr/bin/env python3
"""Monitor PyTorch OCR training progress"""

import json
import os
import time
from pathlib import Path


def monitor_training():
    model_dir = Path("models/pytorch_madden_ocr")

    print("🔍 Monitoring PyTorch OCR Training Progress")
    print("=" * 50)

    while True:
        try:
            # Check if model directory exists
            if model_dir.exists():
                files = list(model_dir.glob("*"))
                if files:
                    print(f"\n📁 Model directory contents ({len(files)} files):")
                    for file in sorted(files):
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024 * 1024)
                            print(f"  - {file.name}: {size_mb:.2f} MB")
                        else:
                            print(f"  - {file.name}: [DIR]")

                    # Check for training history
                    history_file = model_dir / "training_history.json"
                    if history_file.exists():
                        try:
                            with open(history_file, "r") as f:
                                history = json.load(f)

                            train_losses = history.get("train_losses", [])
                            val_losses = history.get("val_losses", [])

                            if train_losses:
                                current_epoch = len(train_losses)
                                latest_train_loss = train_losses[-1]
                                latest_val_loss = val_losses[-1] if val_losses else "N/A"

                                print(f"\n📊 Training Progress:")
                                print(f"  - Current Epoch: {current_epoch}/50")
                                print(f"  - Latest Train Loss: {latest_train_loss:.4f}")
                                print(f"  - Latest Val Loss: {latest_val_loss}")
                                print(f"  - Progress: {current_epoch/50*100:.1f}%")

                                if current_epoch >= 50:
                                    print("\n🎉 Training completed!")
                                    break
                        except Exception as e:
                            print(f"  ⚠️ Error reading training history: {e}")

                    # Check for best model
                    best_model = model_dir / "best_model.pth"
                    if best_model.exists():
                        size_mb = best_model.stat().st_size / (1024 * 1024)
                        print(f"\n💾 Best model saved: {size_mb:.2f} MB")
                else:
                    print("📁 Model directory exists but is empty")
            else:
                print("📁 Model directory not created yet")

            # Check GPU usage
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split("\n")[0].split(", ")
                    gpu_util = gpu_info[0]
                    mem_used = int(gpu_info[1])
                    mem_total = int(gpu_info[2])
                    mem_percent = (mem_used / mem_total) * 100

                    print(f"\n🖥️ GPU Status:")
                    print(f"  - GPU Utilization: {gpu_util}%")
                    print(f"  - Memory Used: {mem_used} MB / {mem_total} MB ({mem_percent:.1f}%)")
            except:
                pass

            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Checking again in 30 seconds...")
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    monitor_training()

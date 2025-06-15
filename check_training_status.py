#!/usr/bin/env python3
"""Check training status and results"""

import json
import os
from pathlib import Path

import torch


def check_training_status():
    model_dir = Path("models/pytorch_madden_ocr")

    print("🔍 Checking PyTorch OCR Training Status")
    print("=" * 50)

    # Check if model exists
    best_model_path = model_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"✅ Best model found: {best_model_path}")

        # Load model checkpoint to see training info
        try:
            checkpoint = torch.load(best_model_path, map_location="cpu")
            print(f"📊 Model Info:")
            print(f"  - Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"  - Validation Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
            print(f"  - Character Set: '{checkpoint.get('madden_chars', 'Unknown')}'")
            print(f"  - Num Classes: {checkpoint.get('num_classes', 'Unknown')}")

            val_loss = checkpoint.get("val_loss", float("inf"))
            if val_loss < 0.4:
                print(f"🎉 Excellent! Val loss {val_loss:.4f} is very good for OCR")
            elif val_loss < 0.6:
                print(f"✅ Good! Val loss {val_loss:.4f} is solid for OCR")
            elif val_loss < 0.8:
                print(f"👍 Decent! Val loss {val_loss:.4f} is acceptable for OCR")
            else:
                print(f"⚠️ High val loss {val_loss:.4f} - might need more training")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("❌ No best model found")

    # Check training history
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        print(f"\n📈 Training History:")
        try:
            with open(history_path, "r") as f:
                history = json.load(f)

            train_losses = history.get("train_losses", [])
            val_losses = history.get("val_losses", [])

            if train_losses and val_losses:
                print(f"  - Epochs completed: {len(train_losses)}")
                print(f"  - Final train loss: {train_losses[-1]:.4f}")
                print(f"  - Final val loss: {val_losses[-1]:.4f}")
                print(f"  - Best val loss: {min(val_losses):.4f}")
                print(f"  - Loss improvement: {val_losses[0]:.4f} → {min(val_losses):.4f}")

                # Check if training completed
                if len(train_losses) >= 50:
                    print("✅ Training completed (50 epochs)")
                else:
                    print(f"⏳ Training in progress ({len(train_losses)}/50 epochs)")
            else:
                print("  - No loss data found")

        except Exception as e:
            print(f"❌ Error reading history: {e}")
    else:
        print("\n📈 No training history file found")

    # Check if training process is running
    try:
        import psutil

        python_processes = []
        for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
            try:
                if (
                    proc.info["name"] == "python.exe"
                    and proc.info["memory_info"].rss > 100 * 1024 * 1024
                ):  # >100MB
                    cmdline = " ".join(proc.info["cmdline"])
                    if "pytorch_madden_ocr_trainer" in cmdline:
                        python_processes.append(proc.info)
            except:
                pass

        if python_processes:
            print(f"\n🔄 Training process still running:")
            for proc in python_processes:
                memory_mb = proc["memory_info"].rss / (1024 * 1024)
                print(f"  - PID {proc['pid']}: {memory_mb:.1f} MB")
        else:
            print(f"\n✅ No training process found - training likely completed")

    except ImportError:
        print(f"\n⚠️ psutil not available - can't check running processes")


if __name__ == "__main__":
    check_training_status()

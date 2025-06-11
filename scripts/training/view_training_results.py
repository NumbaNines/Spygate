#!/usr/bin/env python3
"""View YOLOv8 training results."""

import pandas as pd
from pathlib import Path

def view_results():
    """View the training results."""
    results_path = Path("runs/train/hud_detection_20250610_114636/results.csv")
    
    if not results_path.exists():
        print("❌ Results file not found!")
        return
    
    try:
        results = pd.read_csv(results_path)
        print("📊 Final Training Results:")
        print("=" * 40)
        
        last_row = results.iloc[-1]
        
        print(f"✅ Training completed: Epoch {int(last_row['epoch']) + 1}/50")
        print(f"🎯 mAP50: {last_row['metrics/mAP50(B)']:.4f} (50% IoU)")
        print(f"🎯 mAP50-95: {last_row['metrics/mAP50-95(B)']:.4f} (Overall)")
        print(f"🎯 Precision: {last_row['metrics/precision(B)']:.4f}")
        print(f"🎯 Recall: {last_row['metrics/recall(B)']:.4f}")
        print(f"📉 Box Loss: {last_row['train/box_loss']:.4f}")
        print(f"📉 Class Loss: {last_row['train/cls_loss']:.4f}")
        print(f"📉 Val Box Loss: {last_row['val/box_loss']:.4f}")
        print(f"📉 Val Class Loss: {last_row['val/cls_loss']:.4f}")
        
        # Show improvement over training
        first_row = results.iloc[0]
        print(f"\n📈 Improvement over training:")
        print(f"   mAP50: {first_row['metrics/mAP50(B)']:.4f} → {last_row['metrics/mAP50(B)']:.4f}")
        print(f"   Precision: {first_row['metrics/precision(B)']:.4f} → {last_row['metrics/precision(B)']:.4f}")
        print(f"   Recall: {first_row['metrics/recall(B)']:.4f} → {last_row['metrics/recall(B)']:.4f}")
        
        print(f"\n🏆 Best Model Location:")
        print(f"   runs/train/hud_detection_20250610_114636/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    view_results() 
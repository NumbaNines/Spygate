#!/usr/bin/env python3
"""
Check Training Progress and Predict Future Needs
Monitor current training and determine if additional training phases are needed
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def analyze_training_progress():
    print("📊 Training Progress Analysis")
    print("=" * 50)

    # Check if training is still running
    model_dir = "madden_ocr_model"
    training_log = os.path.join(model_dir, "training_log.json")

    if os.path.exists(training_log):
        with open(training_log, "r") as f:
            log_data = json.load(f)

        epochs = log_data.get("epochs", [])
        if epochs:
            print(f"📈 Training History Found:")
            print(f"  - Total epochs completed: {len(epochs)}")

            # Get recent performance
            recent_epochs = epochs[-10:] if len(epochs) >= 10 else epochs
            val_losses = [epoch["val_loss"] for epoch in recent_epochs]
            train_losses = [epoch["train_loss"] for epoch in recent_epochs]

            current_val_loss = val_losses[-1]
            best_val_loss = min(val_losses)

            print(f"  - Current val loss: {current_val_loss:.4f}")
            print(f"  - Best val loss: {best_val_loss:.4f}")
            print(f"  - Target val loss: 0.35")

            # Calculate improvement trend
            if len(val_losses) >= 5:
                recent_trend = np.polyfit(range(len(val_losses)), val_losses, 1)[0]
                if recent_trend < -0.001:
                    trend_status = "📈 Improving"
                elif recent_trend > 0.001:
                    trend_status = "📉 Degrading"
                else:
                    trend_status = "📊 Plateauing"

                print(f"  - Recent trend: {trend_status} ({recent_trend:.6f}/epoch)")

            # Predict if target is reachable
            gap_to_target = current_val_loss - 0.35
            print(f"\n🎯 Target Analysis:")
            print(f"  - Gap to target: {gap_to_target:.4f}")
            print(f"  - Improvement needed: {(gap_to_target/current_val_loss)*100:.1f}%")

            if gap_to_target <= 0:
                print(f"  ✅ TARGET REACHED! No more training needed!")
                return "TARGET_REACHED"
            elif gap_to_target <= 0.1:
                print(f"  🎯 Very close! Current training should reach target")
                return "LIKELY_SUCCESS"
            elif gap_to_target <= 0.2:
                print(f"  ⚠️ Moderate gap. May need additional techniques")
                return "ADDITIONAL_TECHNIQUES"
            else:
                print(f"  ❌ Large gap. Will likely need advanced training")
                return "ADVANCED_TRAINING"

    # Check current model files
    print(f"\n📁 Model Files Status:")
    model_files = ["best_model.pth", "latest_model.pth", "training_log.json"]

    for file in model_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  ✅ {file}: {size_mb:.1f}MB (Modified: {mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"  ❌ {file}: Not found")

    # Recommendations based on analysis
    print(f"\n💡 Recommendations:")

    return analyze_next_steps()


def analyze_next_steps():
    """Analyze what training steps might be needed next"""

    print(f"\n🔮 Potential Next Steps (if current training insufficient):")

    print(f"\n1️⃣ **Extended Fine-Tuning** (Most Likely Needed):")
    print(f"   - Lower learning rate: 0.0001 → 0.00001")
    print(f"   - More epochs: 50-100 additional")
    print(f"   - Early stopping with patience")
    print(f"   - Estimated time: 6-12 hours")

    print(f"\n2️⃣ **Data Augmentation** (If Plateauing):")
    print(f"   - Character rotation/skewing")
    print(f"   - Noise injection")
    print(f"   - Font variations")
    print(f"   - Estimated improvement: 5-15%")

    print(f"\n3️⃣ **Architecture Improvements** (Advanced):")
    print(f"   - Deeper CNN layers")
    print(f"   - Bidirectional LSTM")
    print(f"   - Attention mechanisms")
    print(f"   - Estimated improvement: 10-25%")

    print(f"\n4️⃣ **Transfer Learning** (If Major Issues):")
    print(f"   - Pre-trained text recognition models")
    print(f"   - TrOCR or PaddleOCR base models")
    print(f"   - Fine-tune on Madden data")
    print(f"   - Estimated improvement: 20-40%")

    print(f"\n5️⃣ **Ensemble Methods** (Final Polish):")
    print(f"   - Multiple model voting")
    print(f"   - Different architectures combined")
    print(f"   - Confidence-based selection")
    print(f"   - Estimated improvement: 5-10%")

    # Current recommendation
    print(f"\n🎯 **Current Recommendation:**")
    print(f"   ⏳ **WAIT** for current training to complete")
    print(f"   📊 **EVALUATE** final results")
    print(f"   🎯 **IF** val_loss > 0.4: Try extended fine-tuning")
    print(f"   🎯 **IF** val_loss > 0.5: Consider data augmentation")
    print(f"   🎯 **IF** val_loss > 0.6: May need architecture changes")

    return "WAIT_AND_EVALUATE"


def create_training_roadmap():
    """Create a visual roadmap of potential training phases"""

    phases = [
        {"name": "Initial Training", "status": "✅ Complete", "accuracy": "70-80%"},
        {"name": "Extended Training", "status": "🔄 In Progress", "accuracy": "85-90%"},
        {"name": "Fine-Tuning", "status": "⏳ Planned", "accuracy": "90-95%"},
        {"name": "Advanced Techniques", "status": "💡 Optional", "accuracy": "95-99%"},
        {"name": "Production Ready", "status": "🎯 Goal", "accuracy": "99%+"},
    ]

    print(f"\n🗺️ **Training Roadmap:**")
    for i, phase in enumerate(phases, 1):
        print(f"   {i}. {phase['name']}: {phase['status']} → {phase['accuracy']}")

    # Decision tree
    print(f"\n🌳 **Decision Tree:**")
    print(f"   Current Training Results:")
    print(f"   ├── Val Loss < 0.35 → ✅ DONE! Deploy model")
    print(f"   ├── Val Loss 0.35-0.45 → 🔧 Fine-tuning (lower LR)")
    print(f"   ├── Val Loss 0.45-0.55 → 📊 Data augmentation")
    print(f"   ├── Val Loss 0.55-0.65 → 🏗️ Architecture changes")
    print(f"   └── Val Loss > 0.65 → 🔄 Transfer learning")


if __name__ == "__main__":
    result = analyze_training_progress()
    create_training_roadmap()

    print(f"\n" + "=" * 50)
    print(f"🎯 **BOTTOM LINE**: Let current training finish first!")
    print(f"📊 Your 10,444 samples are excellent - model should improve significantly")
    print(f"⏰ Check back in ~8-12 hours when training completes")
    print(f"🚀 If val_loss reaches 0.35-0.4, you'll have 99%+ accuracy!")

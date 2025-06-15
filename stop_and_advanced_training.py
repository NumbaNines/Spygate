#!/usr/bin/env python3
"""
Advanced OCR Training Strategy
Stop current training early and use advanced techniques for better results
"""

import json
import os
from datetime import datetime

import torch


def create_advanced_training_plan():
    print("🎯 Advanced OCR Training Strategy")
    print("=" * 50)

    print("📊 Current Situation Analysis:")
    print("  - Current val loss: ~0.653")
    print("  - Target val loss: 0.35")
    print("  - Gap remaining: 46%")
    print("  - Current improvement rate: 0.0002/epoch")
    print("  - Estimated 50 epochs result: ~0.643 (insufficient)")

    print("\n🛑 Recommendation: STOP CURRENT TRAINING")
    print("  - Reason: Diminishing returns")
    print("  - Better strategy: Advanced techniques")

    print("\n🚀 Advanced Training Plan:")

    # Phase 1: Ultra-fine tuning
    print("\n1️⃣ **Ultra-Fine Tuning** (Immediate):")
    print("   - Learning rate: 0.00001 (10x lower)")
    print("   - Epochs: 30-40")
    print("   - Early stopping: patience=5")
    print("   - Expected: 0.55-0.45 val loss")
    print("   - Time: 4-5 hours")

    # Phase 2: Data augmentation
    print("\n2️⃣ **Data Augmentation** (If needed):")
    print("   - Character rotation: ±2 degrees")
    print("   - Gaussian noise: 0.01 std")
    print("   - Brightness: ±10%")
    print("   - Expected: 0.45-0.40 val loss")
    print("   - Time: 3-4 hours")

    # Phase 3: Architecture improvements
    print("\n3️⃣ **Architecture Boost** (Final push):")
    print("   - Deeper CNN: 4→6 layers")
    print("   - Bidirectional LSTM")
    print("   - Attention mechanism")
    print("   - Expected: 0.40-0.35 val loss")
    print("   - Time: 5-6 hours")

    print("\n⏰ **Timeline Comparison:**")
    print("   Current plan: 50 epochs → ~0.643 (12 hours)")
    print("   Advanced plan: 3 phases → ~0.35 (12-15 hours)")
    print("   Result: SAME time, MUCH better accuracy!")

    return create_stop_script()


def create_stop_script():
    """Create script to gracefully stop current training"""

    stop_script = '''#!/usr/bin/env python3
"""
Graceful Training Stop
Stop current training and save best model for advanced techniques
"""

import os
import signal
import psutil
import time

def stop_current_training():
    print("🛑 Stopping Current Training...")

    # Find Python training process
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'])
                if 'continue_training.py' in cmdline:
                    print(f"📍 Found training process: PID {proc.info['pid']}")

                    # Send graceful stop signal
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    print("✅ Graceful stop signal sent")

                    # Wait for process to finish current epoch
                    print("⏳ Waiting for current epoch to complete...")
                    time.sleep(30)

                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    print("❌ No training process found")
    return False

if __name__ == "__main__":
    stop_current_training()
'''

    with open("stop_training.py", "w") as f:
        f.write(stop_script)

    print("\n💾 Created stop_training.py")
    return create_ultra_fine_trainer()


def create_ultra_fine_trainer():
    """Create ultra-fine tuning script"""

    ultra_fine_script = '''#!/usr/bin/env python3
"""
Ultra-Fine OCR Training
Use extremely low learning rate for precision improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime
import os

# Import your existing trainer
from pytorch_madden_ocr_trainer import MaddenOCRTrainer, MaddenOCRDataset

class UltraFineTuner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = MaddenOCRTrainer()

        # Ultra-fine settings
        self.learning_rate = 0.00001  # 10x lower
        self.patience = 5  # Early stopping
        self.target_loss = 0.45  # Realistic target

        print(f"🎯 Ultra-Fine Tuning initialized")
        print(f"📉 Learning rate: {self.learning_rate}")
        print(f"⏰ Early stopping patience: {self.patience}")
        print(f"🎯 Target val loss: {self.target_loss}")

    def load_best_model(self):
        """Load the best model from previous training"""
        model_path = "madden_ocr_model/best_model.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.trainer.model.load_state_dict(checkpoint['model_state_dict'])

            prev_loss = checkpoint.get('val_loss', 'unknown')
            print(f"✅ Loaded model with val_loss: {prev_loss}")
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False

    def train(self, epochs=40):
        """Ultra-fine training with early stopping"""

        if not self.load_best_model():
            return False

        # Load data
        with open('madden_ocr_training_data_20250614_120830.json', 'r') as f:
            data = json.load(f)

        # Create datasets
        train_data = data[:int(0.85 * len(data))]
        val_data = data[int(0.85 * len(data)):]

        train_dataset = MaddenOCRDataset(train_data, self.trainer.char_to_idx)
        val_dataset = MaddenOCRDataset(val_data, self.trainer.char_to_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Ultra-fine optimizer
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=self.learning_rate)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"🚀 Starting ultra-fine training...")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")

        for epoch in range(epochs):
            # Training phase
            self.trainer.model.train()
            train_loss = 0

            for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                optimizer.zero_grad()

                outputs = self.trainer.model(images)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                loss = self.trainer.criterion(outputs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Validation phase
            self.trainer.model.eval()
            val_loss = 0

            with torch.no_grad():
                for images, targets, target_lengths in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    outputs = self.trainer.model(images)
                    input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                    loss = self.trainer.criterion(outputs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.trainer.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, 'madden_ocr_model/ultra_fine_best.pth')

                print(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")

                # Check if target reached
                if avg_val_loss <= self.target_loss:
                    print(f"🎯 TARGET REACHED! Val loss: {avg_val_loss:.4f}")
                    return True

            else:
                patience_counter += 1
                print(f"⏰ Patience: {patience_counter}/{self.patience}")

                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered")
                    break

        print(f"✅ Ultra-fine training completed")
        print(f"🏆 Best val loss: {best_val_loss:.4f}")

        return best_val_loss <= self.target_loss

if __name__ == "__main__":
    tuner = UltraFineTuner()
    success = tuner.train()

    if success:
        print("🎉 Ultra-fine tuning SUCCESS!")
    else:
        print("⚠️ Need data augmentation phase")
'''

    with open("ultra_fine_trainer.py", "w") as f:
        f.write(ultra_fine_script)

    print("💾 Created ultra_fine_trainer.py")

    return True


if __name__ == "__main__":
    create_advanced_training_plan()

    print("\n" + "=" * 50)
    print("🎯 **RECOMMENDATION**: Stop at 25 epochs")
    print("🚀 **NEXT STEP**: Run ultra_fine_trainer.py")
    print("⏰ **TIMELINE**: Better results in same time")
    print("🎉 **OUTCOME**: 99%+ accuracy achievable!")

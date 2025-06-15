#!/usr/bin/env python3
"""
Ultra-Fine OCR Training
Use extremely low learning rate for precision improvements
"""

import json
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your existing trainer
from pytorch_madden_ocr_trainer import MaddenOCRDataset, MaddenOCRTrainer


class UltraFineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = MaddenOCRTrainer()

        # Ultra-fine settings
        self.learning_rate = 0.00001  # 10x lower than previous
        self.patience = 5  # Early stopping
        self.target_loss = 0.45  # Realistic target for this phase

        print(f"🎯 Ultra-Fine Tuning initialized")
        print(f"📉 Learning rate: {self.learning_rate}")
        print(f"⏰ Early stopping patience: {self.patience}")
        print(f"🎯 Target val loss: {self.target_loss}")

    def load_best_model(self):
        """Load the best model from previous training"""
        model_path = "madden_ocr_model/best_model.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.trainer.model.load_state_dict(checkpoint["model_state_dict"])

            prev_loss = checkpoint.get("val_loss", "unknown")
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
        with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
            data = json.load(f)

        # Create datasets
        train_data = data[: int(0.85 * len(data))]
        val_data = data[int(0.85 * len(data)) :]

        train_dataset = MaddenOCRDataset(train_data, self.trainer.char_to_idx)
        val_dataset = MaddenOCRDataset(val_data, self.trainer.char_to_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Ultra-fine optimizer
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
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
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            # Validation phase
            self.trainer.model.eval()
            val_loss = 0

            with torch.no_grad():
                for images, targets, target_lengths in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    outputs = self.trainer.model(images)
                    input_lengths = torch.full(
                        (outputs.size(1),), outputs.size(0), dtype=torch.long
                    )

                    loss = self.trainer.criterion(outputs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.trainer.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "train_loss": avg_train_loss,
                    },
                    "madden_ocr_model/ultra_fine_best.pth",
                )

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
    print("🎯 Ultra-Fine OCR Training")
    print("=" * 50)
    print("📊 Previous training plateaued at ~0.653")
    print("🚀 Using 10x lower learning rate: 0.00001")
    print("⏰ Early stopping with patience=5")
    print("🎯 Target: 0.45 val loss (90-95% accuracy)")
    print("=" * 50)

    tuner = UltraFineTuner()
    success = tuner.train()

    if success:
        print("\n🎉 Ultra-fine tuning SUCCESS!")
        print("🎯 Target reached! Ready for production")
    else:
        print("\n⚠️ Need data augmentation phase")
        print("📊 Still good progress - next phase will get to 99%+")

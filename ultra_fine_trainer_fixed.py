#!/usr/bin/env python3
"""
FIXED Ultra-Fine OCR Training
Corrected validation loss calculation bug
"""

import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CRNN(nn.Module):
    """CRNN model for OCR - EXACT match to original trainer"""

    def __init__(self, num_classes, img_height=64, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN feature extractor - EXACT match to original
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 1 channel input (grayscale)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x32x128
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x16x64
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x8x32
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512x4x16
        )

        # RNN layers - EXACT match to original
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)

        # Output layer - EXACT match to original
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN feature extraction
        conv_out = self.cnn(x)  # [batch, 512, 4, 16]

        # Reshape for RNN: [batch, width, features]
        batch_size, channels, height, width = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        conv_out = conv_out.contiguous().view(batch_size, width, channels * height)

        # RNN processing
        rnn_out, _ = self.rnn(conv_out)  # [batch, width, hidden_size*2]

        # Output layer
        output = self.fc(rnn_out)  # [batch, width, num_classes]

        return output


class MaddenOCRDataset(Dataset):
    """Dataset for Madden OCR training - EXACT match to original"""

    def __init__(self, data, char_to_idx, img_height=64, img_width=256):
        self.data = data
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load image from path (EXACT match to original)
        image_path = sample["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to grayscale (EXACT match to original)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing pipeline (EXACT match to original)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.resize(image, (self.img_width, self.img_height))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

        # Process text
        text = sample["ground_truth_text"]
        target = [self.char_to_idx.get(char, 0) for char in text]
        target = torch.LongTensor(target)

        return image, target, len(target)


class UltraFineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ultra-fine settings
        self.learning_rate = 0.00001  # 10x lower than previous
        self.patience = 5  # Early stopping
        self.target_loss = 0.45  # Realistic target for this phase

        print(f"🎯 Ultra-Fine Training Setup")
        print(f"  - Device: {self.device}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Target Loss: {self.target_loss}")
        print(f"  - Early Stopping Patience: {self.patience}")

    def load_data(self):
        """Load and prepare training data"""
        print("📂 Loading training data...")

        with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
            data = json.load(f)

        # Create character mapping
        all_chars = set()
        for sample in data:
            all_chars.update(sample["ground_truth_text"])

        chars = sorted(list(all_chars))
        char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # +1 for blank
        char_to_idx["<blank>"] = 0
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}

        print(f"📊 Character set: {len(chars)} unique characters")
        print(f"📊 Total samples: {len(data):,}")

        # Split data
        split_idx = int(0.85 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        print(f"📊 Training samples: {len(train_data):,}")
        print(f"📊 Validation samples: {len(val_data):,}")

        return train_data, val_data, char_to_idx, idx_to_char

    def collate_fn(self, batch):
        """Custom collate function for variable length sequences - EXACT match to original"""
        images, targets, target_lengths = zip(*batch)

        # Stack images
        images = torch.stack(images, 0)

        # Pad text sequences
        from torch.nn.utils.rnn import pad_sequence

        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

        return images, targets, target_lengths

    def load_best_model(self):
        """Load the best model from previous training"""
        model_path = "models/pytorch_madden_ocr/best_model.pth"

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            prev_loss = checkpoint.get("val_loss", "unknown")
            print(f"✅ Loaded model with val_loss: {prev_loss}")
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False

    def validate_model(self, val_loader):
        """FIXED validation function with proper loss calculation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # FIXED: Proper CTC loss calculation
                # outputs: [batch, width, num_classes] -> [width, batch, num_classes] for CTC
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    (outputs.size(1),), outputs.size(0), dtype=torch.long, device=self.device
                )

                # Calculate loss
                loss = self.criterion(outputs, targets, input_lengths, target_lengths)

                # FIXED: Proper loss accumulation
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1

        # FIXED: Return average loss
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self):
        """Ultra-fine training with fixed validation"""
        print("🚀 Starting ultra-fine training...")

        # Load data
        train_data, val_data, char_to_idx, idx_to_char = self.load_data()

        # Create datasets
        train_dataset = MaddenOCRDataset(train_data, char_to_idx)
        val_dataset = MaddenOCRDataset(val_data, char_to_idx)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_fn
        )

        # Initialize model
        num_classes = len(char_to_idx)
        self.model = CRNN(num_classes).to(self.device)

        # Load previous best model
        if not self.load_best_model():
            print("❌ Could not load previous model. Exiting.")
            return

        # Setup training
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(40):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)

                # CTC loss calculation (EXACT match to original)
                # outputs: [batch, width, num_classes] -> [width, batch, num_classes] for CTC
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    (outputs.size(1),), outputs.size(0), dtype=torch.long, device=self.device
                )

                loss = self.criterion(outputs, targets, input_lengths, target_lengths)

                # Backward pass
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1

                # Progress logging
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch+1}/40, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            # FIXED: Validation phase
            val_loss = self.validate_model(val_loader)
            avg_train_loss = train_loss / max(num_batches, 1)

            print(
                f"Epoch {epoch+1}/40 - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save model
                os.makedirs("models/pytorch_madden_ocr_ultra", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "epoch": epoch,
                        "char_to_idx": char_to_idx,
                        "idx_to_char": idx_to_char,
                    },
                    "models/pytorch_madden_ocr_ultra/best_model.pth",
                )

                print(f"💾 New best model saved! Val Loss: {val_loss:.4f}")

                # Check if target reached
                if val_loss <= self.target_loss:
                    print(f"🎯 Target loss {self.target_loss} reached! Stopping training.")
                    break
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.patience}")

                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered. Best val loss: {best_val_loss:.4f}")
                    break

        print(f"✅ Ultra-fine training completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    trainer = UltraFineTuner()
    trainer.train()

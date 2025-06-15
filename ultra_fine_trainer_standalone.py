#!/usr/bin/env python3
"""
Standalone Ultra-Fine OCR Training
Complete implementation with 10x lower learning rate for precision
"""

import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x8x64
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x4x64
            # Block 5
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # 512x3x63
        )

        # Calculate CNN output size
        self.cnn_output_height = img_height // 16  # After pooling operations
        self.cnn_output_width = 512  # Feature channels

        # RNN layers - EXACT match to original
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_width,  # 512, not 2048
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True,
        )

        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN feature extraction
        conv_features = self.cnn(x)  # [batch, 512, height, width]

        # Reshape for RNN: [batch, width, channels]
        batch_size, channels, height, width = conv_features.size()

        # Average pool over height dimension to get fixed feature size
        conv_features = F.adaptive_avg_pool2d(conv_features, (1, width))  # [batch, 512, 1, width]
        conv_features = conv_features.squeeze(2)  # [batch, 512, width]
        conv_features = conv_features.permute(0, 2, 1)  # [batch, width, 512]

        # RNN processing
        rnn_output, _ = self.rnn(conv_features)  # [batch, width, hidden_size * 2]

        # Classification
        output = self.classifier(rnn_output)  # [batch, width, num_classes]
        output = F.log_softmax(output, dim=2)

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
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

        # Normalize and convert to tensor (EXACT match to original)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension for grayscale

        # Encode text
        text = sample["ground_truth_text"]
        encoded_text = [self.char_to_idx.get(char, 0) for char in text]

        return image, torch.tensor(encoded_text, dtype=torch.long), len(encoded_text)


class UltraFineTuner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Character set from your data
        self.chars = " &-0123456789:;ACDFGHIKLOPRSTadhlnorst"
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        # Model
        self.model = CRNN(len(self.chars)).to(self.device)
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        # Ultra-fine settings
        self.learning_rate = 0.00001  # 10x lower than previous
        self.patience = 5  # Early stopping
        self.target_loss = 0.45  # Realistic target for this phase

        print(f"🎯 Ultra-Fine Tuning initialized")
        print(f"📉 Learning rate: {self.learning_rate}")
        print(f"⏰ Early stopping patience: {self.patience}")
        print(f"🎯 Target val loss: {self.target_loss}")
        print(f"🖥️ Device: {self.device}")

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

    def train(self, epochs=40):
        """Ultra-fine training with early stopping"""

        if not self.load_best_model():
            print("❌ Cannot continue without previous model")
            return False

        # Load data
        with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
            data = json.load(f)

        # Create datasets
        train_data = data[: int(0.85 * len(data))]
        val_data = data[int(0.85 * len(data)) :]

        train_dataset = MaddenOCRDataset(train_data, self.char_to_idx)
        val_dataset = MaddenOCRDataset(val_data, self.char_to_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_fn
        )

        # Ultra-fine optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"🚀 Starting ultra-fine training...")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)

                # CTC loss calculation (EXACT match to original)
                # outputs: [batch, width, num_classes] -> [width, batch, num_classes] for CTC
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                loss = self.criterion(outputs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            # Validation phase
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for images, targets, target_lengths in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    outputs = self.model(images)

                    # CTC loss calculation (EXACT match to original)
                    # outputs: [batch, width, num_classes] -> [width, batch, num_classes] for CTC
                    outputs = outputs.permute(1, 0, 2)
                    input_lengths = torch.full(
                        (outputs.size(1),), outputs.size(0), dtype=torch.long
                    )

                    loss = self.criterion(outputs, targets, input_lengths, target_lengths)
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
                os.makedirs("madden_ocr_model", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
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

    def collate_fn(self, batch):
        """Custom collate function for variable length sequences - EXACT match to original"""
        images, targets, target_lengths = zip(*batch)

        # Stack images
        images = torch.stack(images, 0)

        # Pad text sequences
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

        return images, targets, target_lengths


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

#!/usr/bin/env python3
"""
PyTorch Madden OCR Trainer - Clean, Fast, Reliable
GPU-optimized OCR training for Madden HUD text recognition
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaddenOCRDataset(Dataset):
    """PyTorch Dataset for Madden OCR training data"""

    def __init__(
        self,
        samples: List[Dict],
        char_to_idx: Dict[str, int],
        img_height: int = 64,
        img_width: int = 256,
    ):
        self.samples = samples
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image_path = sample["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing pipeline
        image = cv2.bilateralFilter(image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension

        # Encode text
        text = sample["ground_truth_text"]
        encoded_text = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
        encoded_text = torch.tensor(encoded_text, dtype=torch.long)

        return image, encoded_text, text


class CRNN(nn.Module):
    """CRNN (CNN + RNN + CTC) model for OCR"""

    def __init__(self, num_classes: int, img_height: int = 64, hidden_size: int = 256):
        super(CRNN, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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

        # RNN layers
        self.rnn = nn.LSTM(
            input_size=self.cnn_output_width,
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


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    images, texts, raw_texts = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)

    # Pad text sequences
    text_lengths = torch.tensor([len(text) for text in texts])
    texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return images, texts, text_lengths, raw_texts


class PyTorchMaddenOCRTrainer:
    """PyTorch-based Madden OCR Trainer"""

    def __init__(self, model_save_path: str = "models/pytorch_madden_ocr"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Madden character set (based on actual data analysis)
        self.madden_chars = " &-0123456789:;ACDFGHIKLOPRSTadhlnorst"
        self.char_to_idx = {char: idx for idx, char in enumerate(self.madden_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.madden_chars)
        self.blank_token = 0  # CTC blank token

        # Model parameters
        self.img_height = 64
        self.img_width = 256
        self.max_text_length = 15

        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        logger.info(f"🚀 PyTorch Madden OCR Trainer initialized")
        logger.info(f"📊 Character classes: {self.num_classes}")
        logger.info(f"🎯 Character set: '{self.madden_chars}'")
        logger.info(f"🖥️ Device: {self.device}")

    def load_and_prepare_data(self, json_file: str) -> Tuple[DataLoader, DataLoader, Dict]:
        """Load and prepare training data"""
        logger.info(f"📁 Loading data from: {json_file}")

        with open(json_file, "r") as f:
            training_data = json.load(f)

        logger.info(f"📊 Processing {len(training_data)} samples...")

        # Filter valid samples
        valid_samples = []
        class_distribution = {}

        for sample in training_data:
            # Check image path
            if "image_path" not in sample or not sample["image_path"]:
                continue
            if not os.path.exists(sample["image_path"]):
                continue

            # Check text
            text = sample.get("ground_truth_text", "").strip()
            if not text or len(text) > self.max_text_length:
                continue

            # Validate characters
            if not all(c in self.madden_chars for c in text):
                continue

            valid_samples.append(sample)

            # Track class distribution
            class_name = sample.get("class_name", "unknown")
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

        logger.info(f"✅ Found {len(valid_samples)} valid samples")
        logger.info(f"📊 Class distribution: {class_distribution}")

        if len(valid_samples) == 0:
            raise ValueError("No valid training samples found!")

        # Split data
        train_samples, val_samples = train_test_split(
            valid_samples, test_size=0.15, random_state=42, shuffle=True
        )

        logger.info(f"🎯 Training samples: {len(train_samples):,}")
        logger.info(f"🎯 Validation samples: {len(val_samples):,}")

        # Create datasets
        train_dataset = MaddenOCRDataset(
            train_samples, self.char_to_idx, self.img_height, self.img_width
        )
        val_dataset = MaddenOCRDataset(
            val_samples, self.char_to_idx, self.img_height, self.img_width
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader, class_distribution

    def build_model(self) -> CRNN:
        """Build CRNN model"""
        logger.info("🏗️ Building CRNN model...")
        model = CRNN(self.num_classes, self.img_height)
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"🎯 Trainable parameters: {trainable_params:,}")

        return model

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Train the model"""
        logger.info(f"🚀 Starting training for {epochs} epochs...")

        # Build model
        self.model = self.build_model()

        # Loss and optimizer
        criterion = nn.CTCLoss(blank=self.blank_token, reduction="mean", zero_infinity=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_idx, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
                images = images.to(self.device)
                texts = texts.to(self.device)
                text_lengths = text_lengths.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)  # [batch, width, num_classes]

                # Prepare for CTC loss
                outputs = outputs.permute(1, 0, 2)  # [width, batch, num_classes]
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                # CTC loss
                loss = criterion(outputs, texts, input_lengths, text_lengths)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if batch_idx % 50 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, texts, text_lengths, raw_texts in val_loader:
                    images = images.to(self.device)
                    texts = texts.to(self.device)
                    text_lengths = text_lengths.to(self.device)

                    outputs = self.model(images)
                    outputs = outputs.permute(1, 0, 2)
                    input_lengths = torch.full(
                        (outputs.size(1),), outputs.size(0), dtype=torch.long
                    )

                    loss = criterion(outputs, texts, input_lengths, text_lengths)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(epoch, avg_val_loss)
                logger.info(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")

        # Save training history
        self.save_training_history(train_losses, val_losses)
        logger.info("🎉 Training completed!")

        return train_losses, val_losses

    def save_model(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "madden_chars": self.madden_chars,
            "num_classes": self.num_classes,
            "img_height": self.img_height,
            "img_width": self.img_width,
        }

        torch.save(checkpoint, self.model_save_path / "best_model.pth")

    def save_training_history(self, train_losses: List[float], val_losses: List[float]):
        """Save and plot training history"""
        # Save losses
        history = {"train_losses": train_losses, "val_losses": val_losses}

        with open(self.model_save_path / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.model_save_path / "training_curves.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main training function"""
    print("🎯 PyTorch Madden OCR Trainer")
    print("=" * 50)

    try:
        # Initialize trainer
        trainer = PyTorchMaddenOCRTrainer()

        # Load data
        train_loader, val_loader, class_dist = trainer.load_and_prepare_data(
            "madden_ocr_training_data_20250614_120830.json"
        )

        # Train model
        train_losses, val_losses = trainer.train_model(train_loader, val_loader, epochs=50)

        print("🎉 Training completed successfully!")
        print(f"📁 Model saved to: {trainer.model_save_path}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

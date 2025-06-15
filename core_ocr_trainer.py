#!/usr/bin/env python3
"""
Optimized OCR trainer for core dataset.
Fast, efficient training on essential patterns only.
"""

import json
import os
import string

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CoreOCRDataset(Dataset):
    def __init__(self, data, char_to_idx, max_length=20):
        self.data = data
        self.char_to_idx = char_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, image_path):
        """Enhanced preprocessing for dark HUD regions."""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros((32, 128), dtype=np.float32)

            # Resize to standard size
            img = cv2.resize(img, (128, 32))

            # Enhanced preprocessing for dark text
            # 1. Brightness boost
            img = cv2.convertScaleAbs(img, alpha=2.0, beta=30)

            # 2. CLAHE for local contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            img = clahe.apply(img)

            # 3. Gamma correction
            gamma = 1.5
            img = np.power(img / 255.0, 1.0 / gamma) * 255.0
            img = img.astype(np.uint8)

            # 4. Bilateral filter for noise reduction
            img = cv2.bilateralFilter(img, 5, 50, 50)

            # Normalize
            img = img.astype(np.float32) / 255.0

            return img

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros((32, 128), dtype=np.float32)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process image
        image = self.preprocess_image(item["image_path"])
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

        # Process text
        text = item["ground_truth_text"]

        # Convert to indices
        indices = [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in text]

        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.char_to_idx["<PAD>"]] * (self.max_length - len(indices)))
        else:
            indices = indices[: self.max_length]

        target = torch.LongTensor(indices)

        return image, target, len(text)


class CoreOCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(CoreOCRModel, self).__init__()

        # CNN feature extractor (optimized for 128x32 images)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x8
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 16x8
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 8x8
        )

        # RNN for sequence modeling
        self.rnn = nn.LSTM(
            256 * 2, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.1
        )

        # Output layer
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # CNN features
        features = self.cnn(x)  # [B, 256, 8, 8]

        # Reshape for RNN
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2)  # [B, W, C, H]
        features = features.reshape(B, W, C * H)  # [B, W, C*H]

        # RNN
        rnn_out, _ = self.rnn(features)  # [B, W, hidden_size*2]
        rnn_out = self.dropout(rnn_out)

        # Classification
        output = self.classifier(rnn_out)  # [B, W, vocab_size]

        return output


def train_core_ocr():
    print("🚀 Training Core OCR Model")
    print("=" * 50)

    # Load core dataset
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    print(f"📊 Core dataset: {len(data):,} samples")

    # Build vocabulary from core patterns only
    all_chars = set()
    for item in data:
        all_chars.update(item["ground_truth_text"])

    # Add special tokens
    vocab = ["<PAD>", "<UNK>"] + sorted(list(all_chars))
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"📝 Vocabulary size: {len(vocab)}")
    print(f"📝 Characters: {''.join(sorted(all_chars))}")

    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = CoreOCRDataset(train_data, char_to_idx)
    val_dataset = CoreOCRDataset(val_data, char_to_idx)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    model = CoreOCRModel(len(vocab), hidden_size=256, num_layers=2).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx["<PAD>"])

    # Training loop
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30")
        for images, targets, lengths in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, len(vocab))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Accuracy calculation
            pred = outputs.argmax(dim=1)
            mask = targets != char_to_idx["<PAD>"]
            train_correct += (pred == targets)[mask].sum().item()
            train_total += mask.sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.3f}"}
            )

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets, lengths in val_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                outputs_flat = outputs.reshape(-1, len(vocab))
                targets_flat = targets.reshape(-1)

                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()

                pred = outputs_flat.argmax(dim=1)
                mask = targets_flat != char_to_idx["<PAD>"]
                val_correct += (pred == targets_flat)[mask].sum().item()
                val_total += mask.sum().item()

        scheduler.step()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "vocab_size": len(vocab),
                },
                "madden_core_ocr_model.pth",
            )
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    print(f"\n🎉 Training complete!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Model saved: madden_core_ocr_model.pth")


if __name__ == "__main__":
    train_core_ocr()

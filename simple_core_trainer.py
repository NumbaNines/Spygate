#!/usr/bin/env python3
"""
Simple, working OCR trainer for core dataset.
Uses a straightforward CNN approach.
"""

import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SimpleOCRDataset(Dataset):
    def __init__(self, data, char_to_idx, max_length=15):
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
            img = cv2.convertScaleAbs(img, alpha=2.5, beta=40)

            # 2. CLAHE for local contrast
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            img = clahe.apply(img)

            # 3. Gamma correction
            gamma = 1.2
            img = np.power(img / 255.0, 1.0 / gamma) * 255.0
            img = img.astype(np.uint8)

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

        # Process text - encode as single sequence
        text = item["ground_truth_text"]

        # Convert to indices
        indices = [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in text]

        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.char_to_idx["<PAD>"]] * (self.max_length - len(indices)))
        else:
            indices = indices[: self.max_length]

        target = torch.LongTensor(indices)

        return image, target


class SimpleOCRModel(nn.Module):
    def __init__(self, vocab_size, max_length=15):
        super(SimpleOCRModel, self).__init__()
        self.max_length = max_length

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Classifier for each character position
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, vocab_size * max_length)
        )

        self.vocab_size = vocab_size

    def forward(self, x):
        # CNN features
        features = self.cnn(x)  # [B, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 256]

        # Classify each position
        output = self.classifier(features)  # [B, vocab_size * max_length]
        output = output.view(-1, self.max_length, self.vocab_size)  # [B, max_length, vocab_size]

        return output


def train_simple_ocr():
    print("🚀 Training Simple Core OCR Model")
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
    max_length = 15
    train_dataset = SimpleOCRDataset(train_data, char_to_idx, max_length)
    val_dataset = SimpleOCRDataset(val_data, char_to_idx, max_length)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    model = SimpleOCRModel(len(vocab), max_length).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx["<PAD>"])

    # Training loop
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(15):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/15")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, max_length, vocab_size]

            # Calculate loss for each position
            loss = 0
            for pos in range(max_length):
                loss += criterion(outputs[:, pos, :], targets[:, pos])
            loss /= max_length

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # Accuracy calculation
            pred = outputs.argmax(dim=2)  # [B, max_length]
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
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)

                # Calculate loss
                loss = 0
                for pos in range(max_length):
                    loss += criterion(outputs[:, pos, :], targets[:, pos])
                loss /= max_length
                val_loss += loss.item()

                # Accuracy
                pred = outputs.argmax(dim=2)
                mask = targets != char_to_idx["<PAD>"]
                val_correct += (pred == targets)[mask].sum().item()
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "vocab_size": len(vocab),
                    "max_length": max_length,
                },
                "madden_simple_core_ocr_model.pth",
            )
            print(f"✅ Saved best model (val_acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    print(f"\n🎉 Training complete!")
    print(f"📊 Best validation accuracy: {best_val_acc:.3f}")
    print(f"💾 Model saved: madden_simple_core_ocr_model.pth")


if __name__ == "__main__":
    train_simple_ocr()

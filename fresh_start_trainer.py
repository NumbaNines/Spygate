#!/usr/bin/env python3
"""
Fresh Start OCR Trainer - Expert Solution
"""

import json
import os
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 64, hidden_size: int = 256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv_features = self.cnn(x)

        batch_size, channels, height, width = conv_features.size()
        conv_features = F.adaptive_avg_pool2d(conv_features, (1, width))
        conv_features = conv_features.squeeze(2)
        conv_features = conv_features.permute(0, 2, 1)

        rnn_output, _ = self.rnn(conv_features)

        output = self.classifier(rnn_output)
        output = F.log_softmax(output, dim=2)

        return output


class MaddenOCRDataset(Dataset):
    def __init__(self, data, char_to_idx, img_height=64, img_width=256):
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = sample["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.bilateralFilter(image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.resize(image, (self.img_width, self.img_height))

        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)

        text = sample["ground_truth_text"]
        target = [self.char_to_idx[char] for char in text]
        target = torch.LongTensor(target)

        return image, target, len(target)


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)

    images = torch.stack(images, 0)

    from torch.nn.utils.rnn import pad_sequence

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return images, targets, target_lengths


def main():
    print("🚀 Fresh Start Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
        data = json.load(f)

    print(f"📊 Total samples: {len(data):,}")

    # Create character mapping
    all_chars = set()
    for sample in data:
        text = sample["ground_truth_text"]
        all_chars.update(text)

    chars = sorted(list(all_chars))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_idx["<blank>"] = 0

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"📊 Characters: {len(chars)} unique")
    print(f"📊 Total classes: {len(char_to_idx)}")

    # Split data
    split_idx = int(0.85 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create datasets
    train_dataset = MaddenOCRDataset(train_data, char_to_idx)
    val_dataset = MaddenOCRDataset(val_data, char_to_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    num_classes = len(char_to_idx)
    model = CRNN(num_classes).to(device)

    print(f"✅ Model initialized with {num_classes} classes")

    # Setup training
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0.0
        valid_batches = 0

        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs_ctc = outputs.permute(1, 0, 2)
            input_lengths = torch.full(
                (outputs_ctc.size(1),), outputs_ctc.size(0), dtype=torch.long, device=device
            )

            loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                valid_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)
                outputs_ctc = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    (outputs_ctc.size(1),), outputs_ctc.size(0), dtype=torch.long, device=device
                )

                loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1

        avg_train_loss = train_loss / max(valid_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            os.makedirs("models/pytorch_madden_ocr_fresh", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "epoch": epoch,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                },
                "models/pytorch_madden_ocr_fresh/best_model.pth",
            )

            print(f"💾 Best model saved! Val Loss: {avg_val_loss:.4f}")

    print(f"✅ Training completed! Best: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

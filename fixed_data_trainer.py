#!/usr/bin/env python3
"""
FIXED Data Trainer - Solves data quality issues
"""

import json
import os
import random
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class FreshCRNN(nn.Module):
    def __init__(self, vocab_size: int, image_height: int = 64, rnn_hidden: int = 256):
        super(FreshCRNN, self).__init__()

        self.feature_extractor = nn.Sequential(
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

        self.sequence_processor = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True,
        )

        self.output_classifier = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, input_tensor):
        cnn_features = self.feature_extractor(input_tensor)

        batch_size, channels, height, width = cnn_features.size()
        cnn_features = F.adaptive_avg_pool2d(cnn_features, (1, width))
        cnn_features = cnn_features.squeeze(2)
        cnn_features = cnn_features.permute(0, 2, 1)

        sequence_output, _ = self.sequence_processor(cnn_features)

        predictions = self.output_classifier(sequence_output)
        predictions = F.log_softmax(predictions, dim=2)

        return predictions


class FixedDataset(Dataset):
    def __init__(self, samples, character_mapping, height=64, width=256):
        self.samples = samples
        self.char_map = character_mapping
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = sample["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.bilateralFilter(img, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.resize(img, (self.width, self.height))

        img = img.astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0)

        text = sample["ground_truth_text"]
        text_indices = [self.char_map[char] for char in text]
        text_tensor = torch.LongTensor(text_indices)

        return img_tensor, text_tensor, len(text_indices)


def fixed_collate_function(batch):
    images, texts, text_lengths = zip(*batch)

    images = torch.stack(images, 0)

    from torch.nn.utils.rnn import pad_sequence

    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return images, texts, text_lengths


def fix_data_quality(raw_data):
    """Fix the critical data quality issues"""
    print("🔧 FIXING DATA QUALITY ISSUES...")
    print("=" * 50)

    # Step 1: Remove very short texts (≤2 chars)
    print("Step 1: Filtering short texts...")
    filtered_data = [sample for sample in raw_data if len(sample["ground_truth_text"].strip()) > 2]
    print(f"  Before: {len(raw_data):,} samples")
    print(f"  After: {len(filtered_data):,} samples")
    print(f"  Removed: {len(raw_data) - len(filtered_data):,} short texts")

    # Step 2: Balance the dataset (limit overrepresented texts)
    print("\nStep 2: Balancing dataset...")
    text_counter = Counter([sample["ground_truth_text"] for sample in filtered_data])

    # Limit each text to max 200 samples (was 1,162 for "1ST & 10")
    max_per_text = 200
    balanced_data = []
    text_counts = {}

    # Shuffle first to get random samples for each text
    random.shuffle(filtered_data)

    for sample in filtered_data:
        text = sample["ground_truth_text"]
        current_count = text_counts.get(text, 0)

        if current_count < max_per_text:
            balanced_data.append(sample)
            text_counts[text] = current_count + 1

    print(f"  Before balancing: {len(filtered_data):,} samples")
    print(f"  After balancing: {len(balanced_data):,} samples")
    print(f"  Max per text: {max_per_text}")

    # Step 3: CRITICAL - Shuffle the entire dataset
    print("\nStep 3: Shuffling dataset...")
    random.shuffle(balanced_data)
    print("  ✅ Dataset completely shuffled!")

    # Step 4: Verify fix
    print("\nStep 4: Verifying fixes...")
    consecutive_identical = 0
    for i in range(1, min(1000, len(balanced_data))):
        if balanced_data[i]["ground_truth_text"] == balanced_data[i - 1]["ground_truth_text"]:
            consecutive_identical += 1

    print(
        f"  Consecutive identical in first 1000: {consecutive_identical} ({consecutive_identical/10:.1f}%)"
    )

    # Show new distribution
    new_counter = Counter([sample["ground_truth_text"] for sample in balanced_data])
    print(
        f"  Most common text now: '{new_counter.most_common(1)[0][0]}' ({new_counter.most_common(1)[0][1]} times)"
    )

    print("✅ DATA QUALITY FIXED!")
    print("=" * 50)

    return balanced_data


def run_fixed_training():
    print("🔧 FIXED Data Training - Quality Issues Resolved")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🔒 Training ID: {timestamp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Device: {device}")

    # Load and fix data
    with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
        raw_data = json.load(f)

    print(f"📊 Raw samples: {len(raw_data):,}")

    # FIX THE DATA QUALITY ISSUES
    fixed_data = fix_data_quality(raw_data)

    # Create vocabulary from FIXED data
    unique_characters = set()
    for sample in fixed_data:
        text = sample["ground_truth_text"]
        unique_characters.update(text)

    sorted_chars = sorted(list(unique_characters))

    fixed_char_to_idx = {"<CTC_BLANK>": 0}
    for i, char in enumerate(sorted_chars):
        fixed_char_to_idx[char] = i + 1

    fixed_idx_to_char = {idx: char for char, idx in fixed_char_to_idx.items()}

    print(f"📊 Fixed vocabulary: {len(sorted_chars)} characters")
    print(f"📊 Total classes: {len(fixed_char_to_idx)}")

    # Split FIXED data
    split_point = int(0.85 * len(fixed_data))
    train_samples = fixed_data[:split_point]
    val_samples = fixed_data[split_point:]

    print(f"📊 Train samples: {len(train_samples):,}")
    print(f"📊 Val samples: {len(val_samples):,}")

    # Create datasets
    train_dataset = FixedDataset(train_samples, fixed_char_to_idx)
    val_dataset = FixedDataset(val_samples, fixed_char_to_idx)

    # Create data loaders with shuffling
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=fixed_collate_function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=fixed_collate_function
    )

    # Initialize model
    vocab_size = len(fixed_char_to_idx)
    fixed_model = FreshCRNN(vocab_size).to(device)

    print(f"✅ Model initialized with {vocab_size} classes")

    # CONSERVATIVE training setup
    fixed_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    fixed_optimizer = optim.Adam(fixed_model.parameters(), lr=0.0001)
    fixed_scheduler = optim.lr_scheduler.StepLR(fixed_optimizer, step_size=10, gamma=0.5)

    print(f"🔧 FIXED training settings:")
    print(f"  - Learning Rate: 0.0001")
    print(f"  - Batch Size: 16")
    print(f"  - Data: SHUFFLED & BALANCED")
    print(f"  - Quality: FIXED")

    # Create save directory
    save_dir = f"models/fixed_ocr_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Save directory: {save_dir}")

    # Training loop
    best_validation_loss = float("inf")
    patience_counter = 0
    max_patience = 15

    print("\n🔧 Starting FIXED training...")
    print("=" * 60)

    for epoch in range(100):
        # Training phase
        fixed_model.train()
        epoch_train_loss = 0.0
        valid_train_batches = 0

        for batch_idx, (images, texts, text_lengths) in enumerate(train_loader):
            images = images.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)

            fixed_optimizer.zero_grad()

            predictions = fixed_model(images)
            predictions_ctc = predictions.permute(1, 0, 2)
            input_lengths = torch.full(
                (predictions_ctc.size(1),), predictions_ctc.size(0), dtype=torch.long, device=device
            )

            loss = fixed_criterion(predictions_ctc, texts, input_lengths, text_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 100.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fixed_model.parameters(), 0.5)
                fixed_optimizer.step()

                epoch_train_loss += loss.item()
                valid_train_batches += 1
            else:
                print(f"⚠️ Skipping invalid loss: {loss.item()}")

            # Frequent logging
            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1:2d}, Batch {batch_idx:3d}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Validation phase
        fixed_model.eval()
        epoch_val_loss = 0.0
        valid_val_batches = 0

        with torch.no_grad():
            for images, texts, text_lengths in val_loader:
                images = images.to(device)
                texts = texts.to(device)
                text_lengths = text_lengths.to(device)

                predictions = fixed_model(images)
                predictions_ctc = predictions.permute(1, 0, 2)
                input_lengths = torch.full(
                    (predictions_ctc.size(1),),
                    predictions_ctc.size(0),
                    dtype=torch.long,
                    device=device,
                )

                loss = fixed_criterion(predictions_ctc, texts, input_lengths, text_lengths)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    epoch_val_loss += loss.item()
                    valid_val_batches += 1

        # Calculate averages
        avg_train_loss = epoch_train_loss / max(valid_train_batches, 1)
        avg_val_loss = epoch_val_loss / max(valid_val_batches, 1)

        # Learning rate scheduling
        fixed_scheduler.step()
        current_lr = fixed_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.6f}"
        )

        # Save best model
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            patience_counter = 0

            model_path = os.path.join(save_dir, "best_fixed_model.pth")
            torch.save(
                {
                    "model_state_dict": fixed_model.state_dict(),
                    "optimizer_state_dict": fixed_optimizer.state_dict(),
                    "scheduler_state_dict": fixed_scheduler.state_dict(),
                    "validation_loss": avg_val_loss,
                    "epoch": epoch,
                    "char_to_idx": fixed_char_to_idx,
                    "idx_to_char": fixed_idx_to_char,
                    "training_id": timestamp,
                    "vocab_size": vocab_size,
                    "data_fixes_applied": True,
                },
                model_path,
            )

            print(f"💾 NEW BEST MODEL! Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss <= 0.4:
                print(f"🎯 Target loss 0.4 reached! Stopping training.")
                break
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{max_patience}")

            if patience_counter >= max_patience:
                print(f"🛑 Early stopping. Best val loss: {best_validation_loss:.4f}")
                break

    print("\n" + "=" * 60)
    print(f"✅ FIXED training completed!")
    print(f"🏆 Best validation loss: {best_validation_loss:.4f}")
    print(f"💾 Model saved in: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_fixed_training()

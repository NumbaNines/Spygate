#!/usr/bin/env python3
"""
Expert Conservative OCR Trainer
Safe continuation training with layer freezing and gradient control
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
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CRNN(nn.Module):
    """CRNN model - EXACT copy from original"""

    def __init__(self, num_classes: int, img_height: int = 64, hidden_size: int = 256):
        super(CRNN, self).__init__()

        # CNN feature extractor - EXACT match
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # RNN layers - EXACT match
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True,
        )

        # Output layer - EXACT match
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN feature extraction
        conv_features = self.cnn(x)

        # Reshape for RNN
        batch_size, channels, height, width = conv_features.size()
        conv_features = F.adaptive_avg_pool2d(conv_features, (1, width))
        conv_features = conv_features.squeeze(2)
        conv_features = conv_features.permute(0, 2, 1)

        # RNN processing
        rnn_output, _ = self.rnn(conv_features)

        # Classification
        output = self.classifier(rnn_output)
        output = F.log_softmax(output, dim=2)

        return output


class MaddenOCRDataset(Dataset):
    """Dataset - EXACT match to original"""

    def __init__(self, data, char_to_idx, img_height=64, img_width=256):
        self.data = data
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load and preprocess image - EXACT match
        image_path = sample["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocessing pipeline - EXACT match
        image = cv2.bilateralFilter(image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.resize(image, (self.img_width, self.img_height))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)

        # Process text
        text = sample["ground_truth_text"]
        target = [self.char_to_idx.get(char, 0) for char in text]
        target = torch.LongTensor(target)

        return image, target, len(target)


class ExpertConservativeTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # EXPERT CONSERVATIVE SETTINGS
        self.learning_rate = 1e-7  # Ultra-conservative
        self.max_grad_norm = 0.1  # Aggressive gradient clipping
        self.patience = 3  # Quick early stopping
        self.target_loss = 0.6  # Conservative target

        print(f"🎯 Expert Conservative Training Setup")
        print(f"  - Device: {self.device}")
        print(f"  - Learning Rate: {self.learning_rate} (ultra-conservative)")
        print(f"  - Gradient Clipping: {self.max_grad_norm}")
        print(f"  - Target Loss: {self.target_loss}")
        print(f"  - Strategy: Layer freezing + minimal updates")

    def load_data(self):
        """Load data with original character mapping"""
        print("📂 Loading training data...")

        with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
            data = json.load(f)

        # Use EXACT character mapping from saved model
        checkpoint = torch.load(
            "models/pytorch_madden_ocr/best_model.pth", map_location="cpu", weights_only=False
        )
        char_to_idx = checkpoint["char_to_idx"]
        idx_to_char = checkpoint["idx_to_char"]

        print(f"📊 Using saved character mapping: {len(char_to_idx)} classes")
        print(f"📊 Total samples: {len(data):,}")

        # Split data
        split_idx = int(0.85 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        print(f"📊 Training samples: {len(train_data):,}")
        print(f"📊 Validation samples: {len(val_data):,}")

        return train_data, val_data, char_to_idx, idx_to_char

    def collate_fn(self, batch):
        """Custom collate function"""
        images, targets, target_lengths = zip(*batch)

        images = torch.stack(images, 0)

        from torch.nn.utils.rnn import pad_sequence

        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

        return images, targets, target_lengths

    def freeze_layers(self, model):
        """EXPERT STRATEGY: Freeze CNN and RNN, only train classifier"""
        print("🧊 Freezing CNN and RNN layers...")

        # Freeze CNN
        for param in model.cnn.parameters():
            param.requires_grad = False

        # Freeze RNN
        for param in model.rnn.parameters():
            param.requires_grad = False

        # Only train classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"🎯 Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def validate_model_compatibility(self, model, val_loader):
        """EXPERT CHECK: Validate model works before training"""
        print("🔍 Validating model compatibility...")

        model.eval()
        with torch.no_grad():
            try:
                # Test single batch
                images, targets, target_lengths = next(iter(val_loader))
                images = images[:4].to(self.device)  # Small batch

                # Forward pass
                outputs = self.model(images)
                print(f"✅ Forward pass successful: {outputs.shape}")

                # Test CTC loss
                outputs_ctc = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    (outputs_ctc.size(1),),
                    outputs_ctc.size(0),
                    dtype=torch.long,
                    device=self.device,
                )

                criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
                loss = criterion(
                    outputs_ctc,
                    targets[:4].to(self.device),
                    input_lengths,
                    target_lengths[:4].to(self.device),
                )

                print(f"✅ CTC loss calculation successful: {loss.item():.4f}")
                return True

            except Exception as e:
                print(f"❌ Model compatibility check failed: {e}")
                return False

    def train(self):
        """Expert conservative training"""
        print("🚀 Starting expert conservative training...")

        # Load data
        train_data, val_data, char_to_idx, idx_to_char = self.load_data()

        # Create datasets
        train_dataset = MaddenOCRDataset(train_data, char_to_idx)
        val_dataset = MaddenOCRDataset(val_data, char_to_idx)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, collate_fn=self.collate_fn
        )  # Smaller batch
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, collate_fn=self.collate_fn
        )

        # Initialize model
        num_classes = len(char_to_idx)
        self.model = CRNN(num_classes).to(self.device)

        # Load previous model
        checkpoint = torch.load(
            "models/pytorch_madden_ocr/best_model.pth", map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        prev_loss = checkpoint.get("val_loss", "unknown")
        print(f"✅ Loaded model with val_loss: {prev_loss}")

        # EXPERT STRATEGY: Freeze layers
        self.freeze_layers(self.model)

        # Validate compatibility
        if not self.validate_model_compatibility(self.model, val_loader):
            print("❌ Model compatibility check failed. Aborting.")
            return

        # Setup training
        criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate
        )

        # Training loop
        best_val_loss = float(prev_loss) if isinstance(prev_loss, (int, float)) else float("inf")
        patience_counter = 0

        print(f"🎯 Starting from val_loss: {best_val_loss:.4f}")

        for epoch in range(20):  # Conservative epoch count
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

                # CTC loss
                outputs_ctc = outputs.permute(1, 0, 2)
                input_lengths = torch.full(
                    (outputs_ctc.size(1),),
                    outputs_ctc.size(0),
                    dtype=torch.long,
                    device=self.device,
                )

                loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)

                # EXPERT: Gradient clipping and validation
                if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 10.0:
                    loss.backward()

                    # Aggressive gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
                else:
                    print(f"⚠️ Skipping bad loss: {loss.item()}")

                # Progress logging
                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch+1}/20, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                    )

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, targets, target_lengths in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    target_lengths = target_lengths.to(self.device)

                    outputs = self.model(images)
                    outputs_ctc = outputs.permute(1, 0, 2)
                    input_lengths = torch.full(
                        (outputs_ctc.size(1),),
                        outputs_ctc.size(0),
                        dtype=torch.long,
                        device=self.device,
                    )

                    loss = criterion(outputs_ctc, targets, input_lengths, target_lengths)

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                        val_batches += 1

            avg_train_loss = train_loss / max(num_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)

            print(
                f"Epoch {epoch+1}/20 - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                # Save model
                os.makedirs("models/pytorch_madden_ocr_expert", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "epoch": epoch,
                        "char_to_idx": char_to_idx,
                        "idx_to_char": idx_to_char,
                    },
                    "models/pytorch_madden_ocr_expert/best_model.pth",
                )

                print(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss <= self.target_loss:
                    print(f"🎯 Target loss {self.target_loss} reached! Stopping training.")
                    break
            else:
                patience_counter += 1
                print(f"⏳ No improvement. Patience: {patience_counter}/{self.patience}")

                if patience_counter >= self.patience:
                    print(f"🛑 Early stopping triggered. Best val loss: {best_val_loss:.4f}")
                    break

        print(f"✅ Expert conservative training completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")

        # Calculate improvement
        original_loss = float(prev_loss) if isinstance(prev_loss, (int, float)) else 0.653
        improvement = ((original_loss - best_val_loss) / original_loss) * 100
        print(f"📈 Improvement: {improvement:.1f}% ({original_loss:.4f} → {best_val_loss:.4f})")


if __name__ == "__main__":
    trainer = ExpertConservativeTrainer()
    trainer.train()

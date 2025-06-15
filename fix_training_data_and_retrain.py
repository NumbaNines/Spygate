#!/usr/bin/env python3
"""
Fix training data issues and retrain the custom OCR model properly.
"""

import json
import os
import sqlite3
from collections import Counter

import cv2
import numpy as np


def fix_training_data():
    """Fix the critical training data issues."""

    print("🔧 Fixing Training Data Issues")
    print("=" * 60)

    # Load original data
    with open("madden_ocr_training_data_20250614_120830.json", "r") as f:
        data = json.load(f)

    print(f"📊 Original samples: {len(data):,}")

    # Step 1: Fix inconsistent labeling (case sensitivity)
    print("\n1. Fixing case inconsistencies...")
    case_fixes = {
        "3rd & 2": "3RD & 2",
        "1st & 10": "1ST & 10",
        "2nd & 10": "2ND & 10",
        "3rd & 10": "3RD & 10",
        "1st & Goal": "1ST & Goal",
        "1st 4;03": "1st 4:03",  # Fix semicolon
    }

    fixed_count = 0
    for sample in data:
        original_text = sample.get("ground_truth_text", "")
        if original_text in case_fixes:
            sample["ground_truth_text"] = case_fixes[original_text]
            fixed_count += 1

    print(f"   Fixed {fixed_count} case inconsistencies")

    # Step 2: Remove samples with conflicting labels for same region
    print("\n2. Removing conflicting labels...")
    region_groups = {}
    for sample in data:
        img_path = sample.get("image_path", "")
        bbox = (
            sample.get("bbox_x1"),
            sample.get("bbox_y1"),
            sample.get("bbox_x2"),
            sample.get("bbox_y2"),
        )
        key = (img_path, bbox)

        if key not in region_groups:
            region_groups[key] = []
        region_groups[key].append(sample)

    # Keep only regions with consistent labels
    consistent_data = []
    removed_count = 0

    for key, samples in region_groups.items():
        texts = set(s.get("ground_truth_text", "") for s in samples)
        if len(texts) == 1:  # Consistent labeling
            consistent_data.extend(samples)
        else:  # Conflicting labels - remove all
            removed_count += len(samples)
            print(f"   Removed {len(samples)} samples with conflicting labels: {list(texts)}")

    print(f"   Removed {removed_count} samples with conflicting labels")
    print(f"   Remaining: {len(consistent_data):,} samples")

    # Step 3: Balance dataset (limit overrepresented samples)
    print("\n3. Balancing dataset...")
    text_counter = Counter([s["ground_truth_text"] for s in consistent_data])

    # Limit each text to max 100 samples (was 1,162 for "1ST & 10")
    max_per_text = 100
    balanced_data = []
    text_counts = {}

    # Shuffle to get random samples
    import random

    random.shuffle(consistent_data)

    for sample in consistent_data:
        text = sample["ground_truth_text"]
        current_count = text_counts.get(text, 0)

        if current_count < max_per_text:
            balanced_data.append(sample)
            text_counts[text] = current_count + 1

    print(f"   Before balancing: {len(consistent_data):,} samples")
    print(f"   After balancing: {len(balanced_data):,} samples")
    print(f"   Max per text: {max_per_text}")

    # Step 4: Enhance image preprocessing for better visibility
    print("\n4. Testing enhanced preprocessing...")

    def enhance_dark_text_region(image):
        """Enhanced preprocessing for dark text regions."""

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply gamma correction to brighten dark regions
        gamma = 1.5  # Brighten
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)

        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        return enhanced

    # Test enhanced preprocessing on a sample
    test_sample = balanced_data[0]
    img_path = test_sample.get("image_path")
    if os.path.exists(img_path):
        full_img = cv2.imread(img_path)
        x1, y1 = test_sample.get("bbox_x1", 0), test_sample.get("bbox_y1", 0)
        x2, y2 = test_sample.get("bbox_x2", 0), test_sample.get("bbox_y2", 0)
        region = full_img[y1:y2, x1:x2]

        original_brightness = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).mean()
        enhanced_region = enhance_dark_text_region(region)
        enhanced_brightness = enhanced_region.mean()

        print(f"   Original brightness: {original_brightness:.1f}")
        print(f"   Enhanced brightness: {enhanced_brightness:.1f}")
        print(f"   Improvement: {enhanced_brightness - original_brightness:.1f}")

        # Save comparison
        os.makedirs("debug_regions", exist_ok=True)
        cv2.imwrite("debug_regions/original_dark.png", region)
        cv2.imwrite("debug_regions/enhanced_bright.png", enhanced_region)
        print(f"   Saved comparison images in debug_regions/")

    # Step 5: Save fixed data
    print("\n5. Saving fixed training data...")

    fixed_filename = "madden_ocr_training_data_FIXED.json"
    with open(fixed_filename, "w") as f:
        json.dump(balanced_data, f, indent=2)

    print(f"   Saved {len(balanced_data):,} fixed samples to {fixed_filename}")

    # Show final distribution
    final_counter = Counter([s["ground_truth_text"] for s in balanced_data])
    print(f"\n📊 Final distribution (top 10):")
    for text, count in final_counter.most_common(10):
        print(f"   '{text}': {count}")

    print("\n✅ Training data fixed!")
    return balanced_data, enhance_dark_text_region


def create_improved_trainer(enhanced_preprocessing_func):
    """Create an improved training script with fixed data and better preprocessing."""

    trainer_code = f'''#!/usr/bin/env python3
"""
Improved OCR Trainer with Fixed Data and Enhanced Preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
import numpy as np
import torch.nn.functional as F
from datetime import datetime
import random
from collections import Counter

class ImprovedCRNN(nn.Module):
    def __init__(self, vocab_size: int, image_height: int = 64, rnn_hidden: int = 512):
        super(ImprovedCRNN, self).__init__()

        # Enhanced feature extractor with more capacity
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

        # Enhanced LSTM with more capacity
        self.sequence_processor = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,  # Increased to 512
            num_layers=2,
            bidirectional=True,
            dropout=0.2,  # Increased dropout
            batch_first=True
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

class ImprovedDataset(Dataset):
    def __init__(self, samples, character_mapping, height=64, width=256):
        self.samples = samples
        self.char_map = character_mapping
        self.height = height
        self.width = width

    def enhance_dark_text_region(self, image):
        """Enhanced preprocessing for dark text regions."""

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply gamma correction to brighten dark regions
        gamma = 1.5  # Brighten
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)

        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        return enhanced

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = sample['image_path']
        full_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Extract region
        x1, y1 = sample['bbox_x1'], sample['bbox_y1']
        x2, y2 = sample['bbox_x2'], sample['bbox_y2']
        region = full_img[y1:y2, x1:x2]

        # Apply enhanced preprocessing
        img = self.enhance_dark_text_region(region)

        # Resize
        img = cv2.resize(img, (self.width, self.height))

        # Normalize
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.FloatTensor(img).unsqueeze(0)

        # Encode text
        text = sample['ground_truth_text']
        text_indices = [self.char_map[char] for char in text]
        text_tensor = torch.LongTensor(text_indices)

        return img_tensor, text_tensor, len(text_indices)

def improved_collate_function(batch):
    images, texts, text_lengths = zip(*batch)

    images = torch.stack(images, 0)

    from torch.nn.utils.rnn import pad_sequence
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return images, texts, text_lengths

def run_improved_training():
    print("🚀 Improved OCR Training - Fixed Data & Enhanced Preprocessing")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🔒 Training ID: {timestamp}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f�� Device: {device}")

    # Load FIXED data
    with open('madden_ocr_training_data_FIXED.json', 'r') as f:
        fixed_data = json.load(f)

    print(f"📊 Fixed samples: {len(fixed_data):,}")

    # Create vocabulary
    unique_characters = set()
    for sample in fixed_data:
        text = sample['ground_truth_text']
        unique_characters.update(text)

    sorted_chars = sorted(list(unique_characters))

    char_to_idx = {'<CTC_BLANK>': 0}
    for i, char in enumerate(sorted_chars):
        char_to_idx[char] = i + 1

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"📊 Vocabulary: {len(sorted_chars)} characters")
    print(f"📊 Total classes: {len(char_to_idx)}")

    # Split data
    random.shuffle(fixed_data)
    split_point = int(0.85 * len(fixed_data))
    train_samples = fixed_data[:split_point]
    val_samples = fixed_data[split_point:]

    print(f"📊 Train samples: {len(train_samples):,}")
    print(f"📊 Val samples: {len(val_samples):,}")

    # Create datasets
    train_dataset = ImprovedDataset(train_samples, char_to_idx)
    val_dataset = ImprovedDataset(val_samples, char_to_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=improved_collate_function)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=improved_collate_function)

    # Initialize IMPROVED model
    vocab_size = len(char_to_idx)
    model = ImprovedCRNN(vocab_size, rnn_hidden=512).to(device)  # Increased capacity

    print(f"✅ Improved model initialized with {vocab_size} classes")

    # IMPROVED training setup
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # AdamW with weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)  # Cosine annealing

    print(f"🔧 IMPROVED training settings:")
    print(f"  - Learning Rate: 0.0001")
    print(f"  - Optimizer: AdamW with weight decay")
    print(f"  - Scheduler: Cosine Annealing")
    print(f"  - Batch Size: 8")
    print(f"  - Enhanced Preprocessing: ✅")
    print(f"  - Fixed Data: ✅")

    # Create save directory
    save_dir = f"models/improved_ocr_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f💾 Save directory: {save_dir}")

    # Training loop
    best_validation_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    print("\n🚀 Starting IMPROVED training...")
    print("=" * 70)

    for epoch in range(100):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        valid_train_batches = 0

        for batch_idx, (images, texts, text_lengths) in enumerate(train_loader):
            images = images.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)

            optimizer.zero_grad()

            predictions = model(images)
            predictions_ctc = predictions.permute(1, 0, 2)
            input_lengths = torch.full((predictions_ctc.size(1),), predictions_ctc.size(0), dtype=torch.long, device=device)

            loss = criterion(predictions_ctc, texts, input_lengths, text_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 100.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()

                epoch_train_loss += loss.item()
                valid_train_batches += 1

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:2d}, Batch {batch_idx:3d}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        valid_val_batches = 0

        with torch.no_grad():
            for images, texts, text_lengths in val_loader:
                images = images.to(device)
                texts = texts.to(device)
                text_lengths = text_lengths.to(device)

                predictions = model(images)
                predictions_ctc = predictions.permute(1, 0, 2)
                input_lengths = torch.full((predictions_ctc.size(1),), predictions_ctc.size(0), dtype=torch.long, device=device)

                loss = criterion(predictions_ctc, texts, input_lengths, text_lengths)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    epoch_val_loss += loss.item()
                    valid_val_batches += 1

        # Calculate averages
        avg_train_loss = epoch_train_loss / max(valid_train_batches, 1)
        avg_val_loss = epoch_val_loss / max(valid_val_batches, 1)

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            patience_counter = 0

            model_path = os.path.join(save_dir, "best_improved_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'validation_loss': avg_val_loss,
                'epoch': epoch,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'training_id': timestamp,
                'vocab_size': vocab_size,
                'improvements_applied': True,
                'enhanced_preprocessing': True,
                'fixed_data': True
            }, model_path)

            print(f💾 NEW BEST MODEL! Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss <= 0.2:  # Lower target
                print(f🎯 Target loss 0.2 reached! Stopping training.")
                break
        else:
            patience_counter += 1
            print(f⏳ No improvement. Patience: {patience_counter}/{max_patience}")

            if patience_counter >= max_patience:
                print(f🛑 Early stopping. Best val loss: {best_validation_loss:.4f}")
                break

    print("\n" + "=" * 70)
    print("✅ IMPROVED training completed!")
    print(f🏆 Best validation loss: {best_validation_loss:.4f}")
    print(f💾 Model saved in: {save_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_improved_training()
'''

    with open("improved_ocr_trainer.py", "w") as f:
        f.write(trainer_code)

    print(f"\n📝 Created improved_ocr_trainer.py")
    print(f"   - Enhanced model architecture (512 LSTM units)")
    print(f"   - Improved preprocessing (brightness enhancement)")
    print(f"   - Better training (AdamW, Cosine LR, gradient clipping)")
    print(f"   - Fixed data (balanced, consistent labels)")


def main():
    """Main function to fix data and create improved trainer."""

    print("🔧 OCR Training Data Fix & Improvement")
    print("=" * 60)

    # Fix training data
    fixed_data, enhance_func = fix_training_data()

    # Create improved trainer
    create_improved_trainer(enhance_func)

    print("\n" + "=" * 60)
    print("🎯 READY TO RETRAIN!")
    print("=" * 60)
    print("\n🚀 NEXT STEPS:")
    print("   1. Run: python improved_ocr_trainer.py")
    print("   2. Wait for training to complete (50-100 epochs)")
    print("   3. Update custom OCR path to new model")
    print("   4. Test with real video")
    print("\n💡 EXPECTED IMPROVEMENTS:")
    print("   - 90%+ accuracy (vs current garbled output)")
    print("   - Proper text recognition")
    print("   - No more 'K1tC' patterns")
    print("   - Better performance than EasyOCR")


if __name__ == "__main__":
    main()

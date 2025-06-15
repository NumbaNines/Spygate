#!/usr/bin/env python3
"""
Expert OCR Solution: TrOCR Fine-tuning for Madden HUD
Uses Microsoft's TrOCR (Transformer-based OCR) - state-of-the-art text recognition.
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel


class MaddenOCRDataset(Dataset):
    def __init__(self, data, processor, max_target_length=20):
        self.data = data
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, image_path):
        """Expert-level preprocessing for dark HUD text."""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return Image.new("RGB", (384, 64), color="white")

            # Resize maintaining aspect ratio
            h, w = img.shape
            target_h = 64
            target_w = int(w * target_h / h)
            img = cv2.resize(img, (target_w, target_h))

            # Pad to standard width
            if target_w < 384:
                pad_w = 384 - target_w
                img = cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=255)
            elif target_w > 384:
                img = cv2.resize(img, (384, 64))

            # Advanced preprocessing for dark text
            # 1. Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 2))
            img = clahe.apply(img)

            # 2. Morphological operations to clean text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            # 3. Bilateral filter for noise reduction while preserving edges
            img = cv2.bilateralFilter(img, 5, 50, 50)

            # 4. Gamma correction for better contrast
            gamma = 0.8
            img = np.power(img / 255.0, gamma) * 255.0
            img = img.astype(np.uint8)

            # Convert to RGB for TrOCR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            return Image.fromarray(img_rgb)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return Image.new("RGB", (384, 64), color="white")

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process image
        image = self.preprocess_image(item["image_path"])

        # Process text
        text = item["ground_truth_text"]

        # Tokenize
        encoding = self.processor(
            image,
            text=text,
            max_target_length=self.max_target_length,
            padding="max_length",
            return_tensors="pt",
        )

        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding


def setup_trocr_model():
    """Setup TrOCR model for fine-tuning."""
    print("🚀 Setting up TrOCR Model")

    # Use TrOCR base model (pre-trained on printed text)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    # Configure for our task
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Set beam search parameters for better inference
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 20
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return processor, model


def train_expert_ocr():
    """Train expert OCR using TrOCR fine-tuning."""
    print("🎯 Expert OCR Training: TrOCR Fine-tuning")
    print("=" * 60)

    # Load core dataset
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    print(f"📊 Dataset: {len(data):,} samples")

    # Setup model
    processor, model = setup_trocr_model()

    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")

    # Create datasets
    train_dataset = MaddenOCRDataset(train_data, processor)
    val_dataset = MaddenOCRDataset(val_data, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trocr-madden-ocr",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,  # Mixed precision for faster training
    )

    # Custom trainer for text generation
    class OCRTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = OCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )

    # Train
    print("🚀 Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model("./trocr-madden-final")
    processor.save_pretrained("./trocr-madden-final")

    print("✅ Training complete!")
    print("💾 Model saved to: ./trocr-madden-final")


def test_expert_ocr():
    """Test the fine-tuned TrOCR model."""
    print("🧪 Testing Expert OCR Model")
    print("=" * 50)

    # Load model
    processor = TrOCRProcessor.from_pretrained("./trocr-madden-final")
    model = VisionEncoderDecoderModel.from_pretrained("./trocr-madden-final")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    with open("madden_ocr_training_data_CORE.json", "r") as f:
        data = json.load(f)

    # Test on samples
    import random

    test_samples = random.sample(data, 10)

    correct = 0
    total = 0

    for i, sample in enumerate(test_samples):
        # Preprocess image
        dataset = MaddenOCRDataset([sample], processor)
        image = dataset.preprocess_image(sample["image_path"])

        # Generate prediction
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_length=20, num_beams=4, early_stopping=True
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ground_truth = sample["ground_truth_text"]

        is_correct = generated_text.strip() == ground_truth
        if is_correct:
            correct += 1
        total += 1

        status = "✅" if is_correct else "❌"
        print(f"{i+1:2d}. {status} GT: '{ground_truth}' | Pred: '{generated_text.strip()}'")

    accuracy = correct / total
    print(f"\n📊 Expert OCR Accuracy: {accuracy:.1%} ({correct}/{total})")


if __name__ == "__main__":
    # Check if model exists
    if os.path.exists("./trocr-madden-final"):
        print("🔍 Found existing model, testing...")
        test_expert_ocr()
    else:
        print("🚀 No existing model found, starting training...")
        train_expert_ocr()

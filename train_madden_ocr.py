"""
SpygateAI Custom Madden OCR Training
Train a specialized OCR model for Madden HUD elements.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


class MaddenOCRDataset(Dataset):
    """Custom dataset for Madden HUD OCR training"""

    def __init__(self, db_path: str, region_type: str = None, transform=None):
        self.db_path = db_path
        self.region_type = region_type
        self.transform = transform
        self.samples = self.load_samples()

    def load_samples(self) -> List[Tuple[str, str]]:
        """Load samples from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if self.region_type:
            cursor.execute(
                """
                SELECT image_path, text_content
                FROM ocr_samples
                WHERE region_type = ? AND text_content != 'UNKNOWN'
            """,
                (self.region_type,),
            )
        else:
            cursor.execute(
                """
                SELECT image_path, text_content
                FROM ocr_samples
                WHERE text_content != 'UNKNOWN'
            """
            )

        samples = cursor.fetchall()
        conn.close()

        print(f"Loaded {len(samples)} samples for region: {self.region_type or 'ALL'}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text_content = self.samples[idx]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {"image": image, "text": text_content, "image_path": image_path}


class MaddenOCRTrainer:
    """Training class for custom Madden OCR model"""

    def __init__(self, dataset_path: str = "madden_ocr_dataset"):
        self.dataset_path = Path(dataset_path)
        self.db_path = self.dataset_path / "ocr_dataset.db"
        self.model_output_dir = Path("trained_madden_ocr")
        self.model_output_dir.mkdir(exist_ok=True)

        # Initialize TrOCR components
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"🔥 MADDEN OCR TRAINER INITIALIZED")
        print(f"Device: {self.device}")
        print(f"Dataset path: {self.dataset_path}")

    def create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline for training robustness"""
        return A.Compose(
            [
                # Geometric transformations
                A.Rotate(limit=5, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
                # Color/brightness adjustments
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3
                ),
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def get_region_datasets(self) -> Dict[str, MaddenOCRDataset]:
        """Create datasets for each region type"""
        datasets = {}
        transform = self.create_augmentation_pipeline()

        # Get available region types
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT region_type FROM ocr_samples WHERE text_content != 'UNKNOWN'"
        )
        region_types = [row[0] for row in cursor.fetchall()]
        conn.close()

        print(f"📊 Creating datasets for regions: {region_types}")

        for region_type in region_types:
            datasets[region_type] = MaddenOCRDataset(
                self.db_path, region_type=region_type, transform=transform
            )

        return datasets

    def train_region_model(self, region_type: str, dataset: MaddenOCRDataset, epochs: int = 10):
        """Train model for specific region type"""
        print(f"\n🚀 TRAINING {region_type.upper()} MODEL")

        if len(dataset) < 5:
            print(f"⚠️  Insufficient data for {region_type} ({len(dataset)} samples). Skipping.")
            return None

        # Create data loader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=self.collate_fn)

        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Training loop
        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Process images and text
                    images = [
                        Image.fromarray(img.numpy().transpose(1, 2, 0)) for img in batch["image"]
                    ]
                    texts = batch["text"]

                    # Encode inputs
                    pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(
                        self.device
                    )
                    labels = self.processor.tokenizer(
                        texts, padding=True, return_tensors="pt"
                    ).input_ids.to(self.device)

                    # Forward pass
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    if batch_idx % 5 == 0:
                        print(
                            f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                        )

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

            avg_loss = epoch_loss / len(dataloader)
            total_loss += avg_loss
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Save region-specific model
        model_path = self.model_output_dir / f"madden_ocr_{region_type}"
        model_path.mkdir(exist_ok=True)

        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)

        print(f"✅ {region_type} model saved to {model_path}")
        return model_path

    def collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        images = torch.stack([item["image"] for item in batch])
        texts = [item["text"] for item in batch]
        image_paths = [item["image_path"] for item in batch]

        return {"image": images, "text": texts, "image_path": image_paths}

    def train_all_models(self, epochs: int = 15):
        """Train models for all region types"""
        print("🔥 STARTING MADDEN OCR TRAINING FOR ALL REGIONS")

        datasets = self.get_region_datasets()
        trained_models = {}

        for region_type, dataset in datasets.items():
            try:
                model_path = self.train_region_model(region_type, dataset, epochs)
                if model_path:
                    trained_models[region_type] = str(model_path)
            except Exception as e:
                print(f"❌ Failed to train {region_type}: {e}")

        # Save training summary
        summary = {
            "trained_models": trained_models,
            "total_regions": len(datasets),
            "successful_training": len(trained_models),
            "training_config": {
                "epochs": epochs,
                "base_model": "microsoft/trocr-base-printed",
                "device": str(self.device),
            },
        }

        with open(self.model_output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎯 TRAINING COMPLETE!")
        print(f"Successfully trained {len(trained_models)}/{len(datasets)} models")
        print(f"Models saved in: {self.model_output_dir}")

        return trained_models

    def test_model(self, region_type: str, test_image_path: str):
        """Test trained model on specific image"""
        model_path = self.model_output_dir / f"madden_ocr_{region_type}"

        if not model_path.exists():
            print(f"❌ No trained model found for {region_type}")
            return None

        # Load trained model
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(self.device)
        model.eval()

        # Load and process image
        image = Image.open(test_image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"🔍 {region_type} OCR Result: '{generated_text}'")
        return generated_text


def main():
    """Main training execution"""
    trainer = MaddenOCRTrainer()

    # Train all region models
    trained_models = trainer.train_all_models(epochs=20)

    print("\n📋 Training Summary:")
    for region, model_path in trained_models.items():
        print(f"  {region}: {model_path}")


if __name__ == "__main__":
    main()

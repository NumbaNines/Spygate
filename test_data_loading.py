#!/usr/bin/env python3
"""Test data loading in isolation"""

import logging

from expert_unified_ocr_trainer import ExpertOCRTrainer

# Set up logging to see debug messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

print("🧪 Testing data loading in isolation...")

try:
    # Create trainer
    trainer = ExpertOCRTrainer()
    print(f"✅ Trainer created with {trainer.num_classes} character classes")

    # Test data loading with just a few samples
    print("\n📁 Loading training data...")
    train_ds, val_ds, class_dist = trainer.load_and_preprocess_data(
        "madden_ocr_training_data_20250614_120830.json"
    )

    print(f"✅ Data loaded successfully!")
    print(f"📊 Class distribution: {class_dist}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

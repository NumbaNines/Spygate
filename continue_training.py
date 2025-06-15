#!/usr/bin/env python3
"""
Continue PyTorch OCR Training - Fine-tuning for 0.35 Val Loss
Load best model and continue training with lower learning rate
"""

import logging
from pathlib import Path

import torch
import torch.optim as optim

from pytorch_madden_ocr_trainer import CRNN, PyTorchMaddenOCRTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def continue_training():
    """Continue training from best checkpoint"""

    print("🔄 Continuing PyTorch OCR Training")
    print("=" * 50)

    # Initialize trainer
    trainer = PyTorchMaddenOCRTrainer()

    # Load best model checkpoint
    best_model_path = Path("models/pytorch_madden_ocr/best_model.pth")
    if not best_model_path.exists():
        raise FileNotFoundError("No best model found! Run initial training first.")

    print("📁 Loading best model checkpoint...")
    checkpoint = torch.load(best_model_path, map_location=trainer.device)

    print(f"📊 Previous training results:")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Best val loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Target: 0.35 val loss")
    print(
        f"  - Improvement needed: {((checkpoint['val_loss'] - 0.35) / checkpoint['val_loss'] * 100):.1f}%"
    )

    # Build model and load weights
    model = CRNN(trainer.num_classes, trainer.img_height)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(trainer.device)
    trainer.model = model

    print("✅ Model loaded successfully!")

    # Load data
    print("📁 Loading training data...")
    train_loader, val_loader, class_dist = trainer.load_and_prepare_data(
        "madden_ocr_training_data_20250614_120830.json"
    )

    # Continue training with lower learning rate
    print("🚀 Starting extended training...")
    print("📉 Reduced learning rate: 0.001 → 0.0001 (10x lower)")
    print("🎯 Target: 50 more epochs to reach 0.35 val loss")

    # Override the train_model method to continue from checkpoint
    import torch.nn as nn

    criterion = nn.CTCLoss(blank=trainer.blank_token, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5)

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = checkpoint["val_loss"]  # Start from previous best
    start_epoch = checkpoint["epoch"] + 1
    total_epochs = start_epoch + 50  # 50 more epochs

    print(f"📊 Training epochs {start_epoch} to {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (images, texts, text_lengths, raw_texts) in enumerate(train_loader):
            images = images.to(trainer.device)
            texts = texts.to(trainer.device)
            text_lengths = text_lengths.to(trainer.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Prepare for CTC loss
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            # CTC loss
            loss = criterion(outputs, texts, input_lengths, text_lengths)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}/{total_epochs-1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, texts, text_lengths, raw_texts in val_loader:
                images = images.to(trainer.device)
                texts = texts.to(trainer.device)
                text_lengths = text_lengths.to(trainer.device)

                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                loss = criterion(outputs, texts, input_lengths, text_lengths)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch}/{total_epochs-1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val_loss,
                "char_to_idx": trainer.char_to_idx,
                "idx_to_char": trainer.idx_to_char,
                "madden_chars": trainer.madden_chars,
                "num_classes": trainer.num_classes,
                "img_height": trainer.img_height,
                "img_width": trainer.img_width,
            }

            torch.save(checkpoint, trainer.model_save_path / "best_model.pth")
            logger.info(f"💾 New best model saved! Val Loss: {avg_val_loss:.4f}")

            # Check if we reached target
            if avg_val_loss <= 0.35:
                logger.info(f"🎉 TARGET REACHED! Val loss {avg_val_loss:.4f} <= 0.35")
                logger.info("🏆 Training completed successfully!")
                break

    # Save extended training history
    import json

    extended_history = {
        "extended_train_losses": train_losses,
        "extended_val_losses": val_losses,
        "start_epoch": start_epoch,
        "final_best_val_loss": best_val_loss,
    }

    with open(trainer.model_save_path / "extended_training_history.json", "w") as f:
        json.dump(extended_history, f, indent=2)

    print(f"\n🎉 Extended training completed!")
    print(f"🏆 Final best validation loss: {best_val_loss:.4f}")
    print(f"📁 Model saved to: {trainer.model_save_path}")

    if best_val_loss <= 0.35:
        print("✅ TARGET ACHIEVED! Ready for production use!")
    else:
        print(f"⚠️ Target not reached. Consider training for more epochs.")


if __name__ == "__main__":
    continue_training()

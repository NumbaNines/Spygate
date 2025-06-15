#!/usr/bin/env python3
"""
KerasOCR Madden Trainer - Transfer Learning for Maximum Accuracy
Optimized for Madden HUD text patterns with pre-trained KerasOCR models
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from ultimate_madden_ocr_system import MaddenOCRDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaddenKerasOCRTrainer:
    """Transfer learning trainer using KerasOCR for Madden HUD text recognition"""

    def __init__(self, model_save_path: str = "models/madden_keras_ocr"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # Madden-specific character set
        self.madden_chars = "0123456789STNDRDTHGOALqtrFLAG&:- "
        self.char_to_idx = {char: idx for idx, char in enumerate(self.madden_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.madden_chars)

        # Model parameters
        self.img_height = 64
        self.img_width = 256
        self.max_text_length = 15  # Max characters in Madden HUD text

        logger.info(f"Initialized KerasOCR trainer with {self.num_classes} character classes")
        logger.info(f"Character set: '{self.madden_chars}'")

    def load_training_data(self, json_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from database or JSON file"""
        if json_file and os.path.exists(json_file):
            logger.info(f"Loading training data from {json_file}")
            with open(json_file, "r") as f:
                training_data = json.load(f)
        else:
            logger.info("Loading training data from database")
            db = MaddenOCRDatabase()
            training_data = db.get_training_data()

        if not training_data:
            raise ValueError("No training data found!")

        logger.info(f"Loaded {len(training_data)} training samples")

        # Prepare data
        images = []
        texts = []

        for sample in training_data:
            # Get image data
            if "region_image_data" in sample and sample["region_image_data"]:
                # Decode base64 image
                import base64

                img_data = base64.b64decode(sample["region_image_data"])
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            else:
                # Skip samples without image data
                continue

            # Get ground truth text
            text = sample.get("ground_truth_text", "").strip()
            if not text or len(text) > self.max_text_length:
                continue

            # Validate text contains only Madden characters
            if not all(c in self.madden_chars for c in text):
                logger.debug(f"Skipping text with invalid characters: '{text}'")
                continue

            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is not None:
                images.append(processed_image)
                texts.append(text)

        logger.info(f"Processed {len(images)} valid samples")

        if len(images) == 0:
            raise ValueError("No valid training samples found!")

        # Convert to numpy arrays
        X = np.array(images)
        y = self.encode_texts(texts)

        return X, y

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for KerasOCR training"""
        if image is None or image.size == 0:
            return None

        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to standard size
        image = cv2.resize(image, (self.img_width, self.img_height))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=-1)

        return image

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text strings to sequences of character indices"""
        encoded_texts = []

        for text in texts:
            # Convert text to character indices
            encoded = [self.char_to_idx.get(char, 0) for char in text]

            # Pad to max length
            if len(encoded) < self.max_text_length:
                encoded.extend([0] * (self.max_text_length - len(encoded)))
            else:
                encoded = encoded[: self.max_text_length]

            encoded_texts.append(encoded)

        return np.array(encoded_texts)

    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """Decode model predictions back to text strings"""
        decoded_texts = []

        for pred in predictions:
            # Get character indices
            char_indices = np.argmax(pred, axis=-1)

            # Convert to text
            text = "".join([self.idx_to_char.get(idx, "") for idx in char_indices])

            # Remove padding (trailing spaces/zeros)
            text = text.rstrip(" \x00")

            decoded_texts.append(text)

        return decoded_texts

    def build_model(self) -> keras.Model:
        """Build KerasOCR-inspired model with transfer learning"""
        logger.info("Building KerasOCR model with transfer learning...")

        # Input layer
        inputs = keras.layers.Input(shape=(self.img_height, self.img_width, 1), name="image_input")

        # CNN Feature Extractor (inspired by KerasOCR)
        x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        x = keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)

        # Reshape for sequence processing
        new_shape = ((self.img_width // 16), (self.img_height // 16) * 256)
        x = keras.layers.Reshape(target_shape=new_shape)(x)

        # Bidirectional LSTM layers (KerasOCR style)
        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)

        # Dense layer for character prediction
        x = keras.layers.Dense(self.num_classes, activation="softmax", name="character_output")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=x, name="MaddenKerasOCR")

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"Model built with {model.count_params():,} parameters")
        return model

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the KerasOCR model"""
        logger.info(f"Starting training with {len(X)} samples...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Build model
        self.model = self.build_model()

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.model_save_path / "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
            ),
        ]

        # Reshape y for sequence prediction
        y_train_seq = np.expand_dims(y_train, axis=-1)
        y_val_seq = np.expand_dims(y_val, axis=-1)

        # Train model
        history = self.model.fit(
            X_train,
            y_train_seq,
            validation_data=(X_val, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Save final model
        self.model.save(str(self.model_save_path / "final_model.h5"))

        # Save training history
        with open(self.model_save_path / "training_history.json", "w") as f:
            json.dump(history.history, f, indent=2)

        # Save character mappings
        char_mappings = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "madden_chars": self.madden_chars,
        }
        with open(self.model_save_path / "char_mappings.json", "w") as f:
            json.dump(char_mappings, f, indent=2)

        logger.info("Training completed!")
        return history

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance"""
        logger.info("Evaluating model...")

        # Make predictions
        y_test_seq = np.expand_dims(y_test, axis=-1)
        predictions = self.model.predict(X_test)

        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_seq, verbose=0)

        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        # Sample predictions
        sample_predictions = self.decode_predictions(predictions[:10])
        sample_ground_truth = [self.decode_sequence(seq) for seq in y_test[:10]]

        logger.info("Sample predictions:")
        for i, (pred, truth) in enumerate(zip(sample_predictions, sample_ground_truth)):
            logger.info(f"  {i+1}: '{pred}' (truth: '{truth}')")

        return test_accuracy, test_loss

    def decode_sequence(self, sequence: np.ndarray) -> str:
        """Decode a sequence of character indices to text"""
        text = "".join([self.idx_to_char.get(idx, "") for idx in sequence])
        return text.rstrip(" \x00")


def main():
    print("🎯 KerasOCR Madden Trainer - Transfer Learning")
    print("=" * 50)

    # Initialize trainer
    trainer = MaddenKerasOCRTrainer()

    # Load training data
    try:
        # Try to load from exported JSON first
        json_files = list(Path(".").glob("madden_ocr_training_data_*.json"))
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Loading data from: {latest_json}")
            X, y = trainer.load_training_data(str(latest_json))
        else:
            print("📁 Loading data from database...")
            X, y = trainer.load_training_data()

        print(f"✅ Loaded {len(X)} training samples")
        print(f"📊 Image shape: {X[0].shape}")
        print(f"📊 Text sequence length: {y.shape[1]}")

        # Train model
        print("\n🚀 Starting KerasOCR training...")
        history = trainer.train_model(X, y, epochs=30, batch_size=16)

        # Evaluate
        print("\n📊 Evaluating model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy, loss = trainer.evaluate_model(X_test, y_test)

        print(f"\n✅ Training completed!")
        print(f"📈 Final accuracy: {accuracy:.2%}")
        print(f"💾 Model saved to: {trainer.model_save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Training failed")


if __name__ == "__main__":
    main()

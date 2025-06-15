#!/usr/bin/env python3
"""
Madden OCR Trainer - Specialized for Madden HUD text with game constraints
Optimized for: Numbers 0-9, ordinals (ST/ND/RD/TH), GOAL, QTR, FLAG, &, :, -
With constraints: Play clock ≤40, Quarter time ≤5:00, Distance ≤34 yards
"""

import json
import logging
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MaddenOCRTrainer:
    """Specialized OCR trainer for Madden HUD text"""

    def __init__(self, database_path: str = "madden_ocr_training.db"):
        self.database_path = database_path
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.max_text_length = 10  # Max length for Madden HUD text

        # Madden-specific character set
        self.madden_chars = sorted(list(set("0123456789STNDRDTHGOALQTRFLAG&:- ")))
        self.vocab_size = len(self.madden_chars) + 1  # +1 for blank/padding

        self._build_char_mappings()

    def _build_char_mappings(self):
        """Build character to index mappings"""
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.madden_chars)}
        self.char_to_idx[""] = 0  # Padding/blank token

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        logger.info(f"Built character mappings for {len(self.madden_chars)} characters")
        logger.info(f"Character set: {''.join(self.madden_chars)}")

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load validated training data from database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM ocr_samples
            WHERE is_validated = TRUE AND ground_truth_text IS NOT NULL
            AND ground_truth_text != ''
            ORDER BY class_name, confidence DESC
        """
        )

        columns = [desc[0] for desc in cursor.description]
        samples = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()

        if not samples:
            raise ValueError("No validated training data found in database")

        logger.info(f"Loaded {len(samples)} validated training samples")

        # Load images and prepare data
        images = []
        texts = []
        contexts = []

        for sample in samples:
            try:
                # Load original image
                image_path = sample["image_path"]
                image = cv2.imread(image_path)

                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue

                # Extract region
                x1, y1, x2, y2 = (
                    sample["bbox_x1"],
                    sample["bbox_y1"],
                    sample["bbox_x2"],
                    sample["bbox_y2"],
                )
                region = image[y1:y2, x1:x2]

                if region.size == 0:
                    continue

                # Preprocess region
                processed_region = self._preprocess_image(region)

                images.append(processed_region)
                texts.append(sample["ground_truth_text"].upper().strip())
                contexts.append(sample["class_name"])

            except Exception as e:
                logger.warning(f"Error processing sample {sample['id']}: {e}")
                continue

        if not images:
            raise ValueError("No valid training images could be loaded")

        logger.info(f"Successfully processed {len(images)} training samples")

        # Convert to numpy arrays
        images = np.array(images)

        # Encode texts
        encoded_texts = self._encode_texts(texts)

        return images, encoded_texts, contexts

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for training"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to standard size (width=128, height=32)
        target_height = 32
        target_width = 128

        # Maintain aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h

        if aspect_ratio > target_width / target_height:
            # Width is limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Height is limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Pad to target size
        padded = np.zeros((target_height, target_width), dtype=np.uint8)

        # Center the image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        padded[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

        # Normalize
        normalized = padded.astype(np.float32) / 255.0

        # Add channel dimension
        return np.expand_dims(normalized, axis=-1)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text labels for training"""
        encoded = []

        for text in texts:
            # Convert to character indices
            char_indices = []
            for char in text[: self.max_text_length]:  # Truncate if too long
                if char in self.char_to_idx:
                    char_indices.append(self.char_to_idx[char])
                else:
                    logger.warning(f"Unknown character '{char}' in text '{text}'")
                    char_indices.append(0)  # Use blank token

            # Pad to max length
            while len(char_indices) < self.max_text_length:
                char_indices.append(0)  # Padding

            encoded.append(char_indices)

        return np.array(encoded)

    def _decode_text(self, indices: np.ndarray) -> str:
        """Decode indices back to text"""
        chars = []
        for idx in indices:
            if idx > 0 and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        return "".join(chars).strip()

    def build_model(self) -> keras.Model:
        """Build CNN + RNN model for OCR"""
        # Input layer
        input_img = layers.Input(shape=(32, 128, 1), name="image")

        # CNN layers for feature extraction
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)  # Keep width for sequence

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)

        # Reshape for RNN
        new_shape = ((128 // 4), (32 // 4) * 128)  # (width, height * channels)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)

        # RNN layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(self.vocab_size, activation="softmax")(x)

        model = keras.Model(inputs=input_img, outputs=x)

        # Compile with CTC loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self._ctc_loss_func,
            metrics=["accuracy"],
        )

        return model

    def _ctc_loss_func(self, y_true, y_pred):
        """CTC loss function"""
        # Get the batch size
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

        # Get the length of the prediction and label sequences
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def train_model(
        self,
        images: np.ndarray,
        texts: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
    ) -> keras.Model:
        """Train the OCR model"""
        logger.info("Building model...")
        self.model = self.build_model()

        logger.info(f"Model summary:")
        self.model.summary()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, texts, test_size=validation_split, random_state=42
        )

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                "madden_ocr_best_model.h5", monitor="val_loss", save_best_only=True
            ),
        ]

        # Train model
        logger.info("Starting training...")
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )

        # Plot training history
        self._plot_training_history(history)

        return self.model

    def _plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("madden_ocr_training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("Training history saved as 'madden_ocr_training_history.png'")

    def evaluate_model(self, images: np.ndarray, texts: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(images)

        # Decode predictions
        predicted_texts = []
        actual_texts = []

        for i in range(len(predictions)):
            # Get prediction
            pred_indices = np.argmax(predictions[i], axis=1)
            pred_text = self._decode_text(pred_indices)
            predicted_texts.append(pred_text)

            # Get actual text
            actual_text = self._decode_text(texts[i])
            actual_texts.append(actual_text)

        # Calculate accuracy
        exact_matches = sum(
            1 for pred, actual in zip(predicted_texts, actual_texts) if pred == actual
        )
        accuracy = exact_matches / len(predicted_texts)

        # Character-level accuracy
        total_chars = 0
        correct_chars = 0

        for pred, actual in zip(predicted_texts, actual_texts):
            total_chars += max(len(pred), len(actual))
            correct_chars += sum(1 for p, a in zip(pred, actual) if p == a)

        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

        results = {
            "exact_match_accuracy": accuracy,
            "character_accuracy": char_accuracy,
            "total_samples": len(predicted_texts),
            "exact_matches": exact_matches,
            "predictions": list(zip(actual_texts, predicted_texts)),
        }

        logger.info(f"Evaluation Results:")
        logger.info(f"Exact Match Accuracy: {accuracy:.4f}")
        logger.info(f"Character Accuracy: {char_accuracy:.4f}")
        logger.info(f"Total Samples: {len(predicted_texts)}")
        logger.info(f"Exact Matches: {exact_matches}")

        return results

    def save_model(
        self,
        model_path: str = "madden_ocr_model.h5",
        mappings_path: str = "madden_ocr_mappings.pkl",
    ):
        """Save trained model and character mappings"""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        self.model.save(model_path)

        # Save character mappings
        mappings = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "madden_chars": self.madden_chars,
            "vocab_size": self.vocab_size,
            "max_text_length": self.max_text_length,
        }

        with open(mappings_path, "wb") as f:
            pickle.dump(mappings, f)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Mappings saved to: {mappings_path}")

    def load_model(
        self,
        model_path: str = "madden_ocr_model.h5",
        mappings_path: str = "madden_ocr_mappings.pkl",
    ):
        """Load trained model and character mappings"""
        # Load character mappings
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)

        self.char_to_idx = mappings["char_to_idx"]
        self.idx_to_char = mappings["idx_to_char"]
        self.madden_chars = mappings["madden_chars"]
        self.vocab_size = mappings["vocab_size"]
        self.max_text_length = mappings["max_text_length"]

        # Load model
        self.model = keras.models.load_model(
            model_path, custom_objects={"_ctc_loss_func": self._ctc_loss_func}
        )

        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Mappings loaded from: {mappings_path}")

    def predict_text(self, image: np.ndarray) -> str:
        """Predict text from image"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Preprocess image
        processed = self._preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)  # Add batch dimension

        # Predict
        prediction = self.model.predict(processed, verbose=0)

        # Decode
        pred_indices = np.argmax(prediction[0], axis=1)
        predicted_text = self._decode_text(pred_indices)

        return predicted_text


def main():
    """Main training pipeline"""
    print("🎯 Madden OCR Trainer - Maximum Accuracy System")
    print("=" * 50)

    # Check if training data exists
    db_path = "madden_ocr_training.db"
    if not os.path.exists(db_path):
        print(f"❌ Training database not found: {db_path}")
        print("Please run the extraction system first to create training data.")
        return

    # Initialize trainer
    trainer = MaddenOCRTrainer(db_path)

    try:
        # Load training data
        print("\n📊 Loading training data...")
        images, texts, contexts = trainer.load_training_data()

        print(f"✅ Loaded {len(images)} training samples")
        print(f"Image shape: {images[0].shape}")
        print(
            f"Text examples: {[trainer._decode_text(texts[i]) for i in range(min(5, len(texts)))]}"
        )

        # Train model
        print("\n🚀 Training model...")
        model = trainer.train_model(images, texts, epochs=50)

        # Evaluate model
        print("\n📈 Evaluating model...")
        results = trainer.evaluate_model(images, texts)

        # Save model
        print("\n💾 Saving model...")
        trainer.save_model()

        print("\n🎉 Training complete!")
        print(f"Final Accuracy: {results['exact_match_accuracy']:.4f}")
        print(f"Character Accuracy: {results['character_accuracy']:.4f}")

        # Show some predictions
        print("\n🔍 Sample Predictions:")
        for i, (actual, predicted) in enumerate(results["predictions"][:10]):
            status = "✅" if actual == predicted else "❌"
            print(f"{status} Actual: '{actual}' | Predicted: '{predicted}'")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

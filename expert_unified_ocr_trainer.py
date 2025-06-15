#!/usr/bin/env python3
"""
Expert Unified OCR Trainer - Maximum Performance & Accuracy
GPU-optimized, compatibility-fixed, production-ready trainer for Madden HUD text
"""

import base64
import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision

from ultimate_madden_ocr_system import MaddenOCRDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExpertOCRTrainer:
    """Expert-level unified OCR trainer with maximum GPU optimization"""

    def __init__(self, model_save_path: str = "models/expert_madden_ocr"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        # GPU optimization setup
        self._setup_gpu_optimization()

        # Madden-specific character set (based on actual data analysis)
        self.madden_chars = " &-0123456789:;ACDFGHIKLOPRSTadhlnorst"  # Space first for padding
        self.char_to_idx = {char: idx for idx, char in enumerate(self.madden_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.madden_chars)
        self.blank_token = 0  # Space character for CTC

        # Optimized model parameters
        self.img_height = 64
        self.img_width = 256
        self.max_text_length = 15

        # Training optimization parameters
        self.mixed_precision = True
        self.use_xla = True
        self.prefetch_buffer = tf.data.AUTOTUNE

        logger.info(f"🚀 Expert OCR Trainer initialized")
        logger.info(f"📊 Character classes: {self.num_classes}")
        logger.info(f"🎯 Character set: '{self.madden_chars}'")
        logger.info(f"⚡ GPU optimization: Enabled")
        logger.info(f"🔥 Mixed precision: {self.mixed_precision}")
        logger.info(f"⚡ XLA compilation: {self.use_xla}")

    def _setup_gpu_optimization(self):
        """Configure optimal GPU settings for maximum performance"""
        # Enable GPU memory growth to prevent OOM
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"🎮 GPU optimization enabled for {len(gpus)} GPU(s)")

                # Enable mixed precision for RTX cards
                if self.mixed_precision:
                    mixed_precision.set_global_policy("mixed_float16")
                    logger.info("🔥 Mixed precision (FP16) enabled for faster training")

                # Enable XLA compilation
                if self.use_xla:
                    tf.config.optimizer.set_jit(True)
                    logger.info("⚡ XLA compilation enabled for optimized kernels")

            except RuntimeError as e:
                logger.warning(f"GPU setup warning: {e}")
        else:
            logger.warning("⚠️ No GPU detected - training will use CPU")

    def load_and_preprocess_data(
        self, json_file: str = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        """Load and create optimized TensorFlow datasets"""
        # Load training data
        if json_file and os.path.exists(json_file):
            logger.info(f"📁 Loading data from {json_file}")
            with open(json_file, "r") as f:
                training_data = json.load(f)
        else:
            logger.info("📁 Loading data from database")
            db = MaddenOCRDatabase()
            training_data = db.get_training_data()

        if not training_data:
            raise ValueError("No training data found!")

        logger.info(f"📊 Processing {len(training_data)} samples...")

        # Prepare data with expert preprocessing
        images, texts, classes = [], [], []
        class_distribution = {}

        # Debug counters
        debug_no_image = 0
        debug_no_text = 0
        debug_invalid_chars = 0
        debug_image_failed = 0
        debug_preprocess_failed = 0
        debug_success = 0

        for i, sample in enumerate(training_data):
            # Skip samples without image data (handle both formats)
            has_image_data = ("region_image_data" in sample and sample["region_image_data"]) or (
                "image_path" in sample and sample["image_path"]
            )
            if not has_image_data:
                debug_no_image += 1
                if i < 5:
                    logger.info(f"❌ Sample {i}: No image data")
                continue

            # Get ground truth text
            text = sample.get("ground_truth_text", "").strip()
            if not text or len(text) > self.max_text_length:
                debug_no_text += 1
                if i < 5:
                    logger.info(f"❌ Sample {i}: Bad text: '{text}' (len={len(text)})")
                continue

            # Validate text contains only Madden characters
            if not all(c in self.madden_chars for c in text):
                debug_invalid_chars += 1
                invalid_chars = [c for c in text if c not in self.madden_chars]
                if i < 5:
                    logger.info(f"❌ Sample {i}: Invalid chars {invalid_chars} in '{text}'")
                continue

            # Load and preprocess image (handle both formats)
            try:
                if "region_image_data" in sample and sample["region_image_data"]:
                    # Base64 encoded image data
                    img_data = base64.b64decode(sample["region_image_data"])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                elif "image_path" in sample and sample["image_path"]:
                    # Image file path
                    image_path = sample["image_path"]
                    if not os.path.exists(image_path):
                        logger.debug(f"Image file not found: {image_path}")
                        continue
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as color first
                else:
                    continue

                if image is None or image.size == 0:
                    debug_image_failed += 1
                    if i < 5:
                        logger.info(f"❌ Sample {i}: Image load failed")
                    continue

                # Expert image preprocessing
                processed_image = self._expert_preprocess_image(image)
                if processed_image is not None:
                    images.append(processed_image)
                    texts.append(text)

                    # Track class distribution
                    class_name = sample.get("class_name", "unknown")
                    classes.append(class_name)
                    class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                    debug_success += 1

                    if i < 5:
                        logger.info(
                            f"✅ Sample {i}: Success - '{text}' -> shape {processed_image.shape}"
                        )
                else:
                    debug_preprocess_failed += 1
                    if i < 5:
                        logger.info(f"❌ Sample {i}: Preprocessing failed")

            except Exception as e:
                debug_image_failed += 1
                if i < 5:
                    logger.info(f"❌ Sample {i}: Exception: {e}")
                continue

        logger.info(f"✅ Processed {len(images)} valid samples")
        logger.info(f"📊 Class distribution: {class_distribution}")
        logger.info(
            f"🔍 Debug stats: no_image={debug_no_image}, no_text={debug_no_text}, invalid_chars={debug_invalid_chars}, image_failed={debug_image_failed}, preprocess_failed={debug_preprocess_failed}, success={debug_success}"
        )

        if len(images) == 0:
            raise ValueError("No valid training samples found!")

        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y_encoded = self._encode_texts_ctc(texts)

        # Split data strategically
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, shuffle=True
        )

        logger.info(f"🎯 Training samples: {len(X_train):,}")
        logger.info(f"🎯 Validation samples: {len(X_val):,}")

        # Create optimized TensorFlow datasets
        train_dataset = self._create_optimized_dataset(X_train, y_train, is_training=True)
        val_dataset = self._create_optimized_dataset(X_val, y_val, is_training=False)

        return train_dataset, val_dataset, class_distribution

    def _expert_preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Expert-level image preprocessing for maximum OCR accuracy"""
        if image is None or image.size == 0:
            return None

        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Advanced preprocessing pipeline
        # 1. Noise reduction with bilateral filter
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # 3. Morphological operations to clean text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # 4. Resize with high-quality interpolation
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

        # 5. Normalize to [0, 1] range
        image = image.astype(np.float32) / 255.0

        # 6. Add channel dimension
        image = np.expand_dims(image, axis=-1)

        return image

    def _encode_texts_ctc(self, texts: List[str]) -> np.ndarray:
        """Encode texts for CTC loss (variable length sequences)"""
        encoded_texts = []

        for text in texts:
            # Convert text to character indices (skip blank token)
            encoded = [self.char_to_idx[char] for char in text if char in self.char_to_idx]
            encoded_texts.append(encoded)

        # Pad sequences to max length for batch processing
        max_len = max(len(seq) for seq in encoded_texts)
        padded_sequences = []

        for seq in encoded_texts:
            padded = seq + [self.blank_token] * (max_len - len(seq))
            padded_sequences.append(padded)

        return np.array(padded_sequences, dtype=np.int32)

    def _create_optimized_dataset(
        self, X: np.ndarray, y: np.ndarray, is_training: bool
    ) -> tf.data.Dataset:
        """Create highly optimized TensorFlow dataset for maximum GPU utilization"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if is_training:
            # Shuffle with large buffer for better randomization
            dataset = dataset.shuffle(buffer_size=min(10000, len(X)))

            # Data augmentation for training
            dataset = dataset.map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and optimize
        batch_size = 32 if is_training else 64  # Larger batch for validation
        dataset = dataset.batch(batch_size)

        # Prefetch for GPU pipeline optimization
        dataset = dataset.prefetch(self.prefetch_buffer)

        return dataset

    def _augment_data(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Lightweight data augmentation for better generalization"""
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Ensure values stay in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    def build_expert_model(self) -> keras.Model:
        """Build expert-level CNN+LSTM+CTC model with state-of-the-art architecture"""
        logger.info("🏗️ Building expert unified OCR model...")

        # Input layer
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1), name="image_input")

        # Expert CNN feature extractor with residual connections
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_1")(inputs)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_1")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3_1")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4_1")(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv4_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool4")(x)
        x = layers.Dropout(0.2)(x)

        # Reshape for sequence processing
        new_shape = ((self.img_width // 16), (self.img_height // 16) * 256)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        # Bidirectional LSTM layers for sequence modeling
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.2), name="bilstm1"
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2), name="bilstm2"
        )(x)

        # Dense layer for character prediction
        x = layers.Dense(512, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="character_output")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="ExpertMaddenOCR")

        # Expert optimizer configuration
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001, weight_decay=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )

        # Compile with CTC loss
        model.compile(
            optimizer=optimizer,
            loss=self._ctc_loss_func,
            metrics=["accuracy"],
            jit_compile=self.use_xla,  # XLA compilation for speed
        )

        logger.info(f"🎯 Model built with {model.count_params():,} parameters")
        return model

    def _ctc_loss_func(self, y_true, y_pred):
        """CTC loss function for variable-length sequences"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def train_model(
        self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, epochs: int = 50
    ) -> keras.callbacks.History:
        """Train the expert model with optimal settings"""
        logger.info(f"🚀 Starting expert training for {epochs} epochs...")

        # Build model
        self.model = self.build_expert_model()

        # Expert callbacks for optimal training
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                str(self.model_save_path / "best_model.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            # Adaptive learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1, cooldown=2
            ),
            # Early stopping with patience
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
            ),
            # Learning rate scheduling
            keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.95**epoch), verbose=0),
            # Memory cleanup
            keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect()),
        ]

        # Train with optimal settings
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=True,
            workers=4,
        )

        # Save final model and artifacts
        self._save_model_artifacts(history)

        logger.info("✅ Expert training completed!")
        return history

    def _save_model_artifacts(self, history: keras.callbacks.History):
        """Save all model artifacts for production deployment"""
        # Save final model
        self.model.save(str(self.model_save_path / "final_model.h5"))

        # Save training history
        with open(self.model_save_path / "training_history.json", "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)

        # Save character mappings
        char_mappings = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "madden_chars": self.madden_chars,
            "num_classes": self.num_classes,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "max_text_length": self.max_text_length,
        }
        with open(self.model_save_path / "model_config.json", "w") as f:
            json.dump(char_mappings, f, indent=2)

        # Save model summary
        with open(self.model_save_path / "model_summary.txt", "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        logger.info(f"💾 Model artifacts saved to: {self.model_save_path}")


def main():
    print("🎯 Expert Unified OCR Trainer - Maximum Performance")
    print("=" * 60)

    # System info
    print(
        f"🖥️ System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total // (1024**3)} GB RAM"
    )
    print(f"🎮 GPU: {len(tf.config.experimental.list_physical_devices('GPU'))} device(s)")

    # Initialize trainer
    trainer = ExpertOCRTrainer()

    try:
        # Load and preprocess data
        json_files = list(Path(".").glob("madden_ocr_training_data_*.json"))
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Loading data from: {latest_json}")
            train_ds, val_ds, class_dist = trainer.load_and_preprocess_data(str(latest_json))
        else:
            print("📁 Loading data from database...")
            train_ds, val_ds, class_dist = trainer.load_and_preprocess_data()

        print(f"✅ Data loaded and optimized")
        print(f"📊 Class distribution: {class_dist}")

        # Train model
        print("\n🚀 Starting expert training...")
        history = trainer.train_model(train_ds, val_ds, epochs=40)

        # Training summary
        final_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        best_val_loss = min(history.history["val_loss"])

        print(f"\n✅ Training completed!")
        print(f"📈 Final training loss: {final_loss:.4f}")
        print(f"📈 Final validation loss: {final_val_loss:.4f}")
        print(f"🎯 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {trainer.model_save_path}")
        print(f"🎉 Ready for production deployment!")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Training failed")


if __name__ == "__main__":
    main()

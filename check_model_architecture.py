#!/usr/bin/env python3
"""
Check the actual architecture of the saved custom OCR model.
"""

from pathlib import Path

import torch


def check_model_architecture():
    """Check the saved model architecture."""

    model_path = Path("models/fixed_ocr_20250614_150024/best_fixed_model.pth")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"🔍 Examining model: {model_path}")
    print("=" * 60)

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Check available keys
    print("\n📋 Checkpoint Keys:")
    for key in checkpoint.keys():
        print(f"  • {key}")

    # Check model state dict structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\n🏗️  Model State Dict ({len(state_dict)} parameters):")

        # Group by layer type
        feature_layers = []
        sequence_layers = []
        output_layers = []

        for key in sorted(state_dict.keys()):
            if key.startswith("feature_extractor"):
                feature_layers.append(key)
            elif key.startswith("sequence_processor"):
                sequence_layers.append(key)
            elif key.startswith("output_classifier"):
                output_layers.append(key)
            else:
                print(f"  ❓ Unknown layer: {key}")

        print(f"\n🔧 Feature Extractor ({len(feature_layers)} params):")
        for layer in feature_layers:
            shape = state_dict[layer].shape if hasattr(state_dict[layer], "shape") else "scalar"
            print(f"  • {layer}: {shape}")

        print(f"\n🔄 Sequence Processor ({len(sequence_layers)} params):")
        for layer in sequence_layers:
            shape = state_dict[layer].shape if hasattr(state_dict[layer], "shape") else "scalar"
            print(f"  • {layer}: {shape}")

        print(f"\n📤 Output Classifier ({len(output_layers)} params):")
        for layer in output_layers:
            shape = state_dict[layer].shape if hasattr(state_dict[layer], "shape") else "scalar"
            print(f"  • {layer}: {shape}")

    # Check metadata
    print(f"\n📊 Model Metadata:")
    for key in ["vocab_size", "epoch", "validation_loss", "training_id"]:
        if key in checkpoint:
            print(f"  • {key}: {checkpoint[key]}")

    # Analyze feature extractor structure
    if "model_state_dict" in checkpoint:
        print(f"\n🔍 Feature Extractor Analysis:")
        state_dict = checkpoint["model_state_dict"]

        # Find all conv layers
        conv_layers = [
            k
            for k in state_dict.keys()
            if "feature_extractor" in k and "weight" in k and "conv" not in k.lower()
        ]
        conv_layers = [
            k for k in conv_layers if len(state_dict[k].shape) == 4
        ]  # Conv2d weights have 4 dimensions

        print(f"  Found {len(conv_layers)} Conv2d layers:")
        for i, layer in enumerate(conv_layers):
            weight_shape = state_dict[layer].shape
            print(
                f"    Layer {i+1}: {layer} -> {weight_shape} (in_ch={weight_shape[1]}, out_ch={weight_shape[0]}, kernel={weight_shape[2]}x{weight_shape[3]})"
            )

        # Find LSTM layers
        lstm_layers = [k for k in state_dict.keys() if "sequence_processor" in k and "weight" in k]
        print(f"\n  Found {len(lstm_layers)} LSTM parameters:")
        for layer in lstm_layers:
            weight_shape = state_dict[layer].shape
            print(f"    {layer}: {weight_shape}")


if __name__ == "__main__":
    check_model_architecture()

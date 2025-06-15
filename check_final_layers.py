#!/usr/bin/env python3
"""
Check Final Layers
"""

import torch


def check_final_layers():
    print("🔍 Checking final layers...")

    checkpoint = torch.load(
        "models/pytorch_madden_ocr/best_model.pth", map_location="cpu", weights_only=False
    )

    print("📊 Final layers (cnn.20+, fc, classifier):")
    for key, tensor in checkpoint["model_state_dict"].items():
        if any(x in key for x in ["cnn.2", "fc", "classifier"]):
            print(f"  {key}: {tensor.shape}")

    print(f"\n📊 All CNN layers:")
    for key, tensor in checkpoint["model_state_dict"].items():
        if "cnn." in key and "weight" in key:
            print(f"  {key}: {tensor.shape}")


if __name__ == "__main__":
    check_final_layers()

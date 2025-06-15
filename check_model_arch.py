#!/usr/bin/env python3
"""
Check Model Architecture
"""

import torch


def check_model():
    print("🔍 Checking saved model architecture...")

    checkpoint = torch.load(
        "models/pytorch_madden_ocr/best_model.pth", map_location="cpu", weights_only=False
    )

    print("📊 Model state dict keys:")
    for key, tensor in checkpoint["model_state_dict"].items():
        print(f"  {key}: {tensor.shape}")

    print(f"\n📊 Other checkpoint keys:")
    for key in checkpoint.keys():
        if key != "model_state_dict":
            print(f"  {key}: {checkpoint[key]}")


if __name__ == "__main__":
    check_model()

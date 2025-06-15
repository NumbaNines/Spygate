"""Model configuration for SpygateAI."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Model configuration
MODEL_CONFIG = {
    "path": str(
        PROJECT_ROOT / "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
    ),
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "max_detections": 1000,
    "image_size": 640,
    "half_precision": True,  # Use FP16 for speed
}

# Hardware configuration
HARDWARE_CONFIG = {
    "device": "cuda",  # Use GPU
    "workers": 8,  # Number of worker threads
    "batch_size": 12,  # Batch size for RTX 4070
}

# UI configuration
UI_CONFIG = {
    "window_title": "SpygateAI",
    "theme": "dark",
    "animation_fps": 1.0,  # Animation frame rate
    "window_size": (1636, 938),  # Default window size
}

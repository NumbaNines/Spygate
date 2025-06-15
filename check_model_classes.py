#!/usr/bin/env python3
"""Check what classes are in the 8-class model."""

from ultralytics import YOLO

model_path = "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"

try:
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")

    # Print each class with its ID
    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

except Exception as e:
    print(f"Error loading model: {e}")

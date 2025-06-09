"""Test YOLO11 model on extracted Madden frames with hardware-aware processing."""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import GPUtil
import psutil
import torch
from ultralytics import YOLO


def detect_hardware_tier():
    """Detect hardware capabilities and return appropriate tier settings."""
    try:
        # Get system info
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=False)

        # Try to get GPU info
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory = gpu.memoryTotal / 1024  # Convert to GB
                gpu_name = gpu.name
            else:
                gpu_memory = 0
                gpu_name = "Integrated"
        except:
            gpu_memory = 0
            gpu_name = "Unknown"

        # Determine tier based on hardware
        if ram_gb >= 32 and gpu_memory >= 12 and cpu_cores >= 8:
            return "professional", 1.0  # Full resolution
        elif ram_gb >= 16 and gpu_memory >= 8 and cpu_cores >= 6:
            return "premium", 0.75  # 75% resolution
        elif ram_gb >= 12 and gpu_memory >= 4 and cpu_cores >= 4:
            return "standard", 0.5  # 50% resolution
        else:
            return "minimum", 0.25  # 25% resolution

    except Exception as e:
        print(f"Error detecting hardware: {e}")
        return "minimum", 0.25  # Default to minimum tier


def load_classes(classes_file: str) -> list:
    """Load class names from file."""
    with open(classes_file) as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]


def test_model(
    model_path: str, test_dir: str, output_dir: str, classes_file: str, conf_threshold: float = 0.25
):
    """Run YOLO11 model on test images and save results.

    Args:
        model_path: Path to YOLO11 model weights
        test_dir: Directory containing test images
        output_dir: Directory to save results
        classes_file: Path to file containing class names
        conf_threshold: Confidence threshold for detections
    """
    # Detect hardware tier and get scaling factor
    tier, scale_factor = detect_hardware_tier()
    print(f"Detected hardware tier: {tier} (scale factor: {scale_factor})")

    # Load model
    model = YOLO(model_path)

    # Load class names
    class_names = load_classes(classes_file)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get test images
    test_images = list(Path(test_dir).glob("*.png"))
    if not test_images:
        raise ValueError(f"No PNG images found in {test_dir}")

    # Process each image
    results = []
    for img_path in test_images:
        # Read and resize image based on hardware tier
        img = cv2.imread(str(img_path))
        if scale_factor < 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            img = cv2.resize(img, (new_w, new_h))

        # Run inference
        pred = model(img, conf=conf_threshold)[0]

        # Get predictions
        boxes = pred.boxes
        img_results = {
            "image": str(img_path),
            "hardware_tier": tier,
            "scale_factor": scale_factor,
            "detections": [],
        }

        # Process each detection
        for box in boxes:
            # Get box coordinates and scale back to original size if needed
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if scale_factor < 1.0:
                x1, x2 = x1 / scale_factor, x2 / scale_factor
                y1, y2 = y1 / scale_factor, y2 / scale_factor

            # Get class and confidence
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Get class name
            class_name = class_names[cls] if cls < len(class_names) else f"unknown_{cls}"

            detection = {
                "class": cls,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
            img_results["detections"].append(detection)

        results.append(img_results)

        # Draw predictions on original size image
        img = cv2.imread(str(img_path))
        for det in img_results["detections"]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label with class name and confidence
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save annotated image
        output_img = os.path.join(output_dir, f"pred_{img_path.name}")
        cv2.imwrite(output_img, img)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(
            {"hardware_tier": tier, "scale_factor": scale_factor, "results": results}, f, indent=2
        )

    print(f"Processed {len(test_images)} images")
    print(f"Results saved to {results_file}")


def main():
    # Setup paths
    model_path = "runs/detect/train5/weights/best.pt"
    test_dir = "test_dataset/images/madden_test"
    output_dir = "test_dataset/results"
    classes_file = "test_dataset/classes.txt"

    # Run test
    try:
        test_model(model_path, test_dir, output_dir, classes_file, conf_threshold=0.25)
    except Exception as e:
        print(f"Error testing model: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize detections from trained YOLOv8 HUD detection model."""

import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")  # Use non-interactive backend to avoid display issues
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_model(model_path):
    """Load the trained YOLOv8 model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    return model


def visualize_detections(
    image_path, model, save_path=None, show_confidence=True, confidence_threshold=0.01
):
    """Visualize detections on a single image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection with very low confidence to see all detections
    results = model(image, conf=confidence_threshold)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_rgb)

    detection_count = 0
    low_conf_count = 0
    high_conf_count = 0

    # Process detections
    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # Define colors for each class
            colors = ["red", "blue", "green", "orange", "purple", "yellow", "pink", "cyan"]

            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                # Choose color based on class
                color = colors[cls % len(colors)]

                # Different line styles based on confidence
                if score >= 0.5:
                    linewidth = 3
                    linestyle = "-"  # Solid line for high confidence
                    high_conf_count += 1
                elif score >= 0.25:
                    linewidth = 2
                    linestyle = "--"  # Dashed line for medium confidence
                else:
                    linewidth = 1
                    linestyle = ":"  # Dotted line for low confidence
                    low_conf_count += 1

                # Draw bounding box
                rect = Rectangle(
                    (x1, y1),
                    width,
                    height,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor="none",
                    linestyle=linestyle,
                )
                ax.add_patch(rect)

                # Get class name
                class_names = ["hud", "qb_position", "left_hash_mark", "right_hash_mark"]
                class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

                # Add label with confidence styling
                if show_confidence:
                    label = f"{class_name}: {score:.3f}"
                else:
                    label = class_name

                # Style label background based on confidence
                if score >= 0.5:
                    alpha = 0.8
                    text_color = "white"
                elif score >= 0.25:
                    alpha = 0.6
                    text_color = "white"
                else:
                    alpha = 0.4
                    text_color = "black"

                ax.text(
                    x1,
                    y1 - 10,
                    label,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha),
                    fontsize=9,
                    color=text_color,
                    weight="bold",
                )

                detection_count += 1

    # Enhanced title with confidence breakdown
    title = f"HUD Detections: {image_path.name}\n"
    title += f"Total: {detection_count} (High conf ≥0.5: {high_conf_count}, Low conf <0.25: {low_conf_count})"
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Save the visualization
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization: {save_path}")

    plt.close()  # Close figure to free memory
    return {"total": detection_count, "high_conf": high_conf_count, "low_conf": low_conf_count}


def main():
    """Main function to visualize detections on random images."""
    print("🔍 Visualizing HUD Detections (LOW CONFIDENCE TEST)")
    print("=" * 60)

    # Model path
    model_path = "runs/detect/spygate_hud_detection_fast/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Load model
    try:
        model = load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Image directory
    image_dir = Path("training_data/images")
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return

    # Get all images
    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        print("❌ No PNG images found in training_data/images/")
        return

    print(f"📂 Found {len(image_files)} images")

    # Create output directory
    output_dir = Path("detection_visualizations_lowconf")
    output_dir.mkdir(exist_ok=True)

    # Select random images (fewer for low confidence test)
    num_images = min(10, len(image_files))  # Process 10 images for low confidence test
    selected_images = random.sample(image_files, num_images)

    print(f"🎯 Processing {num_images} random images with VERY LOW confidence threshold (0.01)...")
    print("📝 Legend: Solid line (≥0.5), Dashed line (≥0.25), Dotted line (<0.25)")

    total_detections = 0
    total_high_conf = 0
    total_low_conf = 0
    successful_visualizations = 0

    for i, image_path in enumerate(selected_images, 1):
        try:
            save_path = output_dir / f"lowconf_detection_{i}_{image_path.name}"
            detection_stats = visualize_detections(
                image_path, model, save_path, show_confidence=True, confidence_threshold=0.01
            )

            if detection_stats is not None:
                total_detections += detection_stats["total"]
                total_high_conf += detection_stats["high_conf"]
                total_low_conf += detection_stats["low_conf"]
                successful_visualizations += 1

                print(
                    f"✅ {i:2d}/{num_images}: {detection_stats['total']} total "
                    f"(high: {detection_stats['high_conf']}, low: {detection_stats['low_conf']}) "
                    f"in {image_path.name}"
                )
            else:
                print(f"❌ {i:2d}/{num_images}: Failed to process {image_path.name}")

        except Exception as e:
            print(f"❌ {i:2d}/{num_images}: Error processing {image_path.name}: {e}")

    print("\n" + "=" * 60)
    print("📊 LOW CONFIDENCE DETECTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful_visualizations}/{num_images} images")
    print(f"🎯 Total detections found: {total_detections}")
    print(f"🔥 High confidence (≥0.5): {total_high_conf}")
    print(f"🤔 Low confidence (<0.25): {total_low_conf}")
    print(f"📈 Average detections per image: {total_detections/successful_visualizations:.1f}")
    print(f"📁 Visualizations saved in: {output_dir}")

    if total_detections > 0:
        high_conf_percent = (total_high_conf / total_detections) * 100
        low_conf_percent = (total_low_conf / total_detections) * 100

        print(f"\n📈 CONFIDENCE BREAKDOWN:")
        print(f"   High confidence (≥0.5): {high_conf_percent:.1f}%")
        print(f"   Low confidence (<0.25): {low_conf_percent:.1f}%")

        print("\n🔍 Analysis:")
        if total_high_conf > 0:
            print("   ✅ Model is making confident detections - this is good!")
        if total_low_conf > total_high_conf:
            print("   📝 Many low confidence detections - model is being cautious")
            print("   💡 Consider if these low-conf detections are actually correct")


if __name__ == "__main__":
    main()

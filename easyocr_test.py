#!/usr/bin/env python3
"""Test EasyOCR on HUD images to see what text can be extracted with visual highlights."""

import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import easyocr

    OCR_AVAILABLE = True
    print("✅ EasyOCR imported successfully")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"❌ EasyOCR import failed: {e}")


def create_ocr_reader():
    """Create EasyOCR reader for English text."""
    if not OCR_AVAILABLE:
        return None

    try:
        # Initialize EasyOCR reader for English
        print("🔄 Initializing EasyOCR reader (this may take a moment)...")
        reader = easyocr.Reader(["en"], gpu=True)  # Use GPU if available
        print("✅ EasyOCR reader initialized successfully")
        return reader
    except Exception as e:
        print(f"❌ Failed to initialize EasyOCR reader: {e}")
        return None


def extract_text_with_highlights(image_path, reader, confidence_threshold=0.1):
    """Extract text from image and return both text and bounding box data for visualization."""
    if not reader:
        return [], []

    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return [], []

        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract text with bounding boxes
        results = reader.readtext(rgb_image, detail=1, paragraph=False)

        # Filter by confidence and format results
        text_detections = []
        bounding_boxes = []

        for bbox, text, confidence in results:
            if confidence >= confidence_threshold:
                # Extract bounding box coordinates
                # bbox is in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                text_detections.append(
                    {
                        "text": text.strip(),
                        "confidence": confidence,
                        "bbox": bbox,
                        "rect": (x_min, y_min, x_max - x_min, y_max - y_min),
                    }
                )

                bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return text_detections, rgb_image

    except Exception as e:
        print(f"❌ OCR extraction failed for {image_path}: {e}")
        return [], []


def create_ocr_visualization(image, text_detections, output_path, image_name):
    """Create a visualization showing detected text with bounding boxes."""
    if len(text_detections) == 0:
        print(f"⚠️ No text detected in {image_name}")
        return

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Display image
    ax.imshow(image)
    ax.set_title(
        f"OCR Text Detection: {image_name}\n{len(text_detections)} text regions detected",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Color map for different confidence levels
    colors = ["red", "orange", "yellow", "lightgreen", "green"]

    # Add bounding boxes and text labels
    legend_elements = []
    confidence_ranges = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.1, 0.3)]
    confidence_labels = [
        "Very High (0.9-1.0)",
        "High (0.7-0.9)",
        "Medium (0.5-0.7)",
        "Low (0.3-0.5)",
        "Very Low (0.1-0.3)",
    ]

    for i, detection in enumerate(text_detections):
        x, y, w, h = detection["rect"]
        confidence = detection["confidence"]
        text = detection["text"]

        # Choose color based on confidence
        color_idx = 0
        for idx, (min_conf, max_conf) in enumerate(confidence_ranges):
            if min_conf <= confidence <= max_conf:
                color_idx = idx
                break

        color = colors[color_idx]

        # Draw bounding box
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none", alpha=0.8)
        ax.add_patch(rect)

        # Add text label with background
        label = f"{text} ({confidence:.2f})"
        ax.text(
            x,
            y - 5,
            label,
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

    # Create legend
    for i, (color, label) in enumerate(zip(colors, confidence_labels)):
        legend_elements.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"💾 OCR visualization saved: {output_path}")


def test_ocr_on_sample_images(sample_size=25):
    """Test OCR on a sample of images with visual output."""

    # Initialize OCR reader
    reader = create_ocr_reader()
    if not reader:
        print("❌ Cannot proceed without OCR reader")
        return

    # Find training images
    images_dir = Path("training_data/images")
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    # Get all PNG files
    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print(f"❌ No PNG files found in {images_dir}")
        return

    print(f"📁 Found {len(image_files)} images in {images_dir}")

    # Sample random images
    if len(image_files) > sample_size:
        sample_files = random.sample(image_files, sample_size)
    else:
        sample_files = image_files
        sample_size = len(image_files)

    print(f"🎯 Testing OCR on {sample_size} random images...")

    # Create output directory
    output_dir = Path("ocr_visualizations_large")
    output_dir.mkdir(exist_ok=True)

    # Track overall statistics
    total_detections = 0
    images_with_text = 0
    all_detected_text = []
    confidence_stats = []

    # Process each image
    for i, image_path in enumerate(sample_files, 1):
        print(f"\n🔍 Processing {i}/{sample_size}: {image_path.name}")

        # Extract text with OCR
        text_detections, rgb_image = extract_text_with_highlights(image_path, reader)

        if len(text_detections) > 0:
            images_with_text += 1
            total_detections += len(text_detections)

            # Collect statistics
            for detection in text_detections:
                all_detected_text.append(detection["text"])
                confidence_stats.append(detection["confidence"])

            # Create visualization
            output_path = output_dir / f"ocr_large_{i:02d}_{image_path.stem}.png"
            create_ocr_visualization(rgb_image, text_detections, output_path, image_path.name)

            # Print text found in this image
            print(f"📝 Found {len(text_detections)} text regions:")
            for detection in text_detections:
                print(f"   • '{detection['text']}' (confidence: {detection['confidence']:.3f})")

        else:
            print("   ⚪ No text detected")

    # Print comprehensive summary
    print(f"\n{'='*60}")
    print(f"🎉 OCR ANALYSIS COMPLETE - LARGE SAMPLE")
    print(f"{'='*60}")
    print(f"📊 SAMPLE STATISTICS:")
    print(f"   • Images processed: {sample_size}")
    print(f"   • Images with text: {images_with_text} ({images_with_text/sample_size*100:.1f}%)")
    print(f"   • Total text detections: {total_detections}")
    print(f"   • Average detections per image: {total_detections/sample_size:.1f}")
    print(
        f"   • Average detections per image with text: {total_detections/max(1,images_with_text):.1f}"
    )

    if confidence_stats:
        print(f"\n📈 CONFIDENCE STATISTICS:")
        print(f"   • Average confidence: {np.mean(confidence_stats):.3f}")
        print(f"   • Median confidence: {np.median(confidence_stats):.3f}")
        print(f"   • Min confidence: {min(confidence_stats):.3f}")
        print(f"   • Max confidence: {max(confidence_stats):.3f}")

        # Confidence distribution
        high_conf = sum(1 for c in confidence_stats if c >= 0.8)
        med_conf = sum(1 for c in confidence_stats if 0.5 <= c < 0.8)
        low_conf = sum(1 for c in confidence_stats if c < 0.5)

        print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
        print(
            f"   • High confidence (≥0.8): {high_conf} ({high_conf/len(confidence_stats)*100:.1f}%)"
        )
        print(
            f"   • Medium confidence (0.5-0.8): {med_conf} ({med_conf/len(confidence_stats)*100:.1f}%)"
        )
        print(f"   • Low confidence (<0.5): {low_conf} ({low_conf/len(confidence_stats)*100:.1f}%)")

    if all_detected_text:
        print(f"\n🔤 UNIQUE TEXT DETECTED:")
        unique_texts = list({text.upper() for text in all_detected_text})
        unique_texts.sort()

        # Group by type
        scores = [t for t in unique_texts if t.isdigit()]
        times = [t for t in unique_texts if ":" in t]
        downs = [
            t
            for t in unique_texts
            if any(word in t.lower() for word in ["1st", "2nd", "3rd", "4th", "&"])
        ]
        teams = [
            t for t in unique_texts if len(t) <= 4 and t.isalpha() and t not in ["THE", "AND", "OF"]
        ]
        others = [t for t in unique_texts if t not in scores + times + downs + teams]

        if scores:
            print(f"   📊 Scores: {', '.join(scores[:10])}{'...' if len(scores) > 10 else ''}")
        if times:
            print(f"   ⏰ Times: {', '.join(times[:10])}{'...' if len(times) > 10 else ''}")
        if downs:
            print(f"   🏈 Down/Distance: {', '.join(downs[:5])}{'...' if len(downs) > 5 else ''}")
        if teams:
            print(f"   🏟️ Teams: {', '.join(teams[:10])}{'...' if len(teams) > 10 else ''}")
        if others:
            print(f"   📝 Other: {', '.join(others[:15])}{'...' if len(others) > 15 else ''}")

    print(f"\n💾 Visualizations saved in: {output_dir}")
    print(f"📁 Check the PNG files to see highlighted text regions!")


if __name__ == "__main__":
    test_ocr_on_sample_images(sample_size=25)

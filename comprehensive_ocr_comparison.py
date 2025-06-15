#!/usr/bin/env python3
"""
Comprehensive OCR Comparison: PaddleOCR vs KerasOCR
Tests on ALL images in 6.12 screenshots folder with visual output
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import at module level to avoid scope issues
try:
    import keras_ocr

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("❌ KerasOCR not available")

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("❌ PaddleOCR not available")


def initialize_ocr_engines():
    """Initialize both OCR engines"""
    paddle_ocr = None
    keras_pipeline = None

    # Initialize PaddleOCR
    if PADDLE_AVAILABLE:
        try:
            print("🔧 Initializing PaddleOCR...")
            paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,  # Force CPU mode
                show_log=False,
                enable_mkldnn=False,
            )
            print("✅ PaddleOCR initialized successfully (CPU mode)!")
        except Exception as e:
            print(f"❌ PaddleOCR failed: {e}")

    # Initialize KerasOCR
    if KERAS_AVAILABLE:
        try:
            print("🔧 Initializing KerasOCR...")
            keras_pipeline = keras_ocr.pipeline.Pipeline()
            print("✅ KerasOCR initialized successfully!")
        except Exception as e:
            print(f"❌ KerasOCR failed: {e}")

    return paddle_ocr, keras_pipeline


def extract_text_paddle(ocr, image_path):
    """Extract text using PaddleOCR with bounding boxes"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Could not load image"

        result = ocr.ocr(image, cls=False)

        if not result or not result[0]:
            return [], "No text detected"

        detections = []
        for line in result[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # Bounding box coordinates
                text = line[1][0]  # Text
                confidence = line[1][1]  # Confidence

                detections.append({"text": text, "confidence": confidence, "bbox": bbox})

        return detections, None

    except Exception as e:
        return None, f"PaddleOCR error: {str(e)}"


def extract_text_keras(pipeline, image_path):
    """Extract text using KerasOCR with bounding boxes"""
    try:
        image = keras_ocr.tools.read(str(image_path))
        prediction_groups = pipeline.recognize([image])
        predictions = prediction_groups[0]

        if not predictions:
            return [], "No text detected"

        detections = []
        for text, box in predictions:
            # Convert box format to match PaddleOCR
            bbox = box.tolist()
            detections.append(
                {
                    "text": text,
                    "confidence": 1.0,  # KerasOCR doesn't provide confidence
                    "bbox": bbox,
                }
            )

        return detections, None

    except Exception as e:
        return None, f"KerasOCR error: {str(e)}"


def create_visual_comparison(image_path, paddle_results, keras_results, output_path):
    """Create side-by-side visual comparison of OCR results"""
    try:
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return False

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # PaddleOCR results (left)
        ax1.imshow(image_rgb)
        ax1.set_title(f"PaddleOCR Results", fontsize=16, fontweight="bold")
        ax1.axis("off")

        if paddle_results and isinstance(paddle_results, list):
            for i, detection in enumerate(paddle_results):
                bbox = detection["bbox"]
                text = detection["text"]
                conf = detection["confidence"]

                # Draw bounding box
                if len(bbox) == 4 and len(bbox[0]) == 2:
                    # Convert to rectangle format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax1.add_patch(rect)

                    # Add text label
                    ax1.text(
                        x_min,
                        y_min - 5,
                        f"{text} ({conf:.2f})",
                        fontsize=10,
                        color="red",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

        # KerasOCR results (right)
        ax2.imshow(image_rgb)
        ax2.set_title(f"KerasOCR Results", fontsize=16, fontweight="bold")
        ax2.axis("off")

        if keras_results and isinstance(keras_results, list):
            for i, detection in enumerate(keras_results):
                bbox = detection["bbox"]
                text = detection["text"]

                # Draw bounding box
                if len(bbox) == 4 and len(bbox[0]) == 2:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    rect = patches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor="blue",
                        facecolor="none",
                    )
                    ax2.add_patch(rect)

                    # Add text label
                    ax2.text(
                        x_min,
                        y_min - 5,
                        f"{text}",
                        fontsize=10,
                        color="blue",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return True

    except Exception as e:
        print(f"❌ Error creating visual comparison: {e}")
        return False


def main():
    print("🚀 COMPREHENSIVE OCR COMPARISON: PaddleOCR vs KerasOCR")
    print("📁 Testing ALL images in '6.12 screenshots' folder")
    print("=" * 80)

    # Initialize OCR engines
    paddle_ocr, keras_pipeline = initialize_ocr_engines()

    if not paddle_ocr and not keras_pipeline:
        print("❌ Both OCR engines failed to initialize!")
        return

    # Get all images from 6.12 screenshots folder
    screenshots_folder = Path("6.12 screenshots")
    if not screenshots_folder.exists():
        print(f"❌ Folder '{screenshots_folder}' not found!")
        return

    # Get all PNG files
    image_files = list(screenshots_folder.glob("*.png"))
    image_files.sort()  # Sort by filename

    if not image_files:
        print("❌ No PNG files found in the screenshots folder!")
        return

    print(f"📸 Found {len(image_files)} images to process...")

    # Create output directory
    output_dir = Path("comprehensive_ocr_comparison_results")
    output_dir.mkdir(exist_ok=True)

    # Results tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(image_files),
        "paddle_success": 0,
        "keras_success": 0,
        "detailed_results": [],
    }

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\n🖼️  Processing {i}/{len(image_files)}: {image_path.name}")
        print("-" * 60)

        # Extract text with both engines
        paddle_results = None
        keras_results = None
        paddle_error = None
        keras_error = None

        if paddle_ocr:
            paddle_results, paddle_error = extract_text_paddle(paddle_ocr, image_path)
            if paddle_error:
                print(f"🟦 PaddleOCR: ❌ {paddle_error}")
            else:
                results["paddle_success"] += 1
                paddle_texts = [d["text"] for d in paddle_results] if paddle_results else []
                print(
                    f"🟦 PaddleOCR: ✅ Found {len(paddle_results)} text regions: {' | '.join(paddle_texts[:3])}{'...' if len(paddle_texts) > 3 else ''}"
                )

        if keras_pipeline:
            keras_results, keras_error = extract_text_keras(keras_pipeline, image_path)
            if keras_error:
                print(f"🟩 KerasOCR:  ❌ {keras_error}")
            else:
                results["keras_success"] += 1
                keras_texts = [d["text"] for d in keras_results] if keras_results else []
                print(
                    f"🟩 KerasOCR:  ✅ Found {len(keras_results)} text regions: {' | '.join(keras_texts[:3])}{'...' if len(keras_texts) > 3 else ''}"
                )

        # Create visual comparison
        output_image_path = output_dir / f"comparison_{i:03d}_{image_path.stem}.png"
        visual_success = create_visual_comparison(
            image_path, paddle_results, keras_results, output_image_path
        )

        if visual_success:
            print(f"📊 Visual comparison saved: {output_image_path.name}")

        # Store detailed results
        image_result = {
            "image_name": image_path.name,
            "image_index": i,
            "paddle_success": paddle_error is None,
            "keras_success": keras_error is None,
            "paddle_text_count": len(paddle_results) if paddle_results else 0,
            "keras_text_count": len(keras_results) if keras_results else 0,
            "paddle_texts": [d["text"] for d in paddle_results] if paddle_results else [],
            "keras_texts": [d["text"] for d in keras_results] if keras_results else [],
            "paddle_error": paddle_error,
            "keras_error": keras_error,
            "visual_output": str(output_image_path) if visual_success else None,
        }
        results["detailed_results"].append(image_result)

        # Progress indicator
        if i % 10 == 0 or i == len(image_files):
            progress = (i / len(image_files)) * 100
            print(f"📈 Progress: {progress:.1f}% ({i}/{len(image_files)})")

    # Save detailed results
    results_file = output_dir / "detailed_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL COMPREHENSIVE RESULTS")
    print("=" * 80)

    total_images = results["total_images"]
    paddle_success = results["paddle_success"]
    keras_success = results["keras_success"]

    if paddle_ocr:
        paddle_rate = (paddle_success / total_images) * 100
        print(f"🟦 PaddleOCR: {paddle_success}/{total_images} ({paddle_rate:.1f}% success rate)")
    else:
        print("🟦 PaddleOCR: ❌ Not available")

    if keras_pipeline:
        keras_rate = (keras_success / total_images) * 100
        print(f"🟩 KerasOCR:  {keras_success}/{total_images} ({keras_rate:.1f}% success rate)")
    else:
        print("🟩 KerasOCR:  ❌ Not available")

    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📄 Detailed JSON report: {results_file}")
    print(f"🖼️  Visual comparisons: {len(list(output_dir.glob('comparison_*.png')))} images")

    # Recommendation
    print("\n🎯 RECOMMENDATION")
    print("-" * 30)

    if paddle_ocr and keras_pipeline:
        if paddle_success > keras_success:
            print("🏆 PaddleOCR performed better overall!")
        elif keras_success > paddle_success:
            print("🏆 KerasOCR performed better overall!")
        else:
            print("🤝 Both engines performed equally well!")

        print(f"\n💡 Both engines are functional - you can use either or both in SpygateAI!")
        print(f"   PaddleOCR provides confidence scores, KerasOCR provides detailed segmentation.")
    elif keras_pipeline:
        print("🏆 KerasOCR is the only working option!")
    elif paddle_ocr:
        print("🏆 PaddleOCR is the only working option!")
    else:
        print("❌ Neither engine worked properly!")


if __name__ == "__main__":
    main()

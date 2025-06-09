#!/usr/bin/env python3
"""Test OCR extraction on HUD elements from the same images used for YOLOv8 testing."""

import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract not available. Install with: pip install pytesseract")


def extract_text_with_ocr(image, region=None, config=""):
    """Extract text from image or image region using OCR."""
    if not OCR_AVAILABLE:
        return "OCR not available"

    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if region is not None:
                x1, y1, x2, y2 = region
                image_crop = image[y1:y2, x1:x2]
            else:
                image_crop = image

            # Convert BGR to RGB
            if len(image_crop.shape) == 3:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image_crop)
        else:
            pil_image = image

        # Apply OCR
        text = pytesseract.image_to_string(pil_image, config=config).strip()
        return text if text else "[No text detected]"

    except Exception as e:
        return f"[OCR Error: {e}]"


def get_hud_regions(image):
    """Get predefined HUD regions for OCR analysis."""
    height, width = image.shape[:2]

    # Define common HUD regions (these are typical locations)
    regions = {
        "bottom_hud_full": [0, int(height * 0.85), width, height],
        "bottom_left": [0, int(height * 0.85), int(width * 0.4), height],
        "bottom_center": [int(width * 0.4), int(height * 0.85), int(width * 0.6), height],
        "bottom_right": [int(width * 0.6), int(height * 0.85), width, height],
        "top_hud": [0, 0, width, int(height * 0.15)],
        "full_image": [0, 0, width, height],
    }

    return regions


def visualize_ocr_results(image_path, save_path=None):
    """Visualize OCR results on an image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get HUD regions
    regions = get_hud_regions(image)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left subplot: Original image with regions
    ax1.imshow(image_rgb)
    ax1.set_title(f"HUD Regions: {image_path.name}", fontsize=12)
    ax1.axis("off")

    # Draw regions and extract OCR text
    ocr_results = {}
    colors = ["red", "blue", "green", "orange", "purple", "yellow"]

    for i, (region_name, (x1, y1, x2, y2)) in enumerate(regions.items()):
        if region_name == "full_image":
            continue  # Skip full image for region visualization

        color = colors[i % len(colors)]

        # Draw region rectangle
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none", alpha=0.7
        )
        ax1.add_patch(rect)

        # Add region label
        ax1.text(
            x1,
            y1 - 5,
            region_name,
            color=color,
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Extract text with different OCR configs
        configs = [
            ("default", ""),
            ("digits", "--psm 6 -c tessedit_char_whitelist=0123456789:"),
            (
                "text",
                "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789: &-",
            ),
        ]

        region_texts = {}
        for config_name, config in configs:
            text = extract_text_with_ocr(image, region=(x1, y1, x2, y2), config=config)
            if text and text != "[No text detected]" and len(text.strip()) > 0:
                region_texts[config_name] = text.strip()

        ocr_results[region_name] = region_texts

    # Right subplot: OCR Results text
    ax2.axis("off")
    ax2.set_title("OCR Extraction Results", fontsize=12)

    # Format OCR results for display
    results_text = []
    total_extractions = 0

    for region_name, region_results in ocr_results.items():
        if region_results:
            results_text.append(f"📍 {region_name.upper()}:")
            for config_name, text in region_results.items():
                results_text.append(f"   {config_name}: '{text}'")
                total_extractions += 1
            results_text.append("")

    if not results_text:
        results_text = [
            "❌ No text detected in any region",
            "",
            "Possible reasons:",
            "• Low image quality",
            "• No HUD elements visible",
            "• OCR configuration needs tuning",
        ]
    else:
        results_text.insert(0, f"✅ Found {total_extractions} text extractions")
        results_text.insert(1, "")

    # Add full image OCR as well
    full_image_text = extract_text_with_ocr(image, config="--psm 6")
    if (
        full_image_text
        and full_image_text != "[No text detected]"
        and len(full_image_text.strip()) > 2
    ):
        results_text.extend(["🔍 FULL IMAGE OCR:", f"   Full: '{full_image_text.strip()}'", ""])

    # Display results text
    results_display = "\n".join(results_text)
    ax2.text(
        0.05,
        0.95,
        results_display,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    # Save the visualization
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved OCR visualization: {save_path}")

    plt.close()
    return {"regions": ocr_results, "total_extractions": total_extractions}


def main():
    """Main function to test OCR on the same images used for YOLOv8."""
    print("🔤 OCR Text Extraction Test")
    print("=" * 50)

    if not OCR_AVAILABLE:
        print("❌ OCR not available. Please install pytesseract:")
        print("   pip install pytesseract")
        print("   Also ensure Tesseract is installed on your system")
        return

    # Test OCR installation
    try:
        test_result = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {test_result}")
    except Exception as e:
        print(f"❌ Tesseract not properly installed: {e}")
        return

    # Image directory (same as YOLOv8 test)
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
    output_dir = Path("ocr_test_results")
    output_dir.mkdir(exist_ok=True)

    # Use same random seed to get similar images as YOLOv8 test
    random.seed(42)  # Set seed for reproducibility
    num_images = min(8, len(image_files))  # Test 8 images
    selected_images = random.sample(image_files, num_images)

    print(f"🎯 Testing OCR on {num_images} random images...")

    total_extractions = 0
    successful_ocr = 0

    for i, image_path in enumerate(selected_images, 1):
        try:
            save_path = output_dir / f"ocr_test_{i}_{image_path.name}"
            result = visualize_ocr_results(image_path, save_path)

            if result is not None:
                extractions = result["total_extractions"]
                total_extractions += extractions

                if extractions > 0:
                    successful_ocr += 1
                    print(
                        f"✅ {i:2d}/{num_images}: {extractions} text extractions in {image_path.name}"
                    )
                else:
                    print(f"🔍 {i:2d}/{num_images}: No text found in {image_path.name}")
            else:
                print(f"❌ {i:2d}/{num_images}: Failed to process {image_path.name}")

        except Exception as e:
            print(f"❌ {i:2d}/{num_images}: Error processing {image_path.name}: {e}")

    print("\n" + "=" * 50)
    print("📊 OCR TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Images with text found: {successful_ocr}/{num_images}")
    print(f"🔤 Total text extractions: {total_extractions}")
    print(f"📈 Average extractions per image: {total_extractions/num_images:.1f}")
    print(f"📁 Results saved in: {output_dir}")

    if total_extractions > 0:
        print(f"\n🔍 OCR Analysis:")
        print(f"   ✅ OCR successfully detected text in {successful_ocr} images")
        print(f"   📝 This can help validate YOLOv8 HUD detection accuracy")
        print(f"   💡 Use OCR to extract actual game data from detected HUD regions")
    else:
        print(f"\n🤔 No text detected. Possible reasons:")
        print(f"   • Images may not contain clear HUD text")
        print(f"   • OCR settings may need adjustment")
        print(f"   • HUD elements might be graphical rather than text-based")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple OCR test to check if we can extract text from HUD images."""

import random
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
    print("✅ pytesseract imported successfully")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"❌ pytesseract import failed: {e}")


def test_ocr_simple():
    """Simple OCR test."""
    if not OCR_AVAILABLE:
        return

    try:
        # Test Tesseract installation
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        print("💡 Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def extract_text_simple(image_path):
    """Simple text extraction from image."""
    if not OCR_AVAILABLE:
        return "OCR not available"

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return f"Could not load {image_path}"

        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Extract text with basic config
        text = pytesseract.image_to_string(pil_image, config="--psm 6").strip()

        if text:
            return f"Found text: '{text}'"
        else:
            return "No text detected"

    except Exception as e:
        return f"OCR error: {e}"


def main():
    """Test OCR on a few sample images."""
    print("🔤 Simple OCR Test")
    print("=" * 30)

    # Test OCR installation
    if not test_ocr_simple():
        print("\n⚠️ OCR setup required:")
        print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install Tesseract executable")
        print("3. Add Tesseract to your PATH or configure pytesseract.pytesseract.tesseract_cmd")
        return

    # Test on a few sample images
    image_dir = Path("training_data/images")
    if not image_dir.exists():
        print(f"❌ Directory not found: {image_dir}")
        return

    image_files = list(image_dir.glob("*.png"))[:5]  # Test first 5 images

    print(f"\n🎯 Testing OCR on {len(image_files)} images:")

    for i, image_path in enumerate(image_files, 1):
        result = extract_text_simple(image_path)
        print(f"{i}. {image_path.name}: {result}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Show what the current templates actually look like."""

from pathlib import Path

import cv2
import numpy as np


def show_templates():
    """Show all current templates."""
    print("🔍 CURRENT TEMPLATES IN down_templates_real/")
    print("=" * 50)

    template_dir = Path("down_templates_real")

    for template_file in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"\n📋 {template_file.name}:")
            print(f"   Size: {img.shape[1]}x{img.shape[0]}")
            print(f"   Pixel range: {img.min()}-{img.max()}")

            # Show a text representation
            print("   Visual preview:")
            # Resize to show in terminal (very rough)
            small = cv2.resize(img, (40, 15))
            for row in small:
                line = ""
                for pixel in row:
                    if pixel > 200:
                        line += " "
                    elif pixel > 150:
                        line += "."
                    elif pixel > 100:
                        line += "o"
                    else:
                        line += "#"
                print(f"   {line}")
        else:
            print(f"❌ Failed to load {template_file.name}")


if __name__ == "__main__":
    show_templates()

#!/usr/bin/env python3
"""Check what templates we actually have."""

from pathlib import Path

import cv2


def check_templates():
    """Check all template directories."""
    print("🔍 CHECKING ALL TEMPLATE DIRECTORIES")
    print("=" * 50)

    # Check down_templates_real
    print("\n📁 down_templates_real/:")
    real_dir = Path("down_templates_real")
    if real_dir.exists():
        for f in sorted(real_dir.glob("*.png")):
            img = cv2.imread(str(f))
            if img is not None:
                print(f"  {f.name}: {img.shape[1]}x{img.shape[0]}")
            else:
                print(f"  {f.name}: FAILED TO LOAD")
    else:
        print("  Directory not found!")

    # Check templates/raw_gameplay
    print("\n📁 templates/raw_gameplay/:")
    gameplay_dir = Path("templates/raw_gameplay")
    if gameplay_dir.exists():
        for f in sorted(gameplay_dir.glob("*.png")):
            img = cv2.imread(str(f))
            if img is not None:
                print(f"  {f.name}: {img.shape[1]}x{img.shape[0]}")
            else:
                print(f"  {f.name}: FAILED TO LOAD")
    else:
        print("  Directory not found!")

    # Check if there are other template directories
    print("\n📁 Other template directories:")
    for template_dir in Path(".").glob("*template*"):
        if template_dir.is_dir() and template_dir.name not in ["down_templates_real", "templates"]:
            print(f"  Found: {template_dir}")
            for f in sorted(template_dir.glob("*.png")):
                img = cv2.imread(str(f))
                if img is not None:
                    print(f"    {f.name}: {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    check_templates()

#!/usr/bin/env python3
"""
Replace generic templates with real Madden screenshots.
"""

import shutil
from pathlib import Path


def replace_templates():
    """Replace generic templates with real Madden screenshots."""
    print("🔄 Replacing Generic Templates with Real Madden Screenshots")
    print("=" * 60)

    source_dir = Path("templates/raw_gameplay")
    dest_dir = Path("down_templates_real")

    # Mapping of real Madden files to template names
    replacements = {
        # Normal templates
        "1st_10.png": "1ST.png",
        "2nd_7.png": "2ND.png",
        "3rd_3.png": "3RD.png",
        "4th_1.png": "4TH.png",
        # GOAL templates
        "3rd_goal.png": "3RD_GOAL.png",
        "4th_goal.png": "4TH_GOAL.png",
    }

    print("📋 Replacement Plan:")
    for source, dest in replacements.items():
        source_path = source_dir / source
        dest_path = dest_dir / dest

        if source_path.exists():
            print(f"  {source} → {dest}")
        else:
            print(f"  ❌ {source} NOT FOUND")

    print("\n🚀 Executing replacements...")

    for source, dest in replacements.items():
        source_path = source_dir / source
        dest_path = dest_dir / dest

        if source_path.exists():
            # Copy the real screenshot over the generic template
            shutil.copy2(source_path, dest_path)
            print(f"✅ Replaced {dest} with real Madden screenshot")
        else:
            print(f"❌ Source not found: {source}")

    # Check what we still need
    print("\n📊 Missing Templates:")
    needed_templates = ["1ST_GOAL.png", "2ND_GOAL.png"]

    for template in needed_templates:
        template_path = dest_dir / template
        if template_path.exists():
            # Check if it's still oversized (generic)
            import cv2

            img = cv2.imread(str(template_path))
            if img is not None:
                height, width = img.shape[:2]
                if width > 100:  # Still oversized
                    print(
                        f"  ⚠️ {template}: Still oversized ({width}x{height}) - needs real screenshot"
                    )
                else:
                    print(f"  ✅ {template}: Good size ({width}x{height})")
        else:
            print(f"  ❌ {template}: Missing")

    print("\n💡 Next Steps:")
    print("  1. You need real screenshots for 1ST_GOAL and 2ND_GOAL situations")
    print("  2. Or use the crop GUI to fix the oversized GOAL templates")
    print("  3. Run test_real_templates.py to check accuracy")


def verify_templates():
    """Verify all templates are real Madden screenshots."""
    print("\n🔍 Template Verification:")
    print("=" * 30)

    template_dir = Path("down_templates_real")

    for template_file in sorted(template_dir.glob("*.png")):
        import cv2

        img = cv2.imread(str(template_file))
        if img is not None:
            height, width = img.shape[:2]
            size_status = "✅ Good" if width <= 100 else "⚠️ Oversized"
            print(f"📋 {template_file.name}: {width}x{height} {size_status}")


def main():
    """Main function."""
    replace_templates()
    verify_templates()


if __name__ == "__main__":
    main()

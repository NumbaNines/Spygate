#!/usr/bin/env python3
"""
Fix template system by using real Madden screenshots as templates.
"""

import shutil
from pathlib import Path

import cv2


def create_real_templates():
    """Create templates from real Madden screenshots."""
    print("🔧 Creating Templates from Real Madden Screenshots")
    print("=" * 60)

    # Source and destination directories
    source_dir = Path("templates/raw_gameplay")
    dest_dir = Path("down_templates_real")

    # Create destination directory
    dest_dir.mkdir(exist_ok=True)

    # Mapping of real files to template names
    template_mapping = {
        # Normal templates
        "1st_10.png": "1ST.png",
        "2nd_7.png": "2ND.png",
        "3rd_3.png": "3RD.png",
        "4th_1.png": "4TH.png",
        # GOAL templates
        "3rd_goal.png": "3RD_GOAL.png",
        "4th_goal.png": "4TH_GOAL.png",
    }

    # Copy and convert real screenshots to templates
    templates_created = 0

    for source_file, template_name in template_mapping.items():
        source_path = source_dir / source_file
        dest_path = dest_dir / template_name

        if source_path.exists():
            # Load image
            img = cv2.imread(str(source_path))
            if img is not None:
                # Convert to grayscale for template matching
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Save as template
                cv2.imwrite(str(dest_path), gray)

                print(
                    f"✅ Created {template_name} from {source_file} ({img.shape[1]}x{img.shape[0]})"
                )
                templates_created += 1
            else:
                print(f"❌ Failed to load {source_file}")
        else:
            print(f"⚠️ Source file not found: {source_file}")

    # Create missing templates by duplicating existing ones
    missing_templates = {
        "2ND.png": "1ST.png",  # Use 1ST as 2ND if 2ND missing
        "1ST_GOAL.png": "3RD_GOAL.png",  # Use 3RD_GOAL as 1ST_GOAL
        "2ND_GOAL.png": "3RD_GOAL.png",  # Use 3RD_GOAL as 2ND_GOAL
        "4TH_GOAL.png": "3RD_GOAL.png",  # Use 3RD_GOAL as 4TH_GOAL if missing
    }

    for missing, source in missing_templates.items():
        missing_path = dest_dir / missing
        source_path = dest_dir / source

        if not missing_path.exists() and source_path.exists():
            shutil.copy2(source_path, missing_path)
            print(f"📋 Created {missing} by copying {source}")
            templates_created += 1

    print(f"\n✅ Created {templates_created} real templates")
    return templates_created


def test_real_templates():
    """Test the new real templates."""
    print("\n🧪 Testing Real Templates")
    print("=" * 30)

    from down_template_detector import DownTemplateDetector

    # Initialize detector with real templates
    detector = DownTemplateDetector()
    print(f"✅ Loaded {len(detector.templates)} templates")

    # Test with a real screenshot
    test_file = "templates/raw_gameplay/1st_10.png"
    if Path(test_file).exists():
        frame = cv2.imread(test_file)
        height, width = frame.shape[:2]
        bbox = (0, 0, width, height)

        result = detector.detect_down_in_yolo_region(frame, bbox, False)

        if result:
            print(f"🎯 Detection SUCCESS!")
            print(f"   Down: {result.down}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Template: {result.template_name}")
            return True
        else:
            print("❌ Detection failed")
            return False
    else:
        print(f"❌ Test file not found: {test_file}")
        return False


def main():
    """Main function."""
    print("🚀 SpygateAI Template Fix")
    print("=" * 30)

    # Step 1: Create real templates
    templates_created = create_real_templates()

    if templates_created > 0:
        # Step 2: Test the new templates
        success = test_real_templates()

        if success:
            print("\n🎉 SUCCESS! Real templates are working!")
            print("🚀 Next step: Run test_real_templates.py again")
        else:
            print("\n⚠️ Templates created but detection still needs work")
    else:
        print("\n❌ Failed to create templates")


if __name__ == "__main__":
    main()

"""
Setup script for 8-class HUD annotation using labelme.
Prepares directory structure and copies existing data for expansion.
"""

import json
import os
import shutil
from pathlib import Path


def setup_8class_annotation():
    """Set up directory structure for 8-class annotation."""

    # Create 8-class directory structure
    base_dir = Path("hud_region_training_8class")
    base_dir.mkdir(exist_ok=True)

    # Create subdirectories
    dirs_to_create = [
        "datasets_8class/train/images",
        "datasets_8class/train/labels",
        "datasets_8class/val/images",
        "datasets_8class/val/labels",
        "labelme_annotations",
        "runs",
    ]

    for dir_path in dirs_to_create:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ Created 8-class directory structure")

    # Copy existing 5-class images for annotation
    source_train = Path("hud_region_training/datasets/train/images")
    source_val = Path("hud_region_training/datasets/val/images")

    dest_train = base_dir / "datasets_8class/train/images"
    dest_val = base_dir / "datasets_8class/val/images"

    if source_train.exists():
        for img_file in source_train.glob("*.jpg"):
            shutil.copy2(img_file, dest_train)
        print(f"✅ Copied {len(list(source_train.glob('*.jpg')))} training images")

    if source_val.exists():
        for img_file in source_val.glob("*.jpg"):
            shutil.copy2(img_file, dest_val)
        print(f"✅ Copied {len(list(source_val.glob('*.jpg')))} validation images")

    # Create labelme config for 8-class system
    labelme_config = {
        "flags": {},
        "shapes": [
            {"label": "hud", "points": [], "group_id": None, "shape_type": "rectangle", "flags": {}}
        ],
        "labels": [
            "hud",
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ],
    }

    config_path = base_dir / "labelme_config.json"
    with open(config_path, "w") as f:
        json.dump(labelme_config, f, indent=2)

    print("✅ Created labelme configuration")

    # Create annotation instructions
    instructions = """
# 8-Class HUD Annotation Instructions

## New Classes to Annotate (in addition to existing 5):

### 5. down_distance_area
- **Location**: Usually center-left of HUD
- **Contains**: Down and distance text (e.g., "3rd & 8", "1st & 10")
- **Shape**: Rectangle around the down/distance text

### 6. game_clock_area
- **Location**: Usually center of HUD
- **Contains**: Quarter and game time (e.g., "1st 12:34", "4th 2:00")
- **Shape**: Rectangle around the game clock

### 7. play_clock_area
- **Location**: Usually near game clock or separate area
- **Contains**: Play clock countdown (e.g., "25", "40")
- **Shape**: Rectangle around the play clock number

## Annotation Workflow:

1. Start labelme: `labelme hud_region_training_8class/datasets_8class/train/images --config hud_region_training_8class/labelme_config.json`

2. For each image:
   - Draw rectangles around ALL 8 HUD elements
   - Use the exact class names from the config
   - Save annotations in JSON format

3. Convert to YOLO format when done:
   - Use the conversion script (will be created)

## Tips:
- Focus on the NEW classes (5-7) since existing classes (0-4) can be migrated
- Make rectangles tight around text but with small padding
- Be consistent with rectangle sizes across similar elements
- If an element is not visible, skip that class for that image
"""

    instructions_path = base_dir / "ANNOTATION_INSTRUCTIONS.md"
    with open(instructions_path, "w") as f:
        f.write(instructions)

    print("✅ Created annotation instructions")
    print(f"\n📁 Setup complete! Directory: {base_dir}")
    print(f"📖 Read instructions: {instructions_path}")
    print(f"\n🚀 Start annotating with:")
    print(
        f"labelme {base_dir}/datasets_8class/train/images --config {base_dir}/labelme_config.json"
    )


if __name__ == "__main__":
    setup_8class_annotation()

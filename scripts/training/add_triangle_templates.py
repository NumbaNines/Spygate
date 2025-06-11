#!/usr/bin/env python3
"""Add triangle template positions to existing annotations based on HUD locations."""

import json
import argparse
from pathlib import Path

def add_triangle_templates(json_file, hud_confidence=0.8):
    """Add triangle templates based on HUD position."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_file}: {e}")
        return False
    
    # Find HUD annotation
    hud_box = None
    for shape in data.get('shapes', []):
        if shape['label'] == 'hud' and shape['shape_type'] == 'rectangle':
            hud_box = shape['points']
            break
    
    if not hud_box:
        print(f"⚠️  No HUD found in {json_file.name}")
        return False
    
    # Calculate HUD dimensions
    x1, y1 = hud_box[0]
    x2, y2 = hud_box[1]
    hud_width = x2 - x1
    hud_height = y2 - y1
    
    # Check if triangles already exist
    existing_triangles = set()
    for shape in data.get('shapes', []):
        if shape['label'] in ['possession_indicator', 'territory_indicator']:
            existing_triangles.add(shape['label'])
    
    templates_added = 0
    
    # Add possession_indicator template (LEFT side of HUD)
    if 'possession_indicator' not in existing_triangles:
        possession_x = x1 + (hud_width * 0.25)  # 25% from left
        possession_y = y1 + (hud_height * 0.5)   # Middle of HUD
        triangle_size = min(hud_width * 0.05, hud_height * 0.4)  # Small triangle
        
        possession_template = {
            "label": "possession_indicator",
            "points": [
                [possession_x - triangle_size, possession_y - triangle_size],
                [possession_x + triangle_size, possession_y + triangle_size]
            ],
            "group_id": None,
            "description": "TEMPLATE - Adjust position & size",
            "shape_type": "rectangle",
            "flags": {}
        }
        data['shapes'].append(possession_template)
        templates_added += 1
    
    # Add territory_indicator template (FAR RIGHT side of HUD)
    if 'territory_indicator' not in existing_triangles:
        territory_x = x2 - (hud_width * 0.1)     # 10% from right edge
        territory_y = y1 + (hud_height * 0.5)     # Middle of HUD
        triangle_size = min(hud_width * 0.05, hud_height * 0.4)  # Small triangle
        
        territory_template = {
            "label": "territory_indicator", 
            "points": [
                [territory_x - triangle_size, territory_y - triangle_size],
                [territory_x + triangle_size, territory_y + triangle_size]
            ],
            "group_id": None,
            "description": "TEMPLATE - Adjust position & size",
            "shape_type": "rectangle",
            "flags": {}
        }
        data['shapes'].append(territory_template)
        templates_added += 1
    
    # Save updated file if templates were added
    if templates_added > 0:
        try:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ {json_file.name}: Added {templates_added} triangle templates")
            return True
        except Exception as e:
            print(f"❌ Error saving {json_file}: {e}")
            return False
    else:
        print(f"ℹ️  {json_file.name}: Triangles already exist, skipping")
        return False

def main():
    parser = argparse.ArgumentParser(description="Add triangle templates to existing annotations")
    parser.add_argument("--labels", default="training_data/labels", help="Directory with label files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels)
    json_files = list(labels_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {labels_dir}")
        return
    
    print(f"🔍 Processing {len(json_files)} annotation files...")
    if args.dry_run:
        print("🔍 DRY RUN - No files will be modified")
    
    processed = 0
    added = 0
    
    for json_file in json_files:
        if args.dry_run:
            print(f"Would process: {json_file.name}")
            processed += 1
        else:
            if add_triangle_templates(json_file):
                added += 1
            processed += 1
    
    print(f"\n📊 Summary:")
    print(f"   📁 Files processed: {processed}")
    if not args.dry_run:
        print(f"   ✅ Templates added to: {added} files")
        print(f"\n🎯 Next steps:")
        print(f"   1. Open LabelMe: labelme {args.labels}")
        print(f"   2. Look for 'TEMPLATE' descriptions")
        print(f"   3. Adjust triangle positions and sizes")
        print(f"   4. Remove 'TEMPLATE' from description when done")

if __name__ == "__main__":
    main() 
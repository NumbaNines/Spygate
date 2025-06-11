#!/usr/bin/env python3
"""Convert possession_indicator and territory_indicator to simplified 'triangle' class."""

import json
from pathlib import Path

def simplify_triangles(json_file):
    """Convert triangle classes to simplified 'triangle' class."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_file}: {e}")
        return False
    
    changes_made = False
    
    # Convert triangle classes
    for shape in data.get('shapes', []):
        if shape['label'] in ['possession_indicator', 'territory_indicator']:
            old_label = shape['label']
            shape['label'] = 'triangle'
            # Update description to remove TEMPLATE text and add position info
            if 'possession' in old_label:
                shape['description'] = 'Left side - shows ball possession'
            else:
                shape['description'] = 'Right side - shows field territory'
            changes_made = True
            print(f"  ✅ {old_label} → triangle")
    
    # Save if changes were made
    if changes_made:
        try:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving {json_file}: {e}")
            return False
    
    return False

def main():
    labels_dir = Path("training_data/labels")
    json_files = list(labels_dir.glob("*.json"))
    
    print(f"🔄 Simplifying triangle classes in {len(json_files)} files...")
    
    updated = 0
    for json_file in json_files:
        print(f"📁 {json_file.name}")
        if simplify_triangles(json_file):
            updated += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Files updated: {updated}")
    print(f"   🎯 Now you have a single 'triangle' class!")
    print(f"   📍 Position context is preserved in descriptions")

if __name__ == "__main__":
    main() 
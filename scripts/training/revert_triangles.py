#!/usr/bin/env python3
"""Revert triangle annotations back to possession_indicator and territory_indicator."""

import json
from pathlib import Path

def revert_triangles(json_file):
    """Revert triangle annotations back to original classes."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_file}: {e}")
        return False
    
    changes_made = False
    
    # Revert triangle classes based on descriptions
    for shape in data.get('shapes', []):
        if shape['label'] == 'triangle':
            description = shape.get('description', '')
            if 'Left side' in description or 'possession' in description:
                shape['label'] = 'possession_indicator'
                shape['description'] = 'TEMPLATE - Left side triangle (possession)'
                changes_made = True
                print(f"  ✅ triangle → possession_indicator")
            elif 'Right side' in description or 'territory' in description:
                shape['label'] = 'territory_indicator'
                shape['description'] = 'TEMPLATE - Right side triangle (territory)'
                changes_made = True
                print(f"  ✅ triangle → territory_indicator")
    
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
    
    print(f"🔄 Reverting triangle classes in {len(json_files)} files...")
    
    updated = 0
    for json_file in json_files:
        print(f"📁 {json_file.name}")
        if revert_triangles(json_file):
            updated += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Files updated: {updated}")
    print(f"   🎯 Restored original triangle classes!")
    print(f"   📍 possession_indicator (left side)")
    print(f"   📍 territory_indicator (right side)")

if __name__ == "__main__":
    main() 
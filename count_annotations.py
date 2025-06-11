#!/usr/bin/env python3
"""Count annotations for each class in the training data."""

import json
import os
from pathlib import Path

def count_class_annotations():
    """Count how many annotations exist for each class."""
    
    # Class mapping - updated to include triangle classes
    classes = ['hud', 'qb_position', 'left_hash_mark', 'right_hash_mark', 'preplay', 'playcall', 'possession_indicator', 'territory_indicator']
    class_counts = {cls: 0 for cls in classes}
    
    # Count annotations in JSON label files
    labels_dir = Path('training_data/labels')
    
    if not labels_dir.exists():
        print("❌ Labels directory not found!")
        return
    
    total_files = 0
    for label_file in labels_dir.glob('*.json'):
        total_files += 1
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape.get('label', '')
                    if label in class_counts:
                        class_counts[label] += 1
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    print('📊 CURRENT ANNOTATION COUNT:')
    print('=' * 50)
    
    for cls, count in class_counts.items():
        if count >= 100:
            status = '🌟 EXCELLENT'
        elif count >= 50:
            status = '✅ GOOD'
        elif count >= 10:
            status = '⚠️  FAIR'
        else:
            status = '❌ POOR'
        
        print(f'{status:<15} {cls:<20}: {count:>3} examples')
    
    total_annotations = sum(class_counts.values())
    print('=' * 50)
    print(f'📝 Total label files: {total_files}')
    print(f'📊 Total annotations: {total_annotations}')
    
    # Recommendations
    print('\n🎯 RECOMMENDATIONS:')
    print('=' * 50)
    
    needs_work = []
    for cls, count in class_counts.items():
        if count < 50:
            needed = 50 - count
            needs_work.append(f'{cls}: need {needed} more')
    
    if needs_work:
        print('Classes that need more annotations:')
        for rec in needs_work:
            print(f'   📝 {rec}')
    else:
        print('🎉 All classes have sufficient annotations!')
    
    return class_counts

if __name__ == "__main__":
    count_class_annotations() 
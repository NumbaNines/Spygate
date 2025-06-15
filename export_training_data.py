#!/usr/bin/env python3
"""
Simple Training Data Exporter
Export validated samples for OCR model training
"""

import json
import os
from datetime import datetime

from ultimate_madden_ocr_system import MaddenOCRDatabase


def main():
    print("📦 Training Data Exporter")
    print("=" * 40)

    # Initialize database
    db = MaddenOCRDatabase()

    # Get statistics
    stats = db.get_statistics()
    total_samples = stats["total_samples"]
    validated_samples = stats["validated_samples"]

    print(f"Total samples: {total_samples:,}")
    print(f"Validated samples: {validated_samples:,}")
    print(f"Progress: {(validated_samples/total_samples)*100:.1f}%")

    if validated_samples == 0:
        print("❌ No validated samples to export!")
        return

    # Show breakdown by class
    validated_by_class = stats.get("validated_by_class", {})
    print(f"\n📋 Validated samples by class:")
    for class_name, count in validated_by_class.items():
        print(f"  {class_name}: {count:,}")

    # Get training data
    print(f"\n📊 Exporting {validated_samples:,} validated samples...")
    training_data = db.get_training_data()

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"madden_ocr_training_data_{timestamp}.json"

    # Export to JSON
    with open(filename, "w") as f:
        json.dump(training_data, f, indent=2, default=str)

    # Get file size
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)

    print(f"✅ Export complete!")
    print(f"📁 File: {filename}")
    print(f"📊 Size: {file_size_mb:.1f} MB")
    print(f"🎯 Ready for OCR model training!")

    # Show summary
    print(f"\n📈 Training Data Summary:")
    print(f"  Total samples: {len(training_data):,}")
    print(f"  Classes: {len(validated_by_class)}")
    print(f"  Quality: High (manually validated)")
    print(f"  Ready for: Keras OCR model training")


if __name__ == "__main__":
    main()

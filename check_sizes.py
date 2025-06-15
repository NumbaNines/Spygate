#!/usr/bin/env python3
"""Check database and model sizes"""

import os

from ultimate_madden_ocr_system import MaddenOCRDatabase


def main():
    # Check database size
    db = MaddenOCRDatabase()
    db_size_mb = os.path.getsize(db.db_path) / (1024 * 1024)

    stats = db.get_statistics()
    total_samples = stats["total_samples"]
    size_per_sample_kb = os.path.getsize(db.db_path) / total_samples / 1024

    print("📊 Current Storage Analysis")
    print("=" * 40)
    print(f"Database file size: {db_size_mb:.1f} MB")
    print(f"Current samples: {total_samples:,}")
    print(f"Size per sample: {size_per_sample_kb:.1f} KB")

    # Estimate for different sample counts
    print(f"\n🔮 Size Estimates:")
    print("=" * 40)

    sample_counts = [5000, 10000, 20000, 50000]
    for count in sample_counts:
        estimated_mb = (count * size_per_sample_kb) / 1024
        print(f"{count:,} samples: {estimated_mb:.1f} MB")

        if estimated_mb < 10:
            status = "✅ Tiny!"
        elif estimated_mb < 50:
            status = "✅ Very manageable"
        elif estimated_mb < 200:
            status = "✅ Reasonable"
        else:
            status = "⚠️  Large"

        print(f"  └─ {status}")


if __name__ == "__main__":
    main()

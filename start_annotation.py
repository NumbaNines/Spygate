#!/usr/bin/env python3

import os
import sqlite3
import sys


def main():
    print("🎯 Madden OCR Annotation - Focus on Down/Distance")
    print("=" * 50)

    # Check database
    if not os.path.exists("madden_ocr_training.db"):
        print("❌ Database not found!")
        return

    conn = sqlite3.connect("madden_ocr_training.db")
    cursor = conn.cursor()

    # Get sample counts
    cursor.execute("SELECT COUNT(*) FROM ocr_samples")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM ocr_samples WHERE ground_truth_text IS NOT NULL")
    validated = cursor.fetchone()[0]

    cursor.execute("SELECT class_name, COUNT(*) FROM ocr_samples GROUP BY class_name")
    by_class = cursor.fetchall()

    print(f"📊 Current Status:")
    print(f"   Total samples: {total}")
    print(f"   Validated: {validated}")
    print(f"   Remaining: {total - validated}")
    print()

    print("📋 Samples by class:")
    for class_name, count in by_class:
        cursor.execute(
            "SELECT COUNT(*) FROM ocr_samples WHERE class_name = ? AND ground_truth_text IS NOT NULL",
            (class_name,),
        )
        validated_count = cursor.fetchone()[0]
        print(f"   {class_name}: {count} total, {validated_count} validated")

    conn.close()

    print()
    print("🎯 Priority: down_distance_area samples (fixes false 3rd down detections)")
    print("💡 Target: ~200-300 validated samples for 99%+ accuracy")
    print()
    print("🚀 Ready to launch annotation GUI!")
    print("   - Use quick buttons for common patterns")
    print("   - Press Enter to save and continue")
    print("   - Arrow keys to navigate")
    print()

    input("Press Enter to launch annotation GUI...")

    # Launch the annotation system
    from ultimate_madden_ocr_system import MaddenOCRDatabase, UltimateMaddenOCRSystem

    database = MaddenOCRDatabase()

    # Import and launch GUI
    from ultimate_madden_ocr_system import MaddenOCRAnnotationGUI

    gui = MaddenOCRAnnotationGUI(database)
    gui.run()


if __name__ == "__main__":
    main()

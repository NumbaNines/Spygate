#!/usr/bin/env python3
"""
Batch Annotation Helper for 20,000 Sample Target
Efficient annotation workflow with progress tracking
"""

from ultimate_madden_ocr_system import MaddenOCRAnnotationGUI, MaddenOCRDatabase


class BatchAnnotationHelper:
    def __init__(self):
        self.db = MaddenOCRDatabase()

    def show_annotation_progress(self):
        """Show detailed annotation progress"""
        stats = self.db.get_statistics()
        total = stats["total_samples"]
        validated = stats["validated_samples"]
        remaining = total - validated

        print("📊 Annotation Progress Report")
        print("=" * 40)
        print(f"Total samples: {total:,}")
        print(f"Validated: {validated:,}")
        print(f"Remaining: {remaining:,}")
        print(f"Progress: {(validated/total)*100:.1f}%")

        # Show progress by class
        by_class = stats.get("by_class", {})
        validated_by_class = stats.get("validated_by_class", {})

        print(f"\n📋 Progress by Class:")
        for class_name in by_class:
            total_class = by_class[class_name]
            validated_class = validated_by_class.get(class_name, 0)
            remaining_class = total_class - validated_class
            progress_class = (validated_class / total_class) * 100 if total_class > 0 else 0

            print(f"  {class_name}:")
            print(f"    ✅ Validated: {validated_class:,}/{total_class:,} ({progress_class:.1f}%)")
            print(f"    ⏳ Remaining: {remaining_class:,}")

        # Estimate time remaining
        if validated > 0 and remaining > 0:
            # Assume 3-5 seconds per sample (conservative estimate)
            estimated_hours = (remaining * 4) / 3600  # 4 seconds average
            print(f"\n⏱️  Estimated time remaining: {estimated_hours:.1f} hours")
            print(f"   (at ~4 seconds per sample)")

        return remaining

    def launch_annotation_session(self):
        """Launch optimized annotation session"""
        print("🎯 Batch Annotation Helper - 20K Target")
        print("=" * 50)

        # Show current progress
        remaining = self.show_annotation_progress()

        if remaining == 0:
            print("\n🎉 All samples annotated! Ready for training!")
            return

        print(f"\n🚀 Starting annotation session...")
        print(f"💡 Tips for efficient annotation:")
        print(f"   • Use quick buttons for common patterns")
        print(f"   • Press Enter to save and continue")
        print(f"   • Use Browse All Samples to edit mistakes")
        print(f"   • Take breaks every 500-1000 samples")

        input("\nPress Enter to launch annotation GUI...")

        # Launch GUI
        gui = MaddenOCRAnnotationGUI(self.db)
        gui.run()

        # Show final progress
        print("\n📊 Session Complete!")
        self.show_annotation_progress()

    def export_when_ready(self):
        """Export training data when annotation is complete"""
        stats = self.db.get_statistics()
        total = stats["total_samples"]
        validated = stats["validated_samples"]

        if validated >= 15000:  # 75% of 20k target
            print(f"\n🎯 Ready for high-quality training!")
            print(f"   Validated samples: {validated:,}")

            export_choice = input("Export training data now? (y/n): ").lower()
            if export_choice == "y":
                # Launch GUI for export
                gui = MaddenOCRAnnotationGUI(self.db)
                gui.export_training_data()
        else:
            print(f"\n⏳ Continue annotating for best results")
            print(f"   Target: 15,000+ samples for training")
            print(f"   Current: {validated:,} samples")


def main():
    helper = BatchAnnotationHelper()

    while True:
        print("\n🎯 Batch Annotation Menu")
        print("=" * 30)
        print("1. Show progress")
        print("2. Start annotation session")
        print("3. Export training data")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            helper.show_annotation_progress()
        elif choice == "2":
            helper.launch_annotation_session()
        elif choice == "3":
            helper.export_when_ready()
        elif choice == "4":
            print("👋 Happy annotating!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()

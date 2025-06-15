from ultimate_madden_ocr_system import MaddenOCRDatabase

db = MaddenOCRDatabase()
stats = db.get_statistics()

print("🎯 4-Class OCR Training Data Status")
print("=" * 40)
print(f"Total samples: {stats['total_samples']:,}")
print(f"Validated samples: {stats['validated_samples']:,}")
print(f"Progress: {(stats['validated_samples']/stats['total_samples'])*100:.1f}%")

print(f"\n📊 Breakdown by class:")
validated_by_class = stats.get("validated_by_class", {})
total_by_class = stats.get("total_by_class", {})

target_classes = [
    "down_distance_area",
    "game_clock_area",
    "play_clock_area",
    "territory_triangle_area",
]

for class_name in target_classes:
    validated = validated_by_class.get(class_name, 0)
    total = total_by_class.get(class_name, 0)
    if total > 0:
        progress = (validated / total) * 100
        print(f"  {class_name}: {validated:,}/{total:,} ({progress:.1f}%)")
    else:
        print(f"  {class_name}: 0/0 (0.0%)")

print(f"\n🎯 Ready for training?")
territory_validated = validated_by_class.get("territory_triangle_area", 0)
if territory_validated > 100:
    print("✅ YES! Territory samples available - ready for 4-class training")
else:
    print(f"⏳ Not yet - need more territory samples (have {territory_validated}, need 100+)")
    print("   Extraction still running in background...")

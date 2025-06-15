import sqlite3

conn = sqlite3.connect("madden_ocr_training.db")
cursor = conn.cursor()

print("📊 Current class distribution:")
cursor.execute("SELECT class_name, COUNT(*) FROM ocr_samples GROUP BY class_name")
results = cursor.fetchall()

for class_name, count in results:
    print(f"  {class_name}: {count:,}")

print(f"\n📊 Validated samples by class:")
cursor.execute(
    "SELECT class_name, COUNT(*) FROM ocr_samples WHERE ground_truth_text IS NOT NULL AND ground_truth_text != '' GROUP BY class_name"
)
validated_results = cursor.fetchall()

for class_name, count in validated_results:
    print(f"  {class_name}: {count:,} validated")

conn.close()

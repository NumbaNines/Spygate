import sqlite3

conn = sqlite3.connect("madden_ocr_training.db")
cursor = conn.cursor()

# Get total samples
cursor.execute("SELECT COUNT(*) FROM ocr_samples")
total = cursor.fetchone()[0]

# Get unique images processed
cursor.execute("SELECT COUNT(DISTINCT image_path) FROM ocr_samples")
unique_images = cursor.fetchone()[0]

# Get samples by class
cursor.execute("SELECT class_name, COUNT(*) FROM ocr_samples GROUP BY class_name")
by_class = cursor.fetchall()

print(f"Total samples: {total}")
print(f"Unique images processed: {unique_images}")
print(f"Samples per image: {total/unique_images:.1f}")
print("\nSamples per class:")
for class_name, count in by_class:
    print(f"  {class_name}: {count}")

conn.close()

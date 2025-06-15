import sqlite3

conn = sqlite3.connect("madden_ocr_training.db")
cursor = conn.cursor()

# Get total samples
cursor.execute("SELECT COUNT(*) FROM ocr_samples")
total = cursor.fetchone()[0]

# Get validated samples
cursor.execute("SELECT COUNT(*) FROM ocr_samples WHERE is_validated = TRUE")
validated = cursor.fetchone()[0]

# Get samples by class
cursor.execute(
    "SELECT class_name, COUNT(*) FROM ocr_samples WHERE is_validated = TRUE GROUP BY class_name"
)
validated_by_class = cursor.fetchall()

print(f"Total samples: {total}")
print(f"Validated samples: {validated}")
print("Validated by class:")
for class_name, count in validated_by_class:
    print(f"  {class_name}: {count}")

conn.close()

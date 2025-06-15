from ultimate_madden_ocr_system import MaddenOCRDatabase

db = MaddenOCRDatabase()
samples = db.get_all_samples(1)

if samples:
    print("Sample keys:", list(samples[0].keys()))
    print("Sample data:", samples[0])
else:
    print("No samples found")

stats = db.get_statistics()
print("Stats:", stats)

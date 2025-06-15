import json
import os

import cv2

# Load core dataset
with open("madden_ocr_training_data_CORE.json", "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Check first sample
sample = data[0]
print(f"Sample text: '{sample['ground_truth_text']}'")
print(f"Sample path: {sample['image_path']}")

# Check if image exists and loads
img_path = sample["image_path"]
if os.path.exists(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"Image loaded: {img.shape}")
        print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

        if img.mean() < 50:
            print("WARNING: Very dark image")
    else:
        print("ERROR: Failed to load image")
else:
    print("ERROR: Image file does not exist")

# Check pattern distribution
from collections import Counter

text_counter = Counter([s["ground_truth_text"] for s in data])
print(f"Top patterns:")
for text, count in text_counter.most_common(5):
    print(f"  '{text}': {count}")

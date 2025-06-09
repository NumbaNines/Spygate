import os
from pathlib import Path


def check_dataset():
    dataset_dir = Path("test_dataset")
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nChecking {split} split:")
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        # Get all image files
        image_files = [f.name for f in images_dir.glob("*.png")]
        label_files = [f.name.replace(".txt", ".png") for f in labels_dir.glob("*.txt")]

        # Check for images without labels
        for img in image_files:
            txt = img.replace(".png", ".txt")
            if not (labels_dir / txt).exists():
                print(f"Missing annotation for {split}/{img}")

        # Check for labels without images
        for txt in labels_dir.glob("*.txt"):
            img = txt.name.replace(".txt", ".png")
            if not (images_dir / img).exists():
                print(f"Missing image for {split}/{txt.name}")

        print(f"{split} set: {len(image_files)} images, {len(label_files)} labels")


if __name__ == "__main__":
    check_dataset()

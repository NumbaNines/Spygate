import glob
import os


def fix_label_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    fixed = False
    fixed_lines = []
    for line in lines:
        if line.strip():
            parts = line.split()
            if parts[0] == "6":
                parts[0] = "4"  # Replace class 6 with class 4 (no huddle)
                line = " ".join(parts) + "\n"
                fixed = True
        fixed_lines.append(line)

    if fixed:
        with open(file_path, "w") as f:
            f.writelines(fixed_lines)
        print(f"Fixed {file_path}")


# Fix labels in all dataset splits
splits = ["train", "val", "test"]
for split in splits:
    label_dir = os.path.join("test_dataset", "labels", split)
    if os.path.exists(label_dir):
        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            fix_label_file(label_file)

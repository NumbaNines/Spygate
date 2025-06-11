"""
Setup workflow for triangle annotation using labelme.

This script helps set up directories and provides instructions for using labelme
to annotate triangle detection training data.

Usage:
    python setup_triangle_annotation_workflow.py
"""

import os
import shutil
from pathlib import Path

def setup_directories():
    """Create necessary directories for the annotation workflow"""
    directories = [
        "images_to_annotate",
        "labelme_annotations", 
        "yolo_triangle_dataset"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}")

def copy_training_images():
    """Copy some training images to the annotation directory"""
    source_dirs = ["training_data", "clean_madden_screenshots"]
    dest_dir = Path("images_to_annotate")
    
    copied_count = 0
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if source_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_file in source_path.glob(ext):
                    dest_file = dest_dir / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)
                        copied_count += 1
    
    print(f"Copied {copied_count} images to {dest_dir}")
    return copied_count

def create_class_config():
    """Create labelme class configuration"""
    class_config = {
        "possession_indicator": "Triangle on LEFT side between team abbreviations (shows ball possession)",
        "territory_indicator": "Triangle on FAR RIGHT side (UP = opponent territory, DOWN = own territory)"
    }
    
    config_file = Path("labelme_config.txt")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("TRIANGLE DETECTION - LABELME ANNOTATION GUIDE\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classes to annotate:\n\n")
        
        for class_name, description in class_config.items():
            f.write(f"Class: {class_name}\n")
            f.write(f"Description: {description}\n\n")
        
        f.write("ANNOTATION INSTRUCTIONS:\n")
        f.write("1. Use 'Create Rectangle' tool for triangles\n")
        f.write("2. Make tight bounding boxes around triangles\n")
        f.write("3. Label exactly as: possession_indicator or territory_indicator\n")
        f.write("4. Save each image as JSON format\n")
        f.write("5. Save JSON files to labelme_annotations/ directory\n\n")
        
        f.write("TRIANGLE LOCATIONS:\n")
        f.write("- possession_indicator: LEFT side, between team names/scores\n")
        f.write("- territory_indicator: FAR RIGHT side, next to yard line\n\n")
        
        f.write("TIPS:\n")
        f.write("- Look for small triangular shapes in HUD\n")
        f.write("- Possession triangle points to team with ball\n")
        f.write("- Territory triangle: UP = opponent territory, DOWN = own territory\n")
        f.write("- Skip images without visible triangles\n")
    
    print(f"Created annotation guide: {config_file}")

def print_workflow_instructions():
    """Print step-by-step workflow instructions"""
    print("\n" + "=" * 60)
    print("TRIANGLE ANNOTATION WORKFLOW")
    print("=" * 60)
    print()
    print("STEP 1: Prepare Images")
    print("- Add Madden HUD screenshots to images_to_annotate/")
    print("- Or use: python capture_clean_madden_screenshots.py")
    print()
    print("STEP 2: Start Labelme")
    print("Command: labelme images_to_annotate/ --output labelme_annotations/")
    print()
    print("STEP 3: Annotate Triangles")
    print("- Use Rectangle tool to box triangles")
    print("- Label as: possession_indicator or territory_indicator")
    print("- Save as JSON files")
    print()
    print("STEP 4: Convert to YOLO Format") 
    print("Command: python convert_labelme_to_yolo.py")
    print()
    print("STEP 5: Train Model")
    print("Command: yolo train model=yolov8n.pt data=yolo_triangle_dataset/dataset.yaml epochs=100")
    print()
    print("=" * 60)
    print("ANNOTATION TARGETS:")
    print("- possession_indicator: LEFT triangle (between team names)")
    print("- territory_indicator: RIGHT triangle (next to yard line)")
    print("=" * 60)

def main():
    print("Setting up Triangle Annotation Workflow")
    print("=" * 40)
    
    # Setup directories
    setup_directories()
    
    # Copy existing images
    image_count = copy_training_images()
    
    # Create configuration
    create_class_config()
    
    # Print instructions
    print_workflow_instructions()
    
    print(f"\nSetup complete!")
    print(f"Images ready for annotation: {image_count}")
    print(f"Next: Run 'labelme images_to_annotate/ --output labelme_annotations/'")

if __name__ == "__main__":
    main() 
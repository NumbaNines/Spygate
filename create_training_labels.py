#!/usr/bin/env python3
"""
Create the missing label files for duplicated training images.
"""

import shutil
from pathlib import Path

def create_training_labels():
    """Create label files for all duplicated training images."""
    print("🔧 Creating missing training labels...")
    
    # The 3 base images that were duplicated
    base_images = [
        "monitor3_screenshot_20250608_021042_6",
        "monitor3_screenshot_20250608_021427_50", 
        "monitor3_screenshot_20250608_021044_7"
    ]
    
    train_img_dir = Path("training_data/train/images")
    train_lbl_dir = Path("training_data/train/labels")
    base_lbl_dir = Path("training_data/labels")
    
    # Ensure labels directory exists
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    labels_created = 0
    
    # Process each base image
    for base_name in base_images:
        print(f"\n📸 Processing duplicates of: {base_name}")
        
        # Find the original label file
        base_label = base_lbl_dir / f"{base_name}.txt"
        if not base_label.exists():
            print(f"   ❌ Base label not found: {base_label}")
            continue
        
        print(f"   ✅ Found base label: {base_label}")
        
        # Find all duplicated images for this base
        duplicated_images = list(train_img_dir.glob(f"{base_name}_copy_*.png"))
        print(f"   📂 Found {len(duplicated_images)} duplicated images")
        
        # Create label file for each duplicated image
        for img_file in duplicated_images:
            label_name = img_file.stem + ".txt"
            target_label = train_lbl_dir / label_name
            
            # Copy the base label to the duplicated image's label
            shutil.copy2(base_label, target_label)
            labels_created += 1
            
        print(f"   ✅ Created {len(duplicated_images)} labels for {base_name}")
    
    print(f"\n🎉 Total labels created: {labels_created}")
    
    # Verify the results
    train_images = list(train_img_dir.glob("*.png"))
    train_labels = list(train_lbl_dir.glob("*.txt"))
    
    print(f"\n📊 Final counts:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Training labels: {len(train_labels)}")
    
    if len(train_images) == len(train_labels):
        print("   ✅ Perfect match! Ready for training.")
        return True
    else:
        print("   ⚠️  Mismatch between images and labels.")
        return False

def verify_label_content():
    """Verify that label files have proper content."""
    print("\n🔍 Verifying label content...")
    
    train_lbl_dir = Path("training_data/train/labels")
    label_files = list(train_lbl_dir.glob("*.txt"))
    
    if not label_files:
        print("   ❌ No label files found!")
        return False
    
    # Check a few random labels
    sample_labels = label_files[:3]
    for label_file in sample_labels:
        print(f"   📄 Checking: {label_file.name}")
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                print(f"      ❌ Empty label file!")
                return False
                
            print(f"      ✅ {len(lines)} annotations found")
            
            # Show first annotation as example
            if lines:
                first_line = lines[0].strip()
                parts = first_line.split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    coords = parts[1:5]
                    print(f"         Sample: class {class_id} at {coords}")
                else:
                    print(f"      ⚠️  Malformed annotation: {first_line}")
                    
        except Exception as e:
            print(f"      ❌ Error reading label: {e}")
            return False
    
    print("   ✅ Label verification passed!")
    return True

def main():
    """Main function."""
    print("🏈 Create Training Labels for Duplicated Images")
    print("=" * 50)
    
    # Create the missing labels
    success = create_training_labels()
    
    if success:
        # Verify label content
        verify_label_content()
        
        print(f"\n🚀 Training data is now ready!")
        print(f"   You can now run the training script.")
    else:
        print(f"\n❌ Issues found. Please check the label files.")

if __name__ == "__main__":
    main() 
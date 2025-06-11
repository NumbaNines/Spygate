#!/usr/bin/env python3
"""Duplicate specific perfect images and annotations to create massive training dataset."""

import json
import shutil
import argparse
from pathlib import Path

class SpecificImageDuplicator:
    """Duplicate specific perfect images for training data expansion."""
    
    def __init__(self, images_dir="training_data/images", labels_dir="training_data/labels"):
        """Initialize with source directories."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
    def find_image_files(self, image_names):
        """Find specific image files and their annotations."""
        print(f"🔍 Looking for {len(image_names)} specific images...")
        
        found_files = []
        
        for name in image_names:
            # Remove extension if provided
            base_name = Path(name).stem
            
            # Find the image file
            image_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_image = self.images_dir / f"{base_name}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            # Find the label file
            label_file = self.labels_dir / f"{base_name}.json"
            
            if image_file and label_file.exists():
                found_files.append({
                    'image': image_file,
                    'label': label_file,
                    'name': base_name
                })
                print(f"   ✅ Found: {base_name}")
            else:
                print(f"   ❌ Missing: {base_name} (image: {image_file is not None}, label: {label_file.exists()})")
        
        return found_files
    
    def duplicate_batch(self, source_file, batch_number, copies_per_batch=100):
        """Duplicate one specific image/annotation pair in a batch."""
        print(f"📦 Creating batch {batch_number} from {source_file['name']} ({copies_per_batch} copies)")
        
        source_image = source_file['image']
        source_label = source_file['label']
        
        # Get file extensions
        image_ext = source_image.suffix
        
        created_files = []
        
        for i in range(copies_per_batch):
            # Create new filenames
            copy_number = (batch_number - 1) * copies_per_batch + i + 1
            new_name = f"{source_file['name']}_copy_{copy_number:04d}"
            
            new_image_path = self.images_dir / f"{new_name}{image_ext}"
            new_label_path = self.labels_dir / f"{new_name}.json"
            
            try:
                # Copy image
                shutil.copy2(source_image, new_image_path)
                
                # Copy and update annotation
                with open(source_label, 'r') as f:
                    label_data = json.load(f)
                
                # Update the imagePath in the annotation
                label_data['imagePath'] = f"{new_name}{image_ext}"
                
                # Save new annotation
                with open(new_label_path, 'w') as f:
                    json.dump(label_data, f, indent=2)
                
                created_files.append({
                    'image': new_image_path,
                    'label': new_label_path,
                    'name': new_name
                })
                
            except Exception as e:
                print(f"❌ Error creating copy {copy_number}: {e}")
        
        print(f"✅ Created {len(created_files)} copies in batch {batch_number}")
        return created_files
    
    def create_dataset(self, image_files, copies_per_image=1000, batch_size=100):
        """Create massive dataset by duplicating specific images."""
        if not image_files:
            print("❌ No images provided for duplication.")
            return
        
        total_batches_per_image = copies_per_image // batch_size
        total_files_created = 0
        
        print(f"🚀 Creating {copies_per_image} copies of each specified image")
        print(f"📊 Processing {total_batches_per_image} batches of {batch_size} copies each")
        print(f"🎯 Total target: {len(image_files) * copies_per_image} training examples")
        print("=" * 60)
        
        for img_idx, image_file in enumerate(image_files):
            print(f"\n🎨 Processing image {img_idx + 1}/{len(image_files)}: {image_file['name']}")
            
            for batch_num in range(1, total_batches_per_image + 1):
                created = self.duplicate_batch(image_file, batch_num, batch_size)
                total_files_created += len(created)
                
                print(f"   📈 Progress: {batch_num}/{total_batches_per_image} batches ({len(created)} files)")
        
        print("=" * 60)
        print(f"🎉 DATASET CREATION COMPLETE!")
        print(f"📊 Total files created: {total_files_created}")
        print(f"📸 Images: {total_files_created} (perfect annotations)")
        print(f"🏷️ Labels: {total_files_created} (guaranteed accuracy)")
        
        return total_files_created


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Duplicate specific perfect images for massive training dataset")
    parser.add_argument("--copies", type=int, default=1000, help="Copies per image (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=100, help="Files per batch (default: 100)")
    parser.add_argument("--images-list", nargs='+', help="Specific image names to duplicate (without extension)")
    parser.add_argument("--find-only", action="store_true", help="Only find specified images, don't duplicate")
    parser.add_argument("--images", default="training_data/images", help="Images directory")
    parser.add_argument("--labels", default="training_data/labels", help="Labels directory")
    
    args = parser.parse_args()
    
    print("🎯 Specific Image Duplicator for Massive Training Dataset")
    print("=" * 60)
    
    # Use provided image names or ask for them
    if args.images_list:
        image_names = args.images_list
    else:
        # Default to asking for the 3 specific images
        print("📋 Please specify which 3 images you want to duplicate.")
        print("💡 Example image names from your dataset:")
        print("   - monitor3_screenshot_20250608_021042_6")
        print("   - monitor3_screenshot_20250608_021044_7") 
        print("   - monitor3_screenshot_20250608_021054_11")
        print()
        
        image_names = []
        for i in range(3):
            name = input(f"Enter image {i+1} name (without extension): ").strip()
            if name:
                image_names.append(name)
        
        if len(image_names) == 0:
            print("❌ No image names provided!")
            return
    
    # Initialize duplicator
    duplicator = SpecificImageDuplicator(args.images, args.labels)
    
    # Find specified images
    found_files = duplicator.find_image_files(image_names)
    
    if not found_files:
        print("❌ No specified images found!")
        return
    
    print(f"\n✅ Found {len(found_files)} images to duplicate:")
    for file in found_files:
        print(f"   📸 {file['name']}")
    
    if args.find_only:
        print("✅ Found specified images. Use without --find-only to start duplication.")
        return
    
    # Calculate totals
    total_target = len(found_files) * args.copies
    print(f"\n📊 DUPLICATION PLAN:")
    print(f"   Specified images: {len(found_files)}")
    print(f"   Copies per image: {args.copies}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Total target: {total_target} training examples")
    
    # Confirm before proceeding
    response = input(f"\n⚠️ This will create {total_target} files. Continue? (y/n): ")
    if response.lower() != 'y':
        print("⏹️ Operation cancelled.")
        return
    
    # Create the dataset
    duplicator.create_dataset(found_files, args.copies, args.batch_size)


if __name__ == "__main__":
    main() 
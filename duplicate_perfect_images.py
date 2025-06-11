#!/usr/bin/env python3
"""Duplicate perfect images and annotations to create massive training dataset."""

import json
import shutil
import argparse
from pathlib import Path

class ImageDuplicator:
    """Duplicate perfect images and annotations for training data expansion."""
    
    def __init__(self, images_dir="training_data/images", labels_dir="training_data/labels"):
        """Initialize with source directories."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.perfect_images = []
        self.perfect_labels = []
        
    def find_perfect_images(self):
        """Find images that have both triangle indicators annotated."""
        print("🔍 Finding perfectly annotated images...")
        
        perfect_files = []
        
        for label_file in self.labels_dir.glob("*.json"):
            try:
                with open(label_file, 'r') as f:
                    data = json.load(f)
                
                # Count triangle annotations
                triangle_classes = {"possession_indicator", "territory_indicator"}
                found_triangles = set()
                
                for shape in data.get('shapes', []):
                    if shape['label'] in triangle_classes:
                        found_triangles.add(shape['label'])
                
                # Check if both triangles are present
                if len(found_triangles) == 2:
                    # Find corresponding image
                    image_name = label_file.stem  # filename without extension
                    
                    # Try different image extensions
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_file = self.images_dir / f"{image_name}{ext}"
                        if image_file.exists():
                            perfect_files.append({
                                'image': image_file,
                                'label': label_file,
                                'name': image_name
                            })
                            break
                            
            except Exception as e:
                print(f"⚠️ Error checking {label_file}: {e}")
        
        self.perfect_images = perfect_files
        print(f"✅ Found {len(perfect_files)} perfectly annotated images:")
        for file in perfect_files:
            print(f"   📸 {file['name']}")
        
        return perfect_files
    
    def duplicate_batch(self, source_file, batch_number, copies_per_batch=100):
        """Duplicate one perfect image/annotation pair in a batch."""
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
    
    def create_dataset(self, copies_per_image=1000, batch_size=100):
        """Create massive dataset by duplicating perfect images."""
        if not self.perfect_images:
            print("❌ No perfect images found. Run find_perfect_images() first.")
            return
        
        total_batches_per_image = copies_per_image // batch_size
        total_files_created = 0
        
        print(f"🚀 Creating {copies_per_image} copies of each perfect image")
        print(f"📊 Processing {total_batches_per_image} batches of {batch_size} copies each")
        print(f"🎯 Total target: {len(self.perfect_images) * copies_per_image} training examples")
        print("=" * 60)
        
        for img_idx, perfect_file in enumerate(self.perfect_images):
            print(f"\n🎨 Processing image {img_idx + 1}/{len(self.perfect_images)}: {perfect_file['name']}")
            
            for batch_num in range(1, total_batches_per_image + 1):
                created = self.duplicate_batch(perfect_file, batch_num, batch_size)
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
    parser = argparse.ArgumentParser(description="Duplicate perfect images for massive training dataset")
    parser.add_argument("--copies", type=int, default=1000, help="Copies per perfect image (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=100, help="Files per batch (default: 100)")
    parser.add_argument("--find-only", action="store_true", help="Only find perfect images, don't duplicate")
    parser.add_argument("--images", default="training_data/images", help="Images directory")
    parser.add_argument("--labels", default="training_data/labels", help="Labels directory")
    
    args = parser.parse_args()
    
    print("🎯 Perfect Image Duplicator for Massive Training Dataset")
    print("=" * 60)
    
    # Initialize duplicator
    duplicator = ImageDuplicator(args.images, args.labels)
    
    # Find perfect images
    perfect_files = duplicator.find_perfect_images()
    
    if not perfect_files:
        print("❌ No perfect images found!")
        print("💡 Make sure you have images with both possession_indicator and territory_indicator annotated.")
        return
    
    if args.find_only:
        print("✅ Found perfect images. Use without --find-only to start duplication.")
        return
    
    # Calculate totals
    total_target = len(perfect_files) * args.copies
    print(f"\n📊 DUPLICATION PLAN:")
    print(f"   Perfect images: {len(perfect_files)}")
    print(f"   Copies per image: {args.copies}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Total target: {total_target} training examples")
    
    # Confirm before proceeding
    response = input(f"\n⚠️ This will create {total_target} files. Continue? (y/n): ")
    if response.lower() != 'y':
        print("⏹️ Operation cancelled.")
        return
    
    # Create the dataset
    duplicator.create_dataset(args.copies, args.batch_size)


if __name__ == "__main__":
    main() 
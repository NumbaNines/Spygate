#!/usr/bin/env python3
"""
Augment triangle detection training data to create massive training dataset.

This script takes the 3 manually annotated images and creates augmented versions
to build a robust training dataset of 2500 images for triangle detection.

Usage:
    python augment_triangle_data.py
"""

import json
import os
import random
import shutil
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class TriangleDataAugmentor:
    def __init__(self, 
                 source_dir="labelme_annotations",
                 images_dir="images_to_annotate", 
                 output_dir="augmented_triangle_annotations",
                 target_total=2500):  # MASSIVE training dataset!
        """
        Initialize the augmentor.
        
        Args:
            source_dir: Directory containing original labelme annotations  
            images_dir: Directory containing additional images (without annotations)
            output_dir: Directory to save augmented data
            target_total: Total number of augmented images to create (2500!)
        """
        self.source_dir = Path(source_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.target_total = target_total
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Clear existing augmented data
        for file in self.output_dir.glob("*"):
            if file.is_file():
                file.unlink()
    
    def extract_image_from_json(self, json_path):
        """Extract embedded image from labelme JSON file."""
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        if 'imageData' in annotation and annotation['imageData']:
            # Decode base64 image data
            image_data = base64.b64decode(annotation['imageData'])
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        
        return None
    
    def get_source_files(self):
        """Get all JSON annotation files from source directory."""
        json_files = list(self.source_dir.glob("*.json"))
        print(f"Found {len(json_files)} source annotation files")
        return json_files
    
    def get_additional_images(self):
        """Get additional image files for augmentation."""
        # Look for common image extensions
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(self.images_dir.glob(ext)))
        
        print(f"Found {len(image_files)} additional images")
        return image_files
    
    def augment_image(self, image, severity_factor=0.3):
        """
        Apply various augmentations to an image.
        
        Args:
            image: PIL Image or file path
            severity_factor: How strong the augmentations should be (0.0 to 1.0)
        
        Returns:
            Augmented PIL Image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Apply random augmentations
        augmentations = [
            self._brightness_adjustment,
            self._contrast_adjustment, 
            self._saturation_adjustment,
            self._hue_adjustment,
            self._gaussian_noise,
            self._gaussian_blur,
            self._rotation,
            self._horizontal_flip,
            self._scaling,
            self._color_jitter
        ]
        
        # Randomly select and apply 2-5 augmentations for more variety
        num_augmentations = random.randint(2, 5)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        for aug_func in selected_augmentations:
            image = aug_func(image, severity_factor)
        
        return image
    
    def _brightness_adjustment(self, image, factor):
        """Adjust image brightness."""
        enhancer = ImageEnhance.Brightness(image)
        # Brightness range: 0.6 to 1.4 (wider range)
        brightness = 1.0 + (random.random() - 0.5) * factor * 0.8
        return enhancer.enhance(max(0.6, min(1.4, brightness)))
    
    def _contrast_adjustment(self, image, factor):
        """Adjust image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        # Contrast range: 0.7 to 1.3 (wider range)
        contrast = 1.0 + (random.random() - 0.5) * factor * 0.6
        return enhancer.enhance(max(0.7, min(1.3, contrast)))
    
    def _saturation_adjustment(self, image, factor):
        """Adjust image saturation."""
        enhancer = ImageEnhance.Color(image)
        # Saturation range: 0.7 to 1.3 (wider range)
        saturation = 1.0 + (random.random() - 0.5) * factor * 0.6
        return enhancer.enhance(max(0.7, min(1.3, saturation)))
    
    def _hue_adjustment(self, image, factor):
        """Slight hue adjustment by converting to HSV."""
        # Convert to numpy array for HSV manipulation
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust hue slightly
        hue_shift = (random.random() - 0.5) * factor * 30  # ±15 degrees max
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)
    
    def _gaussian_noise(self, image, factor):
        """Add subtle gaussian noise."""
        img_array = np.array(image).astype(np.float32)
        
        # Add noise
        noise_std = factor * 15  # Max std of 4.5 for factor=0.3
        noise = np.random.normal(0, noise_std, img_array.shape)
        
        # Apply noise and clip values
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    def _gaussian_blur(self, image, factor):
        """Apply subtle gaussian blur."""
        # Blur radius between 0.3 and 2.0
        radius = 0.3 + factor * 1.7
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _rotation(self, image, factor):
        """Apply small rotation."""
        # Rotation angle between -8 and 8 degrees
        angle = (random.random() - 0.5) * factor * 16
        return image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    def _horizontal_flip(self, image, factor):
        """Randomly apply horizontal flip."""
        if random.random() < factor * 0.5:  # 15% chance at factor=0.3
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    def _scaling(self, image, factor):
        """Apply slight scaling."""
        # Scale factor between 0.9 and 1.1
        scale = 1.0 + (random.random() - 0.5) * factor * 0.2
        
        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize and crop/pad to original size
        scaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new image with original size
        result = Image.new('RGB', (image.width, image.height), (0, 0, 0))
        
        # Paste scaled image in center
        x_offset = (image.width - new_width) // 2
        y_offset = (image.height - new_height) // 2
        result.paste(scaled, (x_offset, y_offset))
        
        return result
    
    def _color_jitter(self, image, factor):
        """Apply random color channel adjustments."""
        img_array = np.array(image).astype(np.float32)
        
        # Apply random multipliers to each channel
        for channel in range(3):
            multiplier = 1.0 + (random.random() - 0.5) * factor * 0.3
            img_array[:, :, channel] *= multiplier
        
        # Clip and convert back
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def augment_annotation(self, json_path, augmented_image_name, flip_applied=False):
        """
        Create augmented annotation file.
        
        Args:
            json_path: Path to original JSON annotation
            augmented_image_name: Name of the augmented image file
            flip_applied: Whether horizontal flip was applied
        
        Returns:
            Dictionary containing the augmented annotation
        """
        # Load original annotation
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        # Update image path
        annotation['imagePath'] = augmented_image_name
        
        # Remove embedded image data to save space
        if 'imageData' in annotation:
            del annotation['imageData']
        
        # If horizontal flip was applied, we need to flip the annotations
        if flip_applied:
            image_width = annotation.get('imageWidth', 1920)  # Default width
            
            for shape in annotation['shapes']:
                # Flip x coordinates for all points
                for point in shape['points']:
                    point[0] = image_width - point[0]
        
        return annotation
    
    def create_synthetic_annotation(self, image_path, augmented_image_name):
        """
        Create synthetic annotations for images without existing annotations.
        This creates empty annotations for now - could be enhanced with auto-detection.
        """
        image = Image.open(image_path)
        
        # Basic annotation structure
        annotation = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],  # No shapes for synthetic annotations
            "imagePath": augmented_image_name,
            "imageData": None,
            "imageHeight": image.height,
            "imageWidth": image.width
        }
        
        return annotation
    
    def generate_augmented_dataset(self):
        """Generate the complete augmented dataset with 2500 images."""
        source_files = self.get_source_files()
        additional_images = self.get_additional_images()
        
        if len(source_files) == 0:
            print("❌ No source annotation files found!")
            return
        
        # Strategy: Use annotated files for most images, supplement with additional images
        annotated_target = int(self.target_total * 0.8)  # 80% from annotated files = 2000 images
        additional_target = self.target_total - annotated_target  # 20% from additional = 500 images
        
        print(f"🎯 Target: {self.target_total} total augmented images")
        print(f"📊 Strategy:")
        print(f"   - {annotated_target} images from {len(source_files)} annotated files")
        print(f"   - {additional_target} images from {len(additional_images)} additional images")
        
        total_generated = 0
        
        # Process annotated files first
        images_per_source = annotated_target // len(source_files)
        extra_images = annotated_target % len(source_files)
        
        print(f"\n🔸 Processing annotated files:")
        print(f"   - Base allocation: {images_per_source} images per source file")
        if extra_images > 0:
            print(f"   - Extra: {extra_images} additional images for first {extra_images} sources")
        
        for i, json_file in enumerate(source_files):
            # Calculate number of images for this source
            num_images = images_per_source
            if i < extra_images:
                num_images += 1
            
            print(f"\n🖼️  Processing {json_file.name} -> {num_images} augmented images")
            
            # Extract image from JSON
            source_image = self.extract_image_from_json(json_file)
            if source_image is None:
                print(f"⚠️  Could not extract image from {json_file.name}")
                continue
            
            # Generate augmented versions
            for j in range(num_images):
                # Create unique filename
                aug_image_name = f"annotated_{json_file.stem}_{j:04d}.jpg"
                aug_json_name = f"annotated_{json_file.stem}_{j:04d}.json"
                
                aug_image_path = self.output_dir / aug_image_name
                aug_json_path = self.output_dir / aug_json_name
                
                # Apply augmentations with varying severity
                severity = random.uniform(0.2, 0.5)  # Random severity for variety
                
                # Track if horizontal flip is applied
                flip_applied = random.random() < 0.2  # 20% chance of flip
                
                # Generate augmented image
                try:
                    augmented_image = self.augment_image(source_image, severity)
                    
                    # Apply horizontal flip if selected
                    if flip_applied:
                        augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # Save augmented image
                    augmented_image.save(aug_image_path, quality=92, optimize=True)
                    
                    # Create augmented annotation
                    augmented_annotation = self.augment_annotation(json_file, aug_image_name, flip_applied)
                    
                    # Save augmented annotation
                    with open(aug_json_path, 'w') as f:
                        json.dump(augmented_annotation, f, indent=2)
                    
                    total_generated += 1
                    
                    if (j + 1) % 100 == 0:  # Progress update every 100 images
                        print(f"   Generated {j + 1}/{num_images} images...")
                
                except Exception as e:
                    print(f"❌ Error generating augmentation {j}: {e}")
                    continue
        
        # Process additional images
        if additional_images and additional_target > 0:
            print(f"\n🔸 Processing additional images for diversity:")
            
            images_per_additional = min(additional_target // len(additional_images), 50)  # Max 50 per image
            if images_per_additional == 0:
                images_per_additional = 1
            
            remaining_target = additional_target
            
            for image_file in additional_images[:remaining_target]:
                if remaining_target <= 0:
                    break
                
                num_images = min(images_per_additional, remaining_target)
                
                # Generate augmented versions
                for j in range(num_images):
                    # Create unique filename
                    aug_image_name = f"additional_{image_file.stem}_{j:04d}.jpg"
                    aug_json_name = f"additional_{image_file.stem}_{j:04d}.json"
                    
                    aug_image_path = self.output_dir / aug_image_name
                    aug_json_path = self.output_dir / aug_json_name
                    
                    # Apply augmentations with varying severity
                    severity = random.uniform(0.3, 0.6)  # Higher severity for more diversity
                    
                    # Generate augmented image
                    try:
                        augmented_image = self.augment_image(image_file, severity)
                        
                        # Save augmented image
                        augmented_image.save(aug_image_path, quality=92, optimize=True)
                        
                        # Create synthetic annotation (no shapes)
                        augmented_annotation = self.create_synthetic_annotation(image_file, aug_image_name)
                        
                        # Save augmented annotation
                        with open(aug_json_path, 'w') as f:
                            json.dump(augmented_annotation, f, indent=2)
                        
                        total_generated += 1
                        remaining_target -= 1
                        
                    except Exception as e:
                        print(f"❌ Error generating additional augmentation {j}: {e}")
                        continue
        
        print(f"\n✅ Dataset augmentation complete!")
        print(f"📊 Generated {total_generated} augmented image pairs")
        print(f"📁 Saved to: {self.output_dir}")
        print(f"💪 Ready for MASSIVE training performance boost!")
        
        return total_generated


def main():
    print("🚀 Starting MASSIVE Triangle Data Augmentation (2500 images!)")
    print("=" * 70)
    
    # Create augmentor with massive target
    augmentor = TriangleDataAugmentor(target_total=2500)  # MASSIVE dataset!
    
    # Generate augmented dataset
    total_generated = augmentor.generate_augmented_dataset()
    
    if total_generated > 0:
        print(f"\n🎉 Successfully created {total_generated} augmented training examples!")
        print(f"💪 This massive dataset will give exceptional model performance!")
        print(f"\n📋 Next steps:")
        print(f"1. Convert to YOLO format: python convert_labelme_to_yolo.py --input augmented_triangle_annotations")
        print(f"2. Train the model: python train_correct_triangle_model.py")
        print(f"3. Expect AMAZING triangle detection performance! 🎯")
    else:
        print("\n❌ No augmented data was generated. Check your source files.")


if __name__ == "__main__":
    main() 
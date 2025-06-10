#!/usr/bin/env python3
"""
SpygateAI Profile Picture Manager
================================

Handles custom profile picture uploads, processing, and storage.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw


class ProfilePictureManager:
    """Manages custom profile picture operations"""

    def __init__(self, storage_dir: str = "profile_pictures"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Supported image formats
        self.supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        # Profile picture dimensions
        self.size = (128, 128)  # Standard size for profile pictures

    def validate_image(self, file_path: str) -> bool:
        """Validate if the file is a supported image"""
        try:
            path = Path(file_path)

            # Check file extension
            if path.suffix.lower() not in self.supported_formats:
                return False

            # Try to open with PIL to validate it's a real image
            with Image.open(file_path) as img:
                img.verify()

            return True
        except Exception:
            return False

    def process_image(self, file_path: str, user_id: int) -> Optional[str]:
        """Process and save custom profile picture"""
        try:
            if not self.validate_image(file_path):
                raise ValueError("Invalid image file")

            # Generate unique filename
            file_extension = Path(file_path).suffix.lower()
            unique_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}{file_extension}"
            output_path = self.storage_dir / unique_filename

            # Process the image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (handles RGBA, CMYK, etc.)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize and crop to make it square
                processed_img = self.create_circular_crop(img)

                # Save the processed image
                processed_img.save(output_path, "JPEG", quality=90, optimize=True)

            return str(output_path)

        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

    def create_circular_crop(self, img: Image.Image) -> Image.Image:
        """Create a circular crop of the image"""
        # Make the image square by cropping to the center
        width, height = img.size
        size = min(width, height)

        # Calculate crop box for center square
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        # Crop to square
        img = img.crop((left, top, right, bottom))

        # Resize to target size
        img = img.resize(self.size, Image.Resampling.LANCZOS)

        # Create circular mask
        mask = Image.new("L", self.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0) + self.size, fill=255)

        # Apply circular mask
        circular_img = Image.new("RGBA", self.size, (0, 0, 0, 0))
        circular_img.paste(img, (0, 0))
        circular_img.putalpha(mask)

        # Convert back to RGB with white background
        final_img = Image.new("RGB", self.size, (255, 255, 255))
        final_img.paste(circular_img, (0, 0), circular_img)

        return final_img

    def delete_profile_picture(self, file_path: str) -> bool:
        """Delete a custom profile picture file"""
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
                return True
        except Exception as e:
            print(f"❌ Error deleting profile picture: {e}")
        return False

    def get_default_emoji_profiles(self) -> list:
        """Get list of default emoji profile options"""
        return [
            ("🏈", "Football"),
            ("👤", "User"),
            ("🎯", "Target"),
            ("⚡", "Lightning"),
            ("🔥", "Fire"),
            ("💪", "Strength"),
            ("🧠", "Brain"),
            ("🏆", "Trophy"),
            ("⭐", "Star"),
            ("🎮", "Gaming"),
        ]

    def create_temporary_preview(self, file_path: str) -> Optional[str]:
        """Create a temporary preview of the uploaded image"""
        try:
            if not self.validate_image(file_path):
                return None

            # Create temporary file
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"spygate_preview_{uuid.uuid4().hex[:8]}.jpg"

            # Process image for preview
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Create smaller preview
                preview_img = self.create_circular_crop(img)
                preview_img = preview_img.resize((64, 64), Image.Resampling.LANCZOS)
                preview_img.save(temp_file, "JPEG", quality=80)

            return str(temp_file)

        except Exception as e:
            print(f"❌ Error creating preview: {e}")
            return None

    def cleanup_old_pictures(self, user_id: int, keep_current: str = None):
        """Clean up old profile pictures for a user (keep only the current one)"""
        try:
            pattern = f"user_{user_id}_*"
            for file_path in self.storage_dir.glob(pattern):
                if keep_current and str(file_path) == keep_current:
                    continue
                file_path.unlink()
                print(f"🗑️ Cleaned up old profile picture: {file_path.name}")
        except Exception as e:
            print(f"❌ Error cleaning up old pictures: {e}")


# Convenience functions
def get_profile_picture_manager() -> ProfilePictureManager:
    """Get a shared instance of the profile picture manager"""
    return ProfilePictureManager()


def is_emoji_profile(profile_picture: str) -> bool:
    """Check if a profile picture string is an emoji (not a file path)"""
    return len(profile_picture) <= 4 and not Path(profile_picture).exists()

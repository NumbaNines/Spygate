#!/usr/bin/env python3
"""
SpygateAI Dataset Organization Module
This module handles the organization of image and label files into a YOLO dataset structure.
"""

import logging
import os
import random
import shutil
from typing import List, Tuple


class DatasetOrganizer:
    """Handles dataset organization for YOLO training."""

    def __init__(self, base_path: str):
        """
        Initialize the dataset organizer.

        Args:
            base_path: Base path where the dataset will be organized
        """
        self.base_path = base_path
        self.source_dir = os.path.join(base_path, "resized_1920x1080")

        # Verify source directory exists
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(
                f"Source directory not found: {self.source_dir}. "
                "Please ensure you're running the script from the correct directory."
            )

    def create_directory_structure(self) -> None:
        """Create the required directory structure for the dataset."""
        directories = (
            [
                os.path.join(self.base_path, "test_dataset", "images", split)
                for split in ["train", "val", "test"]
            ]
            + [
                os.path.join(self.base_path, "test_dataset", "labels", split)
                for split in ["train", "val", "test"]
            ]
            + [os.path.join(self.source_dir, "unannotated")]
        )

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            except Exception as e:
                logging.error(f"Error creating directory {directory}: {str(e)}")
                raise

    def find_annotated_pairs(self) -> list[tuple[str, str]]:
        """
        Find matching pairs of images and annotation files.

        Returns:
            List of tuples containing paths to matching image and annotation files
        """
        # First, get all txt files from the source directory
        txt_files = {
            f
            for f in os.listdir(self.source_dir)
            if f.lower().endswith(".txt") and f != "classes.txt"
        }

        # Get all PNG files from both source and unannotated directories
        unannotated_dir = os.path.join(self.source_dir, "unannotated")
        png_files_source = {f for f in os.listdir(self.source_dir) if f.lower().endswith(".png")}
        png_files_unannotated = set()

        if os.path.exists(unannotated_dir):
            png_files_unannotated = {
                f for f in os.listdir(unannotated_dir) if f.lower().endswith(".png")
            }

        annotated_pairs = []

        # Process each txt file
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            png_file = f"{base_name}.png"

            # Check if matching PNG is in source directory
            if png_file in png_files_source:
                annotated_pairs.append(
                    (
                        os.path.join(self.source_dir, png_file),
                        os.path.join(self.source_dir, txt_file),
                    )
                )
            # Check if matching PNG is in unannotated directory
            elif png_file in png_files_unannotated:
                # Move the PNG file to the source directory
                try:
                    shutil.move(
                        os.path.join(unannotated_dir, png_file),
                        os.path.join(self.source_dir, png_file),
                    )
                    logging.info(f"Moved annotated image from unannotated dir: {png_file}")
                    annotated_pairs.append(
                        (
                            os.path.join(self.source_dir, png_file),
                            os.path.join(self.source_dir, txt_file),
                        )
                    )
                except Exception as e:
                    logging.error(f"Error moving file {png_file}: {str(e)}")
                    raise

        # Move remaining unannotated PNGs
        if os.path.exists(unannotated_dir):
            for png_file in png_files_unannotated:
                base_name = os.path.splitext(png_file)[0]
                txt_file = f"{base_name}.txt"

                # If no matching txt file exists, keep it in unannotated
                if txt_file not in txt_files and os.path.exists(
                    os.path.join(unannotated_dir, png_file)
                ):
                    logging.info(f"Keeping unannotated image: {png_file}")

        return annotated_pairs

    def split_dataset(self, annotated_pairs: list[tuple[str, str]]) -> None:
        """
        Split the dataset into train/val/test sets and move files.

        Args:
            annotated_pairs: List of tuples containing paths to matching image and annotation files
        """
        # Shuffle the pairs to ensure random distribution
        random.shuffle(annotated_pairs)

        # Calculate split indices
        total_pairs = len(annotated_pairs)
        train_idx = int(0.7 * total_pairs)
        val_idx = int(0.9 * total_pairs)

        # Split the pairs
        train_pairs = annotated_pairs[:train_idx]
        val_pairs = annotated_pairs[train_idx:val_idx]
        test_pairs = annotated_pairs[val_idx:]

        # Move files to their respective directories
        splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        for split_name, pairs in splits.items():
            for img_path, txt_path in pairs:
                try:
                    # Move image
                    dst_img = os.path.join(
                        self.base_path,
                        "test_dataset",
                        "images",
                        split_name,
                        os.path.basename(img_path),
                    )
                    shutil.copy2(img_path, dst_img)

                    # Move annotation
                    dst_txt = os.path.join(
                        self.base_path,
                        "test_dataset",
                        "labels",
                        split_name,
                        os.path.basename(txt_path),
                    )
                    shutil.copy2(txt_path, dst_txt)

                    logging.info(f"Moved pair to {split_name} set: {os.path.basename(img_path)}")
                except Exception as e:
                    logging.error(f"Error moving files to {split_name} set: {str(e)}")
                    raise

    def create_yaml_config(self) -> None:
        """Create the data.yaml configuration file."""
        yaml_content = """path: ./  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (relative to 'path')

# Classes
nc: 6  # number of classes
names: ['hud', 'gamertag', 'preplay', 'playcall', 'no huddle', 'audible']  # class names"""

        yaml_path = os.path.join(self.base_path, "test_dataset", "data.yaml")
        try:
            with open(yaml_path, "w") as f:
                f.write(yaml_content)
            logging.info(f"Created YAML configuration file: {yaml_path}")
        except Exception as e:
            logging.error(f"Error creating YAML file: {str(e)}")
            raise

    def organize(self) -> None:
        """Main method to organize the dataset."""
        try:
            # Create directory structure
            self.create_directory_structure()

            # Find and process annotated pairs
            annotated_pairs = self.find_annotated_pairs()
            if not annotated_pairs:
                raise ValueError("No annotated image-text pairs found in the source directory.")
            logging.info(f"Found {len(annotated_pairs)} annotated image-text pairs")

            # Split and organize dataset
            self.split_dataset(annotated_pairs)

            # Create YAML configuration
            self.create_yaml_config()

            logging.info("Dataset organization completed successfully!")

        except Exception as e:
            logging.error(f"An error occurred during dataset organization: {str(e)}")
            raise

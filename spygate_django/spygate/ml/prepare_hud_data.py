"""Prepare training data for YOLO11 HUD detection model."""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HUDDatasetPreparer:
    """Prepare training data for YOLO11 HUD detection."""

    def __init__(self, data_dir: str):
        """Initialize dataset preparer.

        Args:
            data_dir: Root directory for dataset
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"

        # Class mapping
        self.classes = {
            "score_bug": 0,
            "down_distance": 1,
            "game_clock": 2,
            "play_clock": 3,
            "score_home": 4,
            "score_away": 5,
            "possession": 6,
            "yard_line": 7,
            "timeout_indicator": 8,
            "penalty_indicator": 9,
        }

    def setup_directories(self):
        """Create necessary directories for dataset."""
        for split in [self.train_dir, self.val_dir, self.test_dir]:
            (split / "images").mkdir(parents=True, exist_ok=True)
            (split / "labels").mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, output_dir: str, interval: int = 30) -> list[str]:
        """Extract frames from video at specified interval.

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            interval: Frame extraction interval

        Returns:
            List of paths to extracted frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Calculate frames to extract
            frames_to_extract = frame_count // interval
            frame_paths = []

            with tqdm(total=frames_to_extract, desc="Extracting frames") as pbar:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % interval == 0:
                        frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        pbar.update(1)

                    frame_idx += 1

            cap.release()
            return frame_paths

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return []

    def convert_annotations(self, annotations: dict, image_size: tuple[int, int]) -> str:
        """Convert annotations to YOLO format.

        Args:
            annotations: Dictionary of bounding box annotations
            image_size: (width, height) of the image

        Returns:
            String containing YOLO format annotations
        """
        yolo_lines = []
        img_w, img_h = image_size

        for ann in annotations:
            class_name = ann["class"]
            if class_name not in self.classes:
                continue

            class_id = self.classes[class_name]
            x1, y1, x2, y2 = ann["bbox"]

            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (x1 + x2) / (2 * img_w)
            y_center = (y1 + y2) / (2 * img_h)
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return "\n".join(yolo_lines)

    def prepare_dataset(
        self,
        videos_dir: str,
        annotations_file: str,
        split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1),
    ):
        """Prepare complete dataset from videos and annotations.

        Args:
            videos_dir: Directory containing input videos
            annotations_file: Path to JSON file with annotations
            split_ratio: (train, val, test) split ratios
        """
        try:
            # Set up directories
            self.setup_directories()

            # Load annotations
            with open(annotations_file) as f:
                annotations = json.load(f)

            # Process each video
            video_files = [f for f in os.listdir(videos_dir) if f.endswith((".mp4", ".avi"))]
            for video_file in tqdm(video_files, desc="Processing videos"):
                video_path = os.path.join(videos_dir, video_file)
                video_name = os.path.splitext(video_file)[0]

                # Extract frames
                frames = self.extract_frames(video_path, str(self.data_dir / "temp"))

                # Split frames into train/val/test
                np.random.shuffle(frames)
                n_frames = len(frames)
                n_train = int(n_frames * split_ratio[0])
                n_val = int(n_frames * split_ratio[1])

                train_frames = frames[:n_train]
                val_frames = frames[n_train : n_train + n_val]
                test_frames = frames[n_train + n_val :]

                # Process each split
                splits = {
                    "train": (self.train_dir, train_frames),
                    "val": (self.val_dir, val_frames),
                    "test": (self.test_dir, test_frames),
                }

                for split_name, (split_dir, split_frames) in splits.items():
                    for frame_path in tqdm(split_frames, desc=f"Processing {split_name} split"):
                        frame_name = os.path.basename(frame_path)
                        frame_id = frame_name.replace("frame_", "").replace(".jpg", "")

                        # Copy image
                        shutil.copy2(frame_path, split_dir / "images" / frame_name)

                        # Get image size for normalization
                        img = cv2.imread(frame_path)
                        img_h, img_w = img.shape[:2]

                        # Convert and save annotations
                        if frame_id in annotations:
                            yolo_anns = self.convert_annotations(
                                annotations[frame_id], (img_w, img_h)
                            )
                            label_path = split_dir / "labels" / frame_name.replace(".jpg", ".txt")
                            with open(label_path, "w") as f:
                                f.write(yolo_anns)

                # Clean up temporary files
                shutil.rmtree(str(self.data_dir / "temp"))

            logger.info("Dataset preparation completed successfully")

        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise


def main():
    """Main entry point for dataset preparation script."""
    parser = argparse.ArgumentParser(description="Prepare YOLO11 HUD detection dataset")
    parser.add_argument(
        "--videos", type=str, required=True, help="Directory containing input videos"
    )
    parser.add_argument(
        "--annotations", type=str, required=True, help="Path to JSON annotations file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="spygate/models/yolo11/data",
        help="Output directory for prepared dataset",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of data for testing")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("dataset_preparation.log")],
    )

    try:
        preparer = HUDDatasetPreparer(args.output)
        preparer.prepare_dataset(
            args.videos, args.annotations, (args.train_ratio, args.val_ratio, args.test_ratio)
        )
        logger.info("Dataset preparation completed successfully")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

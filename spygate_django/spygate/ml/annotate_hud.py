"""Interactive tool for annotating HUD elements in game footage."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HUDAnnotator:
    """Interactive tool for annotating HUD elements in game footage."""

    def __init__(self, output_file: str):
        """Initialize the annotator.

        Args:
            output_file: Path to save annotations JSON
        """
        self.output_file = Path(output_file)
        self.annotations = {}
        self.current_frame = None
        self.frame_id = None
        self.drawing = False
        self.start_point = None
        self.current_class = "score_bug"
        self.window_name = "HUD Annotator"

        # Class definitions with colors
        self.classes = {
            "score_bug": (255, 0, 0),  # Red
            "down_distance": (0, 255, 0),  # Green
            "game_clock": (0, 0, 255),  # Blue
            "play_clock": (255, 255, 0),  # Yellow
            "score_home": (255, 0, 255),  # Magenta
            "score_away": (0, 255, 255),  # Cyan
            "possession": (128, 0, 0),  # Dark Red
            "yard_line": (0, 128, 0),  # Dark Green
            "timeout_indicator": (0, 0, 128),  # Dark Blue
            "penalty_indicator": (128, 128, 0),  # Dark Yellow
        }

        # Load existing annotations if file exists
        if self.output_file.exists():
            with open(self.output_file) as f:
                self.annotations = json.load(f)

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: None):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw rectangle on copy of frame
            frame_copy = self.current_frame.copy()
            cv2.rectangle(frame_copy, self.start_point, (x, y), self.classes[self.current_class], 2)
            cv2.imshow(self.window_name, frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Save the bounding box
            x1, y1 = self.start_point
            x2, y2 = x, y
            # Ensure coordinates are in correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Add annotation
            if self.frame_id not in self.annotations:
                self.annotations[self.frame_id] = []

            self.annotations[self.frame_id].append(
                {"class": self.current_class, "bbox": [x1, y1, x2, y2]}
            )

            # Draw permanent rectangle
            cv2.rectangle(
                self.current_frame, (x1, y1), (x2, y2), self.classes[self.current_class], 2
            )
            cv2.imshow(self.window_name, self.current_frame)

    def save_annotations(self):
        """Save annotations to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
        logger.info(f"Annotations saved to {self.output_file}")

    def annotate_frame(self, frame: np.ndarray, frame_id: str) -> None:
        """Set up interactive annotation for a frame.

        Args:
            frame: Input frame as numpy array
            frame_id: Unique identifier for the frame
        """
        self.current_frame = frame.copy()
        self.frame_id = frame_id

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Draw existing annotations if any
        if frame_id in self.annotations:
            for ann in self.annotations[frame_id]:
                x1, y1, x2, y2 = ann["bbox"]
                cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), self.classes[ann["class"]], 2)

        while True:
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(1) & 0xFF

            # Class selection keys (1-0)
            if ord("1") <= key <= ord("0"):
                idx = key - ord("1") if key != ord("0") else 9
                if idx < len(list(self.classes.keys())):
                    self.current_class = list(self.classes.keys())[idx]
                    logger.info(f"Selected class: {self.current_class}")

            # Navigation keys
            elif key == ord("s"):  # Save
                self.save_annotations()
            elif key == ord("d"):  # Delete last annotation
                if self.frame_id in self.annotations and self.annotations[self.frame_id]:
                    self.annotations[self.frame_id].pop()
                    # Redraw frame
                    self.current_frame = frame.copy()
                    for ann in self.annotations[self.frame_id]:
                        x1, y1, x2, y2 = ann["bbox"]
                        cv2.rectangle(
                            self.current_frame, (x1, y1), (x2, y2), self.classes[ann["class"]], 2
                        )
                    cv2.imshow(self.window_name, self.current_frame)
            elif key == ord("q"):  # Quit
                break

        cv2.destroyWindow(self.window_name)

    def process_video(self, video_path: str, interval: int = 30) -> None:
        """Process video file and annotate frames at specified interval.

        Args:
            video_path: Path to input video
            interval: Frame extraction interval
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    frame_id = f"{frame_idx:06d}"
                    logger.info(f"Annotating frame {frame_id}")
                    self.annotate_frame(frame, frame_id)

                frame_idx += 1

            cap.release()
            self.save_annotations()

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise


def main():
    """Main entry point for annotation script."""
    parser = argparse.ArgumentParser(description="Annotate HUD elements in game footage")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        type=str,
        default="spygate/models/yolo11/data/annotations.json",
        help="Path to output JSON file",
    )
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("annotation.log")],
    )

    try:
        annotator = HUDAnnotator(args.output)
        annotator.process_video(args.video, args.interval)
        logger.info("Annotation completed successfully")
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()

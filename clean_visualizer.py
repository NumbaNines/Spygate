#!/usr/bin/env python3
"""Clean Madden visualization script with minimal dependencies."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Colors for visualization
COLORS = {
    "hud": (0, 255, 0),
    "possession_triangle_area": (0, 0, 255),
    "territory_triangle_area": (255, 0, 0),
    "preplay_indicator": (255, 255, 0),
    "play_call_screen": (255, 0, 255),
}


class CleanMaddenVisualizer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = YOLO("hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt")
        self.model.conf = 0.25  # Confidence threshold

        logger.info(f"Initialized HUD detection model")
        logger.info(f"Output directory: {output_dir}")

        # Custom HUD classes from your trained model
        self.ui_classes = [
            "hud",  # Main HUD bar
            "possession_triangle_area",  # Left triangle area (shows ball possession)
            "territory_triangle_area",  # Right triangle area (shows territory)
            "preplay_indicator",  # Pre-play indicator
            "play_call_screen",  # Play call screen
        ]

    def process_image(self, image_path):
        """Process a single image and generate visualization."""
        try:
            # Verify file exists
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image file not found: {image_path}")
                return

            # Read image
            image = cv2.imread(str(path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return

            # Run detection
            results = self.model(image, verbose=False)

            # Process results into our format
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    cls = int(box.cls)

                    # Make sure class index is valid
                    if cls < len(self.ui_classes):
                        class_name = self.ui_classes[cls]

                        detections.append(
                            {"bbox": [x1, y1, x2, y2], "confidence": conf, "class": class_name}
                        )

            # Create visualization
            vis_image = self.create_visualization(image, detections)

            # Save visualization
            output_path = self.output_dir / f"{path.stem}_analyzed.jpg"
            cv2.imwrite(str(output_path), vis_image)

            logger.info(f"Processed {path.name} -> {output_path.name}")

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

    def create_visualization(self, image, detections):
        """Create visualization with bounding boxes and labels."""
        # Create a copy of the image for visualization
        vis_image = image.copy()

        # Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class"]
            conf = det["confidence"]

            # Get color for this class
            color = COLORS.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                vis_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1
            )
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add info text
        cv2.putText(
            vis_image,
            f"Detections: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return vis_image

    def process_directory(self):
        """Process all images in the input directory."""
        try:
            # Process all images in directory
            image_files = []
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.extend(self.input_dir.glob(f"*{ext}"))

            if not image_files:
                logger.warning(f"No image files found in {self.input_dir}")
                return

            logger.info(f"Found {len(image_files)} images to process")
            for image_file in image_files:
                self.process_image(str(image_file))

        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(description="Clean Madden visualization tool")
    parser.add_argument("input_dir", help="Directory containing images to process")
    parser.add_argument(
        "--output-dir",
        default="visualization_results",
        help="Directory to save visualization results",
    )
    args = parser.parse_args()

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        visualizer = CleanMaddenVisualizer(Path(args.input_dir), output_dir)

        # Process input directory
        visualizer.process_directory()

    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

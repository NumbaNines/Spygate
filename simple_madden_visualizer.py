#!/usr/bin/env python3
"""Simple Madden visualization script that works with the current file structure."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.spygate.core.hardware import HardwareDetector, HardwareTier

# Import from src structure
from src.spygate.ml.yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleVisualizer:
    """Simple visualization tool for Madden screenshots."""

    def __init__(self, output_dir: str = "visualization_results"):
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save visualization results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize hardware detection
        self.hardware_detector = HardwareDetector()
        self.hardware_tier = self.hardware_detector.detect_tier()

        # Create optimization config
        self.optimization_config = OptimizationConfig(
            enable_dynamic_switching=True,
            enable_adaptive_batch_size=True,
            enable_performance_monitoring=True,
            enable_auto_optimization=True,
            max_inference_time=1.0 / 30.0,  # Target 30 FPS
            min_accuracy=0.85,
            max_memory_usage=0.8,
        )

        # Initialize YOLO model directly
        self.model = EnhancedYOLOv8(
            model_path="yolov8n.pt",
            hardware_tier=self.hardware_tier,
            optimization_config=self.optimization_config,
        )

        logger.info(f"Initialized simple visualizer with {self.hardware_tier.name} hardware")
        logger.info(f"Output directory: {self.output_dir}")

    def process_image(self, image_path: str) -> None:
        """Process a single image and generate visualization.

        Args:
            image_path: Path to the image file
        """
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
            detections = self.model.detect(image)

            # Create visualization
            vis_image = self._create_visualization(image, detections)

            # Save visualization
            output_path = self.output_dir / f"{path.stem}_analyzed.jpg"
            cv2.imwrite(str(output_path), vis_image)

            logger.info(f"Processed {path.name} -> {output_path.name}")

        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")

    def _create_visualization(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Create visualization with bounding boxes and labels.

        Args:
            image: Input image
            detections: List of detections from YOLO model

        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()

        # Draw bounding boxes and labels
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            class_name = det.get("class", "unknown")
            conf = det.get("confidence", 0.0)

            # Get color for this class (default to white if not found)
            colors = {
                "hud": (0, 255, 0),  # Green
                "possession_triangle_area": (255, 0, 0),  # Blue
                "territory_triangle_area": (0, 0, 255),  # Red
                "preplay_indicator": (255, 255, 0),  # Cyan
                "play_call_screen": (0, 255, 255),  # Yellow
            }
            color = colors.get(class_name, (255, 255, 255))

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


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(description="Simple Madden visualization tool")
    parser.add_argument("input", help="Image file or directory containing images to process")
    parser.add_argument(
        "--output-dir",
        default="visualization_results",
        help="Directory to save visualization results",
    )
    args = parser.parse_args()

    try:
        # Initialize visualizer
        visualizer = SimpleVisualizer(output_dir=args.output_dir)

        # Process input (file or directory)
        input_path = Path(args.input)
        if input_path.is_file():
            # Process single file
            visualizer.process_image(str(input_path))
        elif input_path.is_dir():
            # Process all images in directory
            image_files = []
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.extend(input_path.glob(f"*{ext}"))

            if not image_files:
                logger.warning(f"No image files found in {input_path}")
                return

            logger.info(f"Found {len(image_files)} images to process")
            for image_file in image_files:
                visualizer.process_image(str(image_file))
        else:
            logger.error(f"Input path does not exist: {input_path}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

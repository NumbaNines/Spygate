"""Visualize Madden AYOUTUB screenshots with detailed game state analysis."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

# Disable dynamic compilation to avoid Triton errors
torch._dynamo.config.suppress_errors = True

from src.spygate.core.hardware import HardwareDetector
from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.ml.yolov8_model import UI_CLASSES, OptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_visualization(image, game_state):
    """Create a simple visualization of the game state."""
    vis_image = image.copy()

    # Add game state text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    y_offset = 30
    if hasattr(game_state, "down") and game_state.down:
        cv2.putText(
            vis_image,
            f"Down: {game_state.down}",
            (10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += 30

    if hasattr(game_state, "distance") and game_state.distance:
        cv2.putText(
            vis_image,
            f"Distance: {game_state.distance}",
            (10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += 30

    if hasattr(game_state, "possession_team") and game_state.possession_team:
        cv2.putText(
            vis_image,
            f"Possession: {game_state.possession_team}",
            (10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += 30

    if hasattr(game_state, "confidence") and game_state.confidence:
        cv2.putText(
            vis_image,
            f"Confidence: {game_state.confidence:.2f}",
            (10, y_offset),
            font,
            font_scale,
            color,
            thickness,
        )
        y_offset += 30

    return vis_image


def main():
    parser = argparse.ArgumentParser(
        description="Unified Madden visualization tool (advanced pipeline)"
    )
    parser.add_argument("input_dir", help="Directory containing images to process")
    parser.add_argument(
        "--output-dir", default="visualization_results", help="Directory to save results"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect hardware
    hardware_detector = HardwareDetector()
    hardware_tier = hardware_detector.detect_tier()
    logger.info(f"Detected hardware tier: {hardware_tier.name}")

    # Create optimization config
    optimization_config = OptimizationConfig(
        enable_dynamic_switching=True,
        enable_adaptive_batch_size=True,
        enable_performance_monitoring=True,
        enable_auto_optimization=True,
        max_inference_time=1.0 / 30.0,
        min_accuracy=0.85,
        max_memory_usage=0.8,
    )

    # Use a valid YOLO model path
    model_path = "yolov8n.pt"

    analyzer = EnhancedGameAnalyzer(
        model_path=model_path, hardware=hardware_detector, optimization_config=optimization_config
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return

    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(input_dir.glob(f"*{ext}"))
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    logger.info(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        try:
            image = cv2.imread(str(image_file))
            if image is None:
                logger.error(f"Failed to read image: {image_file}")
                continue
            # Use the advanced analyzer
            game_state = analyzer.analyze_frame(image)
            # Create simple visualization
            vis_image = create_simple_visualization(image, game_state)
            output_path = output_dir / f"{image_file.stem}_analyzed.jpg"
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"Processed {image_file.name} -> {output_path.name}")
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spygate.core.hardware import HardwareDetector
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.optimization_config import OptimizationConfig
from spygate.visualization.visualization_manager import (
    VisualizationConfig,
    VisualizationManager,
    VisualizationMode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_screenshot(
    image_path: str, analyzer: EnhancedGameAnalyzer, viz_manager: VisualizationManager
) -> dict:
    """Analyze a single screenshot and visualize the detection process.

    Args:
        image_path: Path to the screenshot
        analyzer: EnhancedGameAnalyzer instance
        viz_manager: VisualizationManager instance

    Returns:
        Dictionary containing analysis results and visualization
    """
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        return None

    # Get game state analysis
    game_state = analyzer.analyze_frame(frame)

    # Create visualization
    vis_frame = viz_manager.update_frame(
        frame,
        {
            "possession_triangle": {
                "position": game_state.possession_team,
                "confidence": game_state.confidence,
            },
            "territory_triangle": {
                "position": game_state.territory,
                "confidence": game_state.confidence,
            },
        },
        frame_number=game_state.frame_number,
    )

    # Add detailed analysis overlay
    info_text = [
        f"File: {Path(image_path).name}",
        f"Possession Team: {game_state.possession_team or 'Unknown'}",
        f"Territory: {game_state.territory or 'Unknown'}",
        f"Teams: {game_state.team_left or '?'} vs {game_state.team_right or '?'}",
        f"Score: {game_state.score_left or '0'} - {game_state.score_right or '0'}",
        f"Down: {game_state.down or '?'} Distance: {game_state.distance or '?'}",
        f"Quarter: {game_state.quarter or '?'} Time: {game_state.time or '?'}",
        f"Overall Confidence: {game_state.confidence:.2f}",
    ]

    y = 30
    for text in info_text:
        cv2.putText(vis_frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

    return {"game_state": game_state, "visualization": vis_frame}


def main():
    parser = argparse.ArgumentParser(description="Analyze game situations from screenshots")
    parser.add_argument("screenshot_dir", help="Directory containing screenshots")
    parser.add_argument(
        "--output", default="analysis_results", help="Output directory for visualizations"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Initialize hardware detection
    hardware = HardwareDetector()

    # Initialize game analyzer with optimization config
    optimization_config = OptimizationConfig(
        enable_dynamic_switching=True,
        enable_adaptive_batch_size=True,
        enable_performance_monitoring=True,
        enable_auto_optimization=True,
        max_inference_time=1.0 / 30.0,  # Target 30 FPS
        min_accuracy=0.85,
        max_memory_usage=0.8,
    )
    analyzer = EnhancedGameAnalyzer(hardware=hardware, optimization_config=optimization_config)

    # Initialize visualization
    viz_config = VisualizationConfig(
        mode=VisualizationMode.FULL,
        show_confidence=True,
        show_player_ids=False,
        line_thickness=2,
        font_scale=0.6,
    )
    viz_manager = VisualizationManager(config=viz_config)

    # Process each screenshot
    screenshot_dir = Path(args.screenshot_dir)
    screenshots = sorted(screenshot_dir.glob("_*.png"))

    logger.info(f"Found {len(screenshots)} screenshots to analyze")

    for screenshot in screenshots:
        logger.info(f"Analyzing {screenshot.name}")

        result = analyze_screenshot(str(screenshot), analyzer, viz_manager)
        if result is None:
            continue

        # Save visualization
        output_path = output_dir / f"analysis_{screenshot.name}"
        cv2.imwrite(str(output_path), result["visualization"])

        # Log analysis results
        game_state = result["game_state"]
        logger.info(f"Analysis results for {screenshot.name}:")
        logger.info(f"  Possession: {game_state.possession_team or 'Unknown'}")
        logger.info(f"  Territory: {game_state.territory or 'Unknown'}")
        logger.info(f"  Confidence: {game_state.confidence:.2f}")
        logger.info("---")


if __name__ == "__main__":
    main()

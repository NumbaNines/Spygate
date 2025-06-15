"""
Script to analyze Madden screenshots using SpygateAI visualization system.
"""

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.game_state import GameState
from spygate.ml.visualization_engine import DetectionVisualizer, VisualizationConfig


def analyze_screenshots(screenshots_dir: str, output_dir: str):
    """Analyze all screenshots in a directory and save visualizations."""

    # Initialize our systems
    analyzer = EnhancedGameAnalyzer()
    visualizer = DetectionVisualizer(
        VisualizationConfig(
            show_confidence=True,
            show_bounding_boxes=True,
            show_triangle_geometry=True,
            show_ocr_regions=True,
            show_hud_analysis=True,
            output_path=Path(output_dir),
        )
    )

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    screenshots = sorted(glob.glob(str(Path(screenshots_dir) / "*.png")))
    print(f"Found {len(screenshots)} screenshots to analyze")

    for idx, screenshot_path in enumerate(screenshots, 1):
        print(f"\nAnalyzing screenshot {idx}/{len(screenshots)}: {Path(screenshot_path).name}")

        # Read image
        frame = cv2.imread(screenshot_path)
        if frame is None:
            print(f"Failed to read image: {screenshot_path}")
            continue

        try:
            # Analyze frame
            game_state = analyzer.analyze_frame(frame)

            # Create visualization
            viz_frame = visualizer.create_visualization(frame, game_state)

            # Save visualization
            output_path = str(Path(output_dir) / f"analyzed_{Path(screenshot_path).name}")
            cv2.imwrite(output_path, viz_frame)
            print(f"Saved visualization to: {output_path}")

            # Print analysis results
            print("\nAnalysis Results:")
            print(f"Down & Distance: {game_state.down} & {game_state.distance}")
            print(f"Territory: {game_state.territory}")
            print(f"Possession: {game_state.possession_team}")
            print(f"Confidence: {game_state.confidence:.2f}")

        except Exception as e:
            print(f"Error analyzing {Path(screenshot_path).name}: {e}")
            continue

    print("\nAnalysis complete!")


if __name__ == "__main__":
    screenshots_dir = "Madden AYOUTUB"
    output_dir = "Madden AYOUTUB/analyzed"
    analyze_screenshots(screenshots_dir, output_dir)

"""
Test suite for game situation analysis with visualization support.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

from spygate.ml.game_state import GameState
from spygate.ml.visualization_engine import DetectionVisualizer, VisualizationConfig


class TestGameSituationAnalysis:
    """Test suite for analyzing game situations with visual validation."""

    @pytest.fixture
    def visualizer(self, tmp_path) -> DetectionVisualizer:
        """Create visualization engine instance."""
        config = VisualizationConfig(
            show_confidence=True,
            show_bounding_boxes=True,
            show_triangle_geometry=True,
            show_ocr_regions=True,
            show_hud_analysis=True,
            output_path=tmp_path / "visualizations",
        )
        return DetectionVisualizer(config)

    def test_possession_triangle_detection(self, visualizer, test_frame):
        """Test possession triangle detection with visualization."""
        # Run detection
        detections = self._run_yolo_detection(test_frame)
        triangle_data = self._analyze_triangles(test_frame, detections)
        ocr_results = self._run_ocr_analysis(test_frame, detections)
        game_state = self._extract_game_state(detections, triangle_data, ocr_results)

        # Create visualizations
        visualizations = visualizer.create_visualization_pipeline(
            frame=test_frame,
            detections=detections,
            triangle_data=triangle_data,
            ocr_results=ocr_results,
            game_state=game_state,
        )

        # Save visualizations for review
        visualizer.save_visualizations(visualizations, visualizer.config.output_path)

        # Optional: Display visualizations during test
        if pytest.config.getoption("--show-plot"):
            visualizer.display_visualizations(visualizations)

        # Validate results
        assert game_state.is_valid()
        assert game_state.confidence > 0.8

    def test_territory_triangle_detection(self, visualizer, test_frame):
        """Test territory triangle detection with visualization."""
        # Run detection
        detections = self._run_yolo_detection(test_frame)
        triangle_data = self._analyze_triangles(test_frame, detections)
        ocr_results = self._run_ocr_analysis(test_frame, detections)
        game_state = self._extract_game_state(detections, triangle_data, ocr_results)

        # Create visualizations
        visualizations = visualizer.create_visualization_pipeline(
            frame=test_frame,
            detections=detections,
            triangle_data=triangle_data,
            ocr_results=ocr_results,
            game_state=game_state,
        )

        # Save visualizations for review
        visualizer.save_visualizations(visualizations, visualizer.config.output_path)

        # Optional: Display visualizations during test
        if pytest.config.getoption("--show-plot"):
            visualizer.display_visualizations(visualizations)

        # Validate results
        assert game_state.is_valid()
        assert game_state.territory in ["OWN", "OPPONENT"]
        assert game_state.confidence > 0.8

    def test_hud_ocr_analysis(self, visualizer, test_frame):
        """Test HUD OCR analysis with visualization."""
        # Run detection
        detections = self._run_yolo_detection(test_frame)
        triangle_data = self._analyze_triangles(test_frame, detections)
        ocr_results = self._run_ocr_analysis(test_frame, detections)
        game_state = self._extract_game_state(detections, triangle_data, ocr_results)

        # Create visualizations
        visualizations = visualizer.create_visualization_pipeline(
            frame=test_frame,
            detections=detections,
            triangle_data=triangle_data,
            ocr_results=ocr_results,
            game_state=game_state,
        )

        # Save visualizations for review
        visualizer.save_visualizations(visualizations, visualizer.config.output_path)

        # Optional: Display visualizations during test
        if pytest.config.getoption("--show-plot"):
            visualizer.display_visualizations(visualizations)

        # Validate results
        assert game_state.is_valid()
        assert game_state.down in range(1, 5)
        assert game_state.distance is not None
        assert game_state.yard_line is not None
        assert game_state.confidence > 0.8

    def _run_yolo_detection(self, frame: np.ndarray) -> dict:
        """Run YOLOv8 detection on frame."""
        # TODO: Implement actual YOLO detection
        # Mock detection for now
        return {
            "hud": [(100, 100, 500, 150, 0.95)],
            "possession_triangle_area": [(50, 120, 80, 140, 0.92)],
            "territory_triangle_area": [(520, 120, 550, 140, 0.94)],
        }

    def _analyze_triangles(
        self, frame: np.ndarray, detections: dict
    ) -> list[tuple[np.ndarray, tuple[bool, float]]]:
        """Analyze detected triangles."""
        # TODO: Implement actual triangle analysis
        # Mock triangle data for now
        triangle_data = []
        for class_name in ["possession_triangle_area", "territory_triangle_area"]:
            if class_name in detections:
                for box in detections[class_name]:
                    x1, y1, x2, y2 = map(int, box[:4])
                    points = np.array([[[x1, y1]], [[x2, y1]], [[(x1 + x2) // 2, y2]]])
                    triangle_data.append((points, (True, 0.95)))
        return triangle_data

    def _run_ocr_analysis(self, frame: np.ndarray, detections: dict) -> dict[str, dict]:
        """Run OCR analysis on detected regions."""
        # TODO: Implement actual OCR analysis
        # Mock OCR results for now
        return {
            "down_distance": {"text": "1ST & 10", "confidence": 0.96, "bbox": (120, 110, 200, 140)},
            "yard_line": {"text": "OWN 25", "confidence": 0.94, "bbox": (220, 110, 300, 140)},
        }

    def _extract_game_state(
        self,
        detections: dict,
        triangle_data: list[tuple[np.ndarray, tuple[bool, float]]],
        ocr_results: dict[str, dict],
    ) -> GameState:
        """Extract game state from detection results."""
        # TODO: Implement actual game state extraction
        # Mock game state for now
        game_state = GameState()
        game_state.down = 1
        game_state.distance = 10
        game_state.yard_line = 25
        game_state.possession_team = "HOME"
        game_state.territory = "OWN"
        game_state.confidence = 0.95
        return game_state

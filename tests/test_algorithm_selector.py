"""Tests for the algorithm selector module."""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from spygate.utils.tracking_hardware import TrackingAlgorithm, TrackingMode
from spygate.video.algorithm_selector import (
    AlgorithmSelector,
    SceneComplexity,
    TrackingRequirements,
)


class TestAlgorithmSelector(unittest.TestCase):
    """Test cases for the AlgorithmSelector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

        # Create a mock frame
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns for complexity analysis
        cv2.rectangle(self.frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(self.frame, (400, 300), 50, (255, 255, 255), -1)

    def test_scene_complexity_analysis(self):
        """Test scene complexity analysis."""
        # Test low complexity scene
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        complexity = self.selector.analyze_scene_complexity(blank_frame)
        self.assertEqual(complexity, SceneComplexity.LOW)

        # Test medium complexity scene (default test frame)
        complexity = self.selector.analyze_scene_complexity(self.frame)
        self.assertEqual(complexity, SceneComplexity.MEDIUM)

        # Test high complexity scene
        complex_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        complexity = self.selector.analyze_scene_complexity(complex_frame)
        self.assertEqual(complexity, SceneComplexity.HIGH)

    def test_fps_tracking(self):
        """Test FPS tracking and drop calculation."""
        # Test initial FPS
        self.selector.update_fps(30.0)
        self.assertEqual(self.selector.base_fps, 30.0)
        self.assertEqual(self.selector.current_fps, 30.0)

        # Test FPS drop calculation
        self.selector.update_fps(25.0)
        self.selector.update_fps(20.0)
        fps_drop = self.selector.get_fps_drop()
        self.assertGreater(fps_drop, 0.0)
        self.assertLess(fps_drop, 1.0)

        # Test FPS history limit
        for _ in range(40):  # More than fps_history_max_size
            self.selector.update_fps(30.0)
        self.assertLessEqual(len(self.selector.fps_history), self.selector.fps_history_max_size)

    def test_algorithm_selection_basic_hardware(self):
        """Test algorithm selection with basic hardware."""
        with patch("spygate.utils.tracking_hardware.TrackingHardwareManager") as mock_manager:
            mock_manager.return_value.tracking_mode = TrackingMode.BASIC
            mock_manager.return_value.get_recommended_algorithm.return_value = TrackingAlgorithm.CSRT

            selector = AlgorithmSelector()
            algorithm = selector.select_algorithm(self.frame)
            self.assertEqual(algorithm, TrackingAlgorithm.CSRT)

    def test_algorithm_selection_professional_hardware(self):
        """Test algorithm selection with professional hardware."""
        with patch("spygate.utils.tracking_hardware.TrackingHardwareManager") as mock_manager:
            mock_manager.return_value.tracking_mode = TrackingMode.PROFESSIONAL
            mock_manager.return_value.get_recommended_algorithm.return_value = TrackingAlgorithm.DEEPSORT

            selector = AlgorithmSelector()
            algorithm = selector.select_algorithm(self.frame)
            self.assertEqual(algorithm, TrackingAlgorithm.DEEPSORT)

    def test_algorithm_selection_with_requirements(self):
        """Test algorithm selection with custom requirements."""
        requirements = TrackingRequirements(
            accuracy_weight=0.8,
            speed_weight=0.1,
            occlusion_weight=0.05,
            recovery_weight=0.05,
        )

        with patch("spygate.utils.tracking_hardware.TrackingHardwareManager") as mock_manager:
            mock_manager.return_value.get_recommended_algorithm.return_value = TrackingAlgorithm.DEEPSORT

            algorithm = self.selector.select_algorithm(
                frame=self.frame,
                requirements=requirements
            )
            self.assertEqual(algorithm, TrackingAlgorithm.DEEPSORT)

            # Verify weights were adjusted based on scene complexity
            mock_manager.return_value.get_recommended_algorithm.assert_called_once()
            call_args = mock_manager.return_value.get_recommended_algorithm.call_args[1]
            self.assertNotEqual(call_args["priority_accuracy"], requirements.accuracy_weight)
            self.assertNotEqual(call_args["priority_speed"], requirements.speed_weight)

    def test_algorithm_selection_with_fps_drop(self):
        """Test algorithm selection behavior with FPS drop."""
        # Simulate FPS drop
        self.selector.update_fps(30.0)  # Initial FPS
        for _ in range(10):
            self.selector.update_fps(15.0)  # 50% drop

        with patch("spygate.utils.tracking_hardware.TrackingHardwareManager") as mock_manager:
            mock_manager.return_value.get_recommended_algorithm.return_value = TrackingAlgorithm.MOSSE

            algorithm = self.selector.select_algorithm(self.frame)
            self.assertEqual(algorithm, TrackingAlgorithm.MOSSE)

            # Verify weights were adjusted to prioritize speed
            call_args = mock_manager.return_value.get_recommended_algorithm.call_args[1]
            self.assertGreater(
                call_args["priority_speed"],
                self.selector.current_requirements.speed_weight
            )

    def test_get_algorithm_stats(self):
        """Test retrieving algorithm statistics."""
        with patch("spygate.utils.tracking_hardware.TrackingHardwareManager") as mock_manager:
            # Mock hardware manager responses
            mock_manager.return_value.get_algorithm_requirements.return_value = {
                "min_mode": TrackingMode.ADVANCED,
                "gpu_accelerated": True,
                "accuracy": 0.9,
                "speed": 0.7,
                "occlusion_handling": 0.85,
                "recovery": 0.85,
            }
            mock_manager.return_value.get_mode_requirements.return_value = {
                "cpu_cores": 6,
                "ram_gb": 16,
                "gpu_required": True,
                "min_vram_gb": 6,
                "cuda_required": True,
            }

            stats = self.selector.get_algorithm_stats(TrackingAlgorithm.DEEPSORT)

            self.assertEqual(stats["algorithm"], TrackingAlgorithm.DEEPSORT)
            self.assertEqual(stats["min_mode"], TrackingMode.ADVANCED)
            self.assertTrue(stats["gpu_accelerated"])
            self.assertEqual(stats["performance_metrics"]["accuracy"], 0.9)
            self.assertEqual(stats["hardware_requirements"]["cpu_cores"], 6)
            self.assertEqual(stats["current_scene_complexity"], SceneComplexity.MEDIUM)

    def test_texture_analysis(self):
        """Test texture analysis function."""
        # Create a frame with varying texture
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Add some texture patterns
        frame[100:200, 100:200] = 255  # Solid block
        frame[300:400, 300:400] = np.random.randint(0, 255, (100, 100))  # Random noise

        score = self.selector._analyze_texture(frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0) 
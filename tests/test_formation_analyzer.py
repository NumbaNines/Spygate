"""Tests for the formation analyzer module."""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from spygate.core.game_detector import GameVersion
from spygate.video.formation_analyzer import FormationAnalyzer, FormationConfig, FormationType


class TestFormationAnalyzer(unittest.TestCase):
    """Test cases for the FormationAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = FormationConfig(
            min_players=11,
            clustering_eps=0.15,
            clustering_min_samples=2,
            confidence_threshold=0.7,
        )
        self.analyzer = FormationAnalyzer(config=self.config)

        # Create a mock frame
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.game_version = GameVersion.MADDEN_25

    def test_initialization(self):
        """Test formation analyzer initialization."""
        self.assertEqual(self.analyzer.config.min_players, 11)
        self.assertEqual(len(self.analyzer.formation_history), 0)
        self.assertIsNotNone(self.analyzer.offensive_templates)
        self.assertIsNotNone(self.analyzer.defensive_templates)

    def test_offensive_templates(self):
        """Test offensive formation templates."""
        templates = self.analyzer.offensive_templates
        self.assertIn(FormationType.SPREAD, templates)
        self.assertIn(FormationType.I_FORMATION, templates)

        # Check template structure
        spread = templates[FormationType.SPREAD]
        self.assertEqual(len(spread), 11)  # Should have 11 player positions
        self.assertTrue(all(len(pos) == 2 for pos in spread))  # Each position should be (x, y)

    def test_defensive_templates(self):
        """Test defensive formation templates."""
        templates = self.analyzer.defensive_templates
        self.assertIn(FormationType.FOUR_THREE, templates)
        self.assertIn(FormationType.THREE_FOUR, templates)

        # Check template structure
        four_three = templates[FormationType.FOUR_THREE]
        self.assertEqual(len(four_three), 11)  # Should have 11 player positions
        self.assertTrue(all(len(pos) == 2 for pos in four_three))  # Each position should be (x, y)

    @patch("spygate.video.player_detector.PlayerDetector.detect_players")
    def test_analyze_formation_not_enough_players(self, mock_detect):
        """Test formation analysis with insufficient players."""
        # Mock detection with only 5 players
        mock_detect.return_value = [
            {"bbox": np.array([100, 100, 150, 150]), "confidence": 0.9} for _ in range(5)
        ]

        result = self.analyzer.analyze_formation(self.frame, self.game_version, is_offense=True)

        self.assertIsNone(result["formation_type"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(len(result["player_positions"]), 5)
        self.assertEqual(len(result["clusters"]), 0)

    @patch("spygate.video.player_detector.PlayerDetector.detect_players")
    def test_analyze_formation_spread(self, mock_detect):
        """Test spread formation detection."""
        # Mock detection with spread formation positions
        spread_positions = self.analyzer.offensive_templates[FormationType.SPREAD]
        mock_detections = []

        for x, y in spread_positions:
            # Convert normalized coordinates to pixel coordinates
            px = int(x * 1280)
            py = int(y * 720)
            mock_detections.append(
                {"bbox": np.array([px - 25, py - 25, px + 25, py + 25]), "confidence": 0.9}
            )

        mock_detect.return_value = mock_detections

        result = self.analyzer.analyze_formation(self.frame, self.game_version, is_offense=True)

        self.assertEqual(result["formation_type"], FormationType.SPREAD)
        self.assertGreaterEqual(result["confidence"], self.config.confidence_threshold)
        self.assertEqual(len(result["player_positions"]), 11)

    @patch("spygate.video.player_detector.PlayerDetector.detect_players")
    def test_analyze_formation_four_three(self, mock_detect):
        """Test 4-3 defense formation detection."""
        # Mock detection with 4-3 formation positions
        four_three_positions = self.analyzer.defensive_templates[FormationType.FOUR_THREE]
        mock_detections = []

        for x, y in four_three_positions:
            # Convert normalized coordinates to pixel coordinates
            px = int(x * 1280)
            py = int(y * 720)
            mock_detections.append(
                {"bbox": np.array([px - 25, py - 25, px + 25, py + 25]), "confidence": 0.9}
            )

        mock_detect.return_value = mock_detections

        result = self.analyzer.analyze_formation(self.frame, self.game_version, is_offense=False)

        self.assertEqual(result["formation_type"], FormationType.FOUR_THREE)
        self.assertGreaterEqual(result["confidence"], self.config.confidence_threshold)
        self.assertEqual(len(result["player_positions"]), 11)

    def test_cluster_positions(self):
        """Test player position clustering."""
        # Create test positions with known clusters
        positions = [
            (0.2, 0.2),
            (0.21, 0.19),  # Cluster 1
            (0.5, 0.5),
            (0.51, 0.49),  # Cluster 2
            (0.8, 0.8),
            (0.79, 0.81),  # Cluster 3
        ]

        clusters = self.analyzer._cluster_positions(positions)

        self.assertEqual(len(clusters), 3)  # Should find 3 clusters

        # Check cluster centers are roughly correct
        expected_centers = [(0.205, 0.195), (0.505, 0.495), (0.795, 0.805)]
        for cluster, expected in zip(clusters, expected_centers):
            self.assertAlmostEqual(cluster[0], expected[0], places=2)
            self.assertAlmostEqual(cluster[1], expected[1], places=2)

    def test_formation_history(self):
        """Test formation history tracking."""
        # Add some test formations to history
        test_formations = [
            (FormationType.SPREAD, 0.9),
            (FormationType.SPREAD, 0.85),
            (FormationType.I_FORMATION, 0.8),
        ]

        for formation, confidence in test_formations:
            self.analyzer._update_formation_history(formation, confidence)

        stats = self.analyzer.get_formation_stats()

        self.assertEqual(stats["total_detections"], 3)
        self.assertAlmostEqual(stats["avg_confidence"], 0.85, places=2)
        self.assertEqual(len(stats["formation_types"]), 2)  # Should have SPREAD and I_FORMATION

    def test_game_specific_adjustments(self):
        """Test game-specific confidence adjustments."""
        base_confidence = 0.8
        formation_type = FormationType.SPREAD

        # Test Madden 25 adjustment (should increase confidence)
        madden_confidence = self.analyzer._adjust_confidence_for_game(
            base_confidence, formation_type, GameVersion.MADDEN_25
        )
        self.assertGreater(madden_confidence, base_confidence)

        # Test CFB 25 adjustment (should decrease confidence)
        cfb_confidence = self.analyzer._adjust_confidence_for_game(
            base_confidence, formation_type, GameVersion.CFB_25
        )
        self.assertLess(cfb_confidence, base_confidence)

    def test_template_similarity(self):
        """Test template similarity calculation."""
        # Create a slightly perturbed version of a template
        template = self.analyzer.offensive_templates[FormationType.SPREAD]
        test_positions = []

        for x, y in template:
            # Add small random perturbation
            dx = np.random.uniform(-0.05, 0.05)
            dy = np.random.uniform(-0.05, 0.05)
            test_positions.append((x + dx, y + dy))

        similarity = self.analyzer._calculate_template_similarity(test_positions, template)

        # Should have high similarity despite perturbations
        self.assertGreaterEqual(similarity, 0.7)


if __name__ == "__main__":
    unittest.main()

"""
Tests for the Enhanced Game Detection Pipeline

This test suite validates the enhanced game detection system including:
- Multi-method detection (template, HUD analysis, ML)
- Caching and performance optimization
- Cross-game compatibility and universal data structures
- Stability and confidence management
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from spygate.core.game_detector import (
    DetectionResult,
    GameDetectionError,
    GameDetector,
    GameProfile,
    GameVersion,
    InvalidFrameError,
    UniversalGameData,
    UnsupportedGameError,
)


@pytest.fixture
def enhanced_detector():
    """Create an enhanced GameDetector instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        detector = GameDetector(cache_dir=temp_dir)
        yield detector


@pytest.fixture
def mock_frame():
    """Create a realistic mock frame for testing."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add mock HUD elements
    # Score bug region
    frame[50:100, 50:300] = 128  # Gray score bug

    # Add some text-like patterns
    frame[60:80, 60:120] = 255  # White text area
    frame[60:80, 130:190] = 255  # More white text

    return frame


class TestEnhancedGameDetection:
    """Test enhanced game detection features."""

    def test_enhanced_detection_with_caching(self, enhanced_detector, mock_frame):
        """Test that enhanced detection works and uses caching."""
        # First detection
        result1 = enhanced_detector.detect_game(mock_frame)

        assert isinstance(result1, DetectionResult)
        assert result1.version in [GameVersion.MADDEN_25, GameVersion.CFB_25]
        assert 0.0 <= result1.confidence <= 1.0
        assert result1.detection_method is not None

        # Second detection should hit cache
        result2 = enhanced_detector.detect_game(mock_frame)

        assert result1.version == result2.version
        assert result1.confidence == result2.confidence

        # Check that cache was used
        stats = enhanced_detector.get_performance_stats()
        assert stats["cache_hits"] > 0

    def test_cross_game_conversion(self, enhanced_detector):
        """Test conversion between game versions."""
        # Create universal data from Madden
        madden_data = {"formation": "Shotgun Trips TE", "strategy": "Cover 2 Man"}

        universal_data = enhanced_detector.create_universal_data(madden_data, GameVersion.MADDEN_25)

        # Convert to CFB
        cfb_data = enhanced_detector.convert_to_target_game(universal_data, GameVersion.CFB_25)

        assert cfb_data["formation"] == "Spread Trips Right"
        assert cfb_data["strategy"] == "Cover 2 Match"

    def test_performance_stats(self, enhanced_detector, mock_frame):
        """Test performance statistics tracking."""
        # Initial stats
        stats = enhanced_detector.get_performance_stats()
        assert stats["total_detections"] == 0
        assert stats["cache_hit_rate"] == 0.0

        # After detection
        enhanced_detector.detect_game(mock_frame)
        enhanced_detector.detect_game(mock_frame)  # Should hit cache

        stats = enhanced_detector.get_performance_stats()
        assert stats["total_detections"] == 2
        assert stats["cache_hit_rate"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])

"""Tests for the GameManager class."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from spygate.core.game_detector import GameVersion
from spygate.core.game_manager import GameManager, GameSettings
from spygate.core.hardware import PerformanceTier


@pytest.fixture
def game_manager():
    """Create a GameManager instance for testing."""
    return GameManager()


@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_game_manager_initialization(game_manager):
    """Test GameManager initialization."""
    assert game_manager.current_settings is None
    assert game_manager.current_game is None
    assert game_manager.hardware is not None
    assert game_manager.game_detector is not None


def test_performance_profiles_initialization(game_manager):
    """Test performance profiles initialization."""
    profiles = game_manager._performance_profiles
    
    # Check if profiles exist for all combinations
    for game in GameVersion:
        for tier in PerformanceTier:
            profile = profiles.get((game, tier))
            assert profile is not None
            assert isinstance(profile, GameSettings)
            
            # Verify profile attributes
            assert isinstance(profile.resolution, tuple)
            assert len(profile.resolution) == 2
            assert isinstance(profile.frame_rate, int)
            assert isinstance(profile.analysis_features, list)
            assert isinstance(profile.performance_mode, str)
            assert isinstance(profile.gpu_enabled, bool)


@patch("spygate.core.hardware.HardwareDetector.get_performance_tier")
@patch("spygate.core.game_detector.GameDetector.detect_game_version")
def test_detect_and_configure(mock_detect, mock_tier, game_manager, mock_frame):
    """Test game detection and configuration."""
    # Mock return values
    mock_detect.return_value = (GameVersion.MADDEN_25, 0.95)
    mock_tier.return_value = PerformanceTier.PREMIUM
    
    # Test detection and configuration
    version, settings = game_manager.detect_and_configure(mock_frame)
    
    assert version == GameVersion.MADDEN_25
    assert isinstance(settings, GameSettings)
    assert settings == game_manager.current_settings
    
    # Verify settings match the performance profile
    profile = game_manager._performance_profiles[(GameVersion.MADDEN_25, PerformanceTier.PREMIUM)]
    assert settings.resolution == profile.resolution
    assert settings.frame_rate == profile.frame_rate
    assert settings.performance_mode == profile.performance_mode


@patch("spygate.core.hardware.HardwareDetector.has_gpu_support")
def test_get_optimal_settings(mock_gpu, game_manager):
    """Test optimal settings generation."""
    # Test with GPU support
    mock_gpu.return_value = True
    settings = game_manager.get_optimal_settings(
        GameVersion.MADDEN_25, PerformanceTier.PREMIUM
    )
    assert settings.gpu_enabled is True
    
    # Test without GPU support
    mock_gpu.return_value = False
    settings = game_manager.get_optimal_settings(
        GameVersion.MADDEN_25, PerformanceTier.PREMIUM
    )
    assert settings.gpu_enabled is False


def test_get_interface_mapping(game_manager):
    """Test interface mapping retrieval."""
    # Should raise error when no game is detected
    with pytest.raises(ValueError):
        game_manager.get_interface_mapping()
    
    # Set a current game and test again
    game_manager.game_detector._current_game = GameVersion.MADDEN_25
    mapping = game_manager.get_interface_mapping()
    assert mapping.version == GameVersion.MADDEN_25


@patch("spygate.core.hardware.HardwareDetector.get_performance_tier")
def test_update_settings(mock_tier, game_manager, mock_frame):
    """Test settings update functionality."""
    mock_tier.return_value = PerformanceTier.PREMIUM
    
    # Test update with new frame
    with patch("spygate.core.game_detector.GameDetector.detect_game_version") as mock_detect:
        mock_detect.return_value = (GameVersion.MADDEN_25, 0.95)
        settings = game_manager.update_settings(mock_frame)
        assert isinstance(settings, GameSettings)
        assert settings == game_manager.current_settings
    
    # Test update without frame (should raise error if no game detected)
    with pytest.raises(ValueError):
        game_manager.update_settings()
    
    # Test update without frame but with current game
    game_manager.game_detector._current_game = GameVersion.CFB_25
    settings = game_manager.update_settings()
    assert isinstance(settings, GameSettings)
    assert settings == game_manager.current_settings


def test_settings_validation(game_manager):
    """Test settings validation and constraints."""
    # Test minimum tier settings
    settings = game_manager.get_optimal_settings(
        GameVersion.MADDEN_25, PerformanceTier.MINIMUM
    )
    assert settings.resolution == (1280, 720)
    assert settings.frame_rate == 30
    assert len(settings.analysis_features) >= 2  # Should have at least basic features
    assert settings.performance_mode == "performance"
    
    # Test professional tier settings
    settings = game_manager.get_optimal_settings(
        GameVersion.MADDEN_25, PerformanceTier.PROFESSIONAL
    )
    assert settings.resolution == (2560, 1440)
    assert settings.frame_rate == 60
    assert len(settings.analysis_features) >= 5  # Should have all features
    assert settings.performance_mode == "quality"


def test_feature_support_validation(game_manager):
    """Test feature support validation across games and tiers."""
    for game in GameVersion:
        for tier in PerformanceTier:
            settings = game_manager.get_optimal_settings(game, tier)
            
            # Verify all features in settings are supported by the game
            for feature in settings.analysis_features:
                assert game_manager.game_detector.is_feature_supported(feature, game)
            
            # Verify minimum features for each tier
            if tier == PerformanceTier.MINIMUM:
                assert "hud_analysis" in settings.analysis_features
                assert "situation_detection" in settings.analysis_features
            elif tier == PerformanceTier.PROFESSIONAL:
                assert "advanced_analytics" in settings.analysis_features
                assert "player_tracking" in settings.analysis_features 
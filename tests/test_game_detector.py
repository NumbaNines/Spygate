"""Tests for the GameDetector class."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch

from spygate.core.game_detector import (
    GameDetector,
    GameVersion,
    GameDetectionError,
    InvalidFrameError,
    UnsupportedGameError
)


@pytest.fixture
def game_detector():
    """Create a GameDetector instance for testing."""
    return GameDetector()


def create_mock_frame(game_version: GameVersion) -> np.ndarray:
    """
    Create a realistic mock frame for testing.
    
    Args:
        game_version: Which game version to simulate
        
    Returns:
        numpy array representing a frame
    """
    # Create a black background
    width, height = 1920, 1080
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Add score bug based on game version
    if game_version == GameVersion.MADDEN_25:
        # Madden 25 style score bug
        draw.rectangle([50, 50, 300, 100], fill='gray')
        draw.text((60, 60), "21", fill='white')  # Score
        draw.text((120, 60), "2:30", fill='white')  # Time
        draw.text((180, 60), "1st", fill='white')  # Down
        draw.text((240, 60), "10", fill='white')  # Distance
    else:
        # CFB 25 style score bug
        draw.rectangle([40, 40, 280, 90], fill='darkgray')
        draw.text((48, 48), "14", fill='white')  # Score
        draw.text((108, 48), "4:15", fill='white')  # Time
        draw.text((168, 48), "3rd", fill='white')  # Down
        draw.text((228, 48), "5", fill='white')  # Distance
    
    return np.array(image)


def create_mock_play_art(game_version: GameVersion) -> np.ndarray:
    """
    Create a realistic mock play art image for testing.
    
    Args:
        game_version: Which game version to simulate
        
    Returns:
        numpy array representing play art
    """
    # Create a black background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    
    # Draw player markers
    marker_positions = [
        (400, 500),  # QB
        (350, 500), (450, 500),  # RB/FB
        (200, 500), (300, 500), (500, 500), (600, 500),  # WR/TE
        (250, 450), (350, 450), (450, 450), (550, 450),  # OL
    ]
    
    for x, y in marker_positions:
        if game_version == GameVersion.MADDEN_25:
            # Madden uses circles
            draw.ellipse([x-10, y-10, x+10, y+10], outline='white')
        else:
            # CFB uses triangles
            draw.polygon([x, y-10, x-10, y+10, x+10, y+10], outline='white')
    
    # Draw route lines
    if game_version == GameVersion.MADDEN_25:
        # Madden style routes
        draw.line([200, 500, 200, 400, 300, 400], fill='white', width=2)  # Hook route
        draw.line([600, 500, 600, 300], fill='white', width=2)  # Vertical route
    else:
        # CFB style routes
        draw.line([200, 500, 200, 350, 350, 350], fill='white', width=2)  # In route
        draw.line([600, 500, 600, 350, 450, 350], fill='white', width=2)  # Post route
    
    return np.array(image)


@pytest.mark.parametrize("game_version", [
    GameVersion.MADDEN_25,
    GameVersion.CFB_25
])
def test_game_detection(game_detector, game_version):
    """Test game version detection with realistic mock frames."""
    frame = create_mock_frame(game_version)
    detected_version = game_detector.detect_game(frame)
    assert detected_version == game_version


@pytest.mark.parametrize("element_type,content", [
    ("score", "21"),
    ("time", "2:30"),
    ("down", "1st"),
    ("distance", "10")
])
def test_element_characteristics(game_detector, element_type, content):
    """Test HUD element characteristic detection."""
    # Create test image
    width, height = 100, 40
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), content, fill='white')
    
    confidence = game_detector._check_element_characteristics(image, element_type)
    assert confidence > 0.8, f"Low confidence ({confidence}) for {element_type} element"


@pytest.mark.parametrize("game_version", [
    GameVersion.MADDEN_25,
    GameVersion.CFB_25
])
def test_play_art_detection(game_detector, game_version):
    """Test play art detection with realistic mock play art."""
    play_art = create_mock_play_art(game_version)
    play_art_image = Image.fromarray(play_art)
    
    confidence = game_detector._check_play_art_characteristics(play_art_image)
    assert confidence > 0.8, f"Low confidence ({confidence}) for {game_version} play art"


def test_invalid_frame_handling(game_detector):
    """Test handling of invalid frames."""
    with pytest.raises(InvalidFrameError):
        game_detector.detect_game(None)
    
    with pytest.raises(InvalidFrameError):
        game_detector.detect_game(np.array([]))


def test_detection_confidence_threshold(game_detector):
    """Test confidence threshold for game detection."""
    # Create a frame with ambiguous characteristics
    width, height = 1920, 1080
    ambiguous_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    with pytest.raises(UnsupportedGameError):
        game_detector.detect_game(ambiguous_frame)


def test_frame_buffer_stability(game_detector):
    """Test frame buffer for stable game detection."""
    madden_frame = create_mock_frame(GameVersion.MADDEN_25)
    cfb_frame = create_mock_frame(GameVersion.CFB_25)
    
    # Feed alternating frames
    frames = [madden_frame, cfb_frame] * 3
    versions = []
    
    for frame in frames:
        try:
            version = game_detector.detect_game(frame)
            versions.append(version)
        except GameDetectionError:
            pass
    
    # Check that detection remains stable
    assert len(set(versions[-3:])) == 1, "Game detection not stable over multiple frames"


def test_detection_stability(game_detector):
    """Test that game detection remains stable across multiple frames."""
    game_version = GameVersion.MADDEN_25
    frame = create_mock_frame(game_version)
    
    # Process multiple frames
    detections = []
    for _ in range(10):
        detected = game_detector.detect_game(frame)
        detections.append(detected)
    
    # All detections should be the same
    assert all(d == game_version for d in detections)


def test_game_switching(game_detector):
    """Test detection when switching between games."""
    # Start with Madden
    madden_frame = create_mock_frame(GameVersion.MADDEN_25)
    detected = game_detector.detect_game(madden_frame)
    assert detected == GameVersion.MADDEN_25
    
    # Switch to CFB
    cfb_frame = create_mock_frame(GameVersion.CFB_25)
    detected = game_detector.detect_game(cfb_frame)
    assert detected == GameVersion.CFB_25


def test_unsupported_game(game_detector):
    """Test handling of unrecognized game versions."""
    # Create a completely black frame that won't match any game
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    with pytest.raises(UnsupportedGameError):
        game_detector.detect_game(frame)


def test_logging(game_detector):
    """Test that logging is properly configured."""
    with patch('logging.getLogger') as mock_logger:
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        
        # Create a new detector to trigger logging setup
        GameDetector()
        
        # Verify logger was configured
        assert mock_logger.called
        assert mock_logger_instance.setLevel.called
        assert len(mock_logger_instance.handlers) > 0 
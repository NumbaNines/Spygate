import pytest
from unittest.mock import Mock, patch
from PyQt6.QtMultimedia import QMediaPlayer, QMediaMetaData
from spygate.gui.video.timeline_manager import TimelineManager

@pytest.fixture
def media_player():
    """Create a mock QMediaPlayer."""
    player = Mock(spec=QMediaPlayer)
    player.duration.return_value = 10000  # 10 seconds
    player.position.return_value = 0
    player.playbackState.return_value = QMediaPlayer.PlaybackState.StoppedState
    return player

@pytest.fixture
def timeline_manager(media_player):
    """Create a TimelineManager instance with mock media player."""
    return TimelineManager(media_player)

def test_initialization(timeline_manager, media_player):
    """Test TimelineManager initialization."""
    assert timeline_manager._media_player == media_player
    assert timeline_manager._frame_rate == 0
    assert timeline_manager._current_frame == 0
    assert timeline_manager._total_frames == 0
    assert timeline_manager._last_position == 0

def test_handle_metadata_changed(timeline_manager, media_player):
    """Test frame rate detection from metadata."""
    # Mock metadata with frame rate
    metadata = Mock()
    metadata.value.return_value = 30.0  # 30 fps
    media_player.metaData.return_value = metadata
    media_player.duration.return_value = 5000  # 5 seconds
    
    # Trigger metadata changed
    timeline_manager._handle_metadata_changed()
    
    assert timeline_manager._frame_rate == 30.0
    assert timeline_manager._total_frames == 150  # 5 seconds * 30 fps

def test_handle_position_changed(timeline_manager):
    """Test position change handling and frame calculation."""
    timeline_manager._frame_rate = 30.0  # Set frame rate
    
    # Test position change
    timeline_manager._handle_position_changed(1000)  # 1 second
    
    assert timeline_manager._last_position == 1000
    assert timeline_manager._current_frame == 30  # 1 second * 30 fps

def test_next_frame(timeline_manager, media_player):
    """Test next frame navigation."""
    timeline_manager._frame_rate = 30.0
    timeline_manager._current_frame = 30
    timeline_manager._total_frames = 150
    
    timeline_manager.next_frame()
    
    media_player.pause.assert_called_once()
    media_player.setPosition.assert_called_once_with(1033)  # (31 frames / 30 fps) * 1000

def test_previous_frame(timeline_manager, media_player):
    """Test previous frame navigation."""
    timeline_manager._frame_rate = 30.0
    timeline_manager._current_frame = 30
    timeline_manager._total_frames = 150
    
    timeline_manager.previous_frame()
    
    media_player.pause.assert_called_once()
    media_player.setPosition.assert_called_once_with(967)  # (29 frames / 30 fps) * 1000

def test_seek_to_frame(timeline_manager, media_player):
    """Test seeking to specific frame."""
    timeline_manager._frame_rate = 30.0
    timeline_manager._total_frames = 150
    
    timeline_manager.seek_to_frame(45)  # Seek to frame 45
    
    media_player.setPosition.assert_called_once_with(1500)  # (45 frames / 30 fps) * 1000

def test_seek_to_position(timeline_manager, media_player):
    """Test seeking to specific position."""
    media_player.duration.return_value = 10000
    
    timeline_manager.seek_to_position(5000)  # Seek to 5 seconds
    
    media_player.setPosition.assert_called_once_with(5000)

def test_property_getters(timeline_manager, media_player):
    """Test property getter methods."""
    timeline_manager._frame_rate = 30.0
    timeline_manager._current_frame = 45
    timeline_manager._total_frames = 150
    media_player.duration.return_value = 5000
    media_player.position.return_value = 1500
    media_player.playbackState.return_value = QMediaPlayer.PlaybackState.PlayingState
    
    assert timeline_manager.frame_rate == 30.0
    assert timeline_manager.current_frame == 45
    assert timeline_manager.total_frames == 150
    assert timeline_manager.duration == 5000
    assert timeline_manager.position == 1500
    assert timeline_manager.is_playing is True

def test_frame_boundaries(timeline_manager, media_player):
    """Test frame navigation at boundaries."""
    timeline_manager._frame_rate = 30.0
    timeline_manager._total_frames = 150
    
    # Test at start
    timeline_manager._current_frame = 0
    timeline_manager.previous_frame()
    media_player.setPosition.assert_not_called()
    
    # Test at end
    timeline_manager._current_frame = 149
    media_player.reset_mock()
    timeline_manager.next_frame()
    media_player.setPosition.assert_not_called() 
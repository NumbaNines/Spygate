"""Video player component for the VideoTimeline."""

from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class VideoPlayer(QWidget):
    """Video player component using QMediaPlayer."""

    # Signals
    positionChanged = pyqtSignal(int)  # Current position in milliseconds
    durationChanged = pyqtSignal(int)  # Video duration in milliseconds
    playbackStateChanged = pyqtSignal(QMediaPlayer.PlaybackState)  # Playback state changes

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the video player.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setup_ui()

        # Connect signals
        self.media_player.positionChanged.connect(self.positionChanged.emit)
        self.media_player.durationChanged.connect(self.durationChanged.emit)
        self.media_player.playbackStateChanged.connect(self.playbackStateChanged.emit)

        # Set accessibility
        self.setAccessibleName("Video Player")
        self.setAccessibleDescription("Video playback component")

    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        layout.addWidget(self.video_widget)

        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(self.audio_output)

    def load_video(self, video_path: str):
        """Load a video file.

        Args:
            video_path: Path to the video file
        """
        self.media_player.setSource(QUrl.fromLocalFile(video_path))

    def play(self):
        """Start video playback."""
        self.media_player.play()

    def pause(self):
        """Pause video playback."""
        self.media_player.pause()

    def stop(self):
        """Stop video playback."""
        self.media_player.stop()

    def seek(self, position: int):
        """Seek to a specific position in milliseconds.

        Args:
            position: Position in milliseconds
        """
        self.media_player.setPosition(position)

    def duration(self) -> int:
        """Get video duration in milliseconds.

        Returns:
            Duration in milliseconds
        """
        return self.media_player.duration()

    def position(self) -> int:
        """Get current position in milliseconds.

        Returns:
            Current position in milliseconds
        """
        return self.media_player.position()

    def set_volume(self, volume: float):
        """Set audio volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.audio_output.setVolume(volume)

    def set_playback_rate(self, rate: float):
        """Set playback speed rate.

        Args:
            rate: Playback speed multiplier (e.g., 0.5 for half speed, 2.0 for double speed)
        """
        self.media_player.setPlaybackRate(rate)

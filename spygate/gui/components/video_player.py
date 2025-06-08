"""
Spygate - Video Player Component
"""

import vlc
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .video_timeline import VideoTimeline


class VideoPlayer(QWidget):
    """Video player component using VLC backend."""

    # Custom signals
    frameChanged = pyqtSignal(int)  # Emits current frame number
    videoLoaded = pyqtSignal(str, int, int)  # Emits path, total frames, fps

    def __init__(self):
        """Initialize the video player."""
        super().__init__()

        # Create VLC instance and media player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

        # Initialize variables
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.playback_speed = 1.0
        self.current_path = None

        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create video frame
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet(
            """
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #3a3a3a;
            }
        """
        )
        layout.addWidget(self.video_frame)

        # Create timeline
        self.timeline = VideoTimeline()
        self.timeline.positionChanged.connect(self.on_timeline_position_changed)
        self.timeline.frameRateChanged.connect(self.on_frame_rate_changed)
        self.timeline.playbackSpeedChanged.connect(self.on_playback_speed_changed)
        layout.addWidget(self.timeline)

        # Create controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # Create playback controls
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.clicked.connect(self.toggle_playback)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.clicked.connect(self.stop)

        # Add controls to layout
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Create update timer
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 fps update rate
        self.timer.timeout.connect(self.update_ui)

        # Set up video frame
        if hasattr(self.video_frame, "winId"):
            self.player.set_hwnd(self.video_frame.winId())

    def load_video(self, path):
        """Load a video file."""
        self.current_path = path
        media = self.instance.media_new(path)
        self.player.set_media(media)

        # Get video information
        media.parse()
        self.fps = media.get_fps() or 30
        duration = media.get_duration()
        if duration > 0:
            self.total_frames = int(self.fps * (duration / 1000.0))
        else:
            self.total_frames = 0

        # Update timeline
        self.timeline.setVideoInfo(self.total_frames, self.fps, path)

        # Emit video loaded signal
        self.videoLoaded.emit(path, self.total_frames, self.fps)

        self.update_ui()

    def play(self):
        """Start video playback."""
        self.player.play()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.timer.start()

    def pause(self):
        """Pause video playback."""
        self.player.pause()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.timer.stop()

    def stop(self):
        """Stop video playback."""
        self.player.stop()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.timer.stop()
        self.current_frame = 0
        self.timeline.setCurrentFrame(0)

    def toggle_playback(self):
        """Toggle between play and pause."""
        if self.player.is_playing():
            self.pause()
        else:
            self.play()

    def on_timeline_position_changed(self, position):
        """Handle timeline position changes."""
        self.player.set_position(position)
        self.current_frame = int(position * self.total_frames)
        self.frameChanged.emit(self.current_frame)

    def on_frame_rate_changed(self, fps):
        """Handle frame rate changes."""
        self.fps = fps
        if self.timer.isActive():
            self.timer.setInterval(int(1000 / fps))

    def on_playback_speed_changed(self, speed):
        """Handle playback speed changes."""
        self.playback_speed = speed
        self.player.set_rate(speed)

    def update_ui(self):
        """Update the UI with current playback status."""
        if not self.player.get_media():
            return

        # Get current position and update frame count
        position = self.player.get_position()
        if position >= 0:
            self.current_frame = int(position * self.total_frames)
            self.timeline.setCurrentFrame(self.current_frame)
            self.frameChanged.emit(self.current_frame)

    def add_annotation(self, annotation):
        """Add an annotation to the timeline."""
        self.timeline.addAnnotation(annotation)

    def add_marker(self, frame, marker_type):
        """Add a marker to the timeline."""
        self.timeline.addMarker(frame, marker_type)

    def resizeEvent(self, event):
        """Handle resize events to maintain video aspect ratio."""
        super().resizeEvent(event)
        if self.player:
            self.player.video_set_scale(0)  # Auto scale

    def cleanup(self):
        """Clean up resources before closing."""
        if self.timer.isActive():
            self.timer.stop()
        if self.player:
            self.player.stop()
            self.player.release()
        if self.instance:
            self.instance.release()
        self.timeline.cleanup()

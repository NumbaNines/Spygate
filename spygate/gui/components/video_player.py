"""
Spygate - Video Player Component
"""

import vlc
from PyQt6.QtCore import Qt, QTimer
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


class VideoPlayer(QWidget):
    """Video player component using VLC backend."""

    def __init__(self):
        """Initialize the video player."""
        super().__init__()

        # Create VLC instance and media player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

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

        # Create time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.sliderMoved.connect(self.set_position)

        # Create time label
        self.time_label = QLabel("00:00 / 00:00")

        # Add controls to layout
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.time_slider)
        controls_layout.addWidget(self.time_label)

        layout.addLayout(controls_layout)

        # Create update timer
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui)

        # Set up video frame
        if hasattr(self.video_frame, "winId"):
            self.player.set_hwnd(self.video_frame.winId())

    def load_video(self, path):
        """Load a video file."""
        media = self.instance.media_new(path)
        self.player.set_media(media)
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
        self.time_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")

    def toggle_playback(self):
        """Toggle between play and pause."""
        if self.player.is_playing():
            self.pause()
        else:
            self.play()

    def set_position(self, position):
        """Set the video position."""
        self.player.set_position(position / 1000.0)

    def update_ui(self):
        """Update the UI with current playback status."""
        media = self.player.get_media()
        if not media:
            return

        # Update time slider
        length = self.player.get_length()
        if length > 0:
            position = int(self.player.get_position() * 1000)
            self.time_slider.setValue(position)

            # Update time label
            current = self.player.get_time() // 1000
            total = length // 1000
            self.time_label.setText(
                f"{current//60:02d}:{current%60:02d} / {total//60:02d}:{total%60:02d}"
            )

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

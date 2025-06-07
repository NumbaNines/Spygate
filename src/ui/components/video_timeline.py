"""VideoTimeline component for video playback and analysis."""

from typing import Dict, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QShortcut
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..models.annotation import Annotation
from .annotation_display import AnnotationDisplay
from .annotation_tool import AnnotationTool
from .video_player import VideoPlayer


class VideoTimeline(QWidget):
    """Custom timeline component for video playback and analysis."""

    # Signals
    frameChanged = pyqtSignal(int)  # Current frame number
    annotationAdded = pyqtSignal(dict)  # Annotation data
    clipExported = pyqtSignal(str)  # Export path

    # Speed options
    SPEED_OPTIONS = {
        "0.25x": 0.25,
        "0.5x": 0.5,
        "0.75x": 0.75,
        "1x": 1.0,
        "1.25x": 1.25,
        "1.5x": 1.5,
        "2x": 2.0,
    }

    def __init__(
        self, video_path: str, player_name: str, parent: Optional[QWidget] = None
    ):
        """Initialize the VideoTimeline component.

        Args:
            video_path: Path to the video file
            player_name: Name of the player ("Self" or "Opponent: Name")
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.video_path = video_path
        self.player_name = player_name
        self.annotations: Dict[str, Annotation] = {}
        self.current_annotation: Optional[Annotation] = None
        self.setup_ui()
        self.setup_shortcuts()

        # Set accessibility
        self.setAccessibleName("Video Timeline")
        self.setAccessibleDescription(
            f"Video timeline for {player_name}'s gameplay analysis. "
            "Use space to play/pause, arrow keys to navigate, "
            "and Ctrl+[ or Ctrl+] to adjust speed."
        )

        # Connect video player signals
        self.video_player.positionChanged.connect(self.on_position_changed)
        self.video_player.durationChanged.connect(self.on_duration_changed)
        self.video_player.playbackStateChanged.connect(self.on_state_changed)

        # Load the video
        self.video_player.load_video(self.video_path)

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout()

        # Video player
        self.video_player = VideoPlayer(self)
        self.video_player.setMinimumHeight(360)
        layout.addWidget(self.video_player)

        # Timeline controls
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setAccessibleName("Play/Pause")
        self.play_button.setAccessibleDescription("Toggle video playback (Space)")
        self.play_button.setToolTip("Play/Pause (Space)")
        controls_layout.addWidget(self.play_button)

        # Seek backward button
        back_button = QPushButton()
        back_button.setIcon(QIcon.fromTheme("media-seek-backward"))
        back_button.clicked.connect(lambda: self.seek_relative(-5))
        back_button.setAccessibleName("Seek backward")
        back_button.setAccessibleDescription("Skip back 5 seconds (Left arrow)")
        back_button.setToolTip("Skip back 5 seconds (Left arrow)")
        controls_layout.addWidget(back_button)

        # Seek forward button
        forward_button = QPushButton()
        forward_button.setIcon(QIcon.fromTheme("media-seek-forward"))
        forward_button.clicked.connect(lambda: self.seek_relative(5))
        forward_button.setAccessibleName("Seek forward")
        forward_button.setAccessibleDescription("Skip forward 5 seconds (Right arrow)")
        forward_button.setToolTip("Skip forward 5 seconds (Right arrow)")
        controls_layout.addWidget(forward_button)

        # Speed control
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(list(self.SPEED_OPTIONS.keys()))
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.set_playback_speed)
        self.speed_combo.setAccessibleName("Playback speed")
        self.speed_combo.setAccessibleDescription(
            "Change video playback speed. "
            "Use Ctrl+[ to decrease or Ctrl+] to increase speed."
        )
        self.speed_combo.setToolTip(
            "Change playback speed (Ctrl+[ to decrease, Ctrl+] to increase)"
        )
        controls_layout.addWidget(self.speed_combo)

        # Time display
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAccessibleName("Time display")
        self.time_label.setAccessibleDescription(
            "Current video position and total duration"
        )
        controls_layout.addWidget(self.time_label)

        # Add annotation button
        add_annotation_button = QPushButton("Add Annotation")
        add_annotation_button.clicked.connect(self.show_annotation_tool)
        add_annotation_button.setAccessibleName("Add annotation")
        add_annotation_button.setAccessibleDescription(
            "Create a new annotation at current time (Ctrl+A)"
        )
        add_annotation_button.setToolTip("Add annotation at current time (Ctrl+A)")
        controls_layout.addWidget(add_annotation_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Timeline slider and annotation display
        timeline_layout = QVBoxLayout()

        # Annotation display
        self.annotation_display = AnnotationDisplay(
            video_duration=self.video_player.duration(), parent=self
        )
        self.annotation_display.annotationSelected.connect(self.seek_to_annotation)
        self.annotation_display.annotationDeleted.connect(self.remove_annotation)
        self.annotation_display.annotationEdited.connect(self.edit_annotation)
        timeline_layout.addWidget(self.annotation_display)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setAccessibleName("Timeline slider")
        self.timeline_slider.setAccessibleDescription(
            "Video timeline slider. "
            "Use left/right arrow keys to seek, "
            "up/down arrow keys to seek in larger increments."
        )
        self.timeline_slider.valueChanged.connect(self.seek_to_position)
        timeline_layout.addWidget(self.timeline_slider)

        layout.addLayout(timeline_layout)
        self.setLayout(layout)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Speed control shortcuts
        QShortcut(QKeySequence("Ctrl+["), self, self.decrease_speed)
        QShortcut(QKeySequence("Ctrl+]"), self, self.increase_speed)

        # Playback shortcuts
        QShortcut(QKeySequence("Space"), self, self.toggle_playback)
        QShortcut(QKeySequence("Left"), self, lambda: self.seek_relative(-5))
        QShortcut(QKeySequence("Right"), self, lambda: self.seek_relative(5))
        QShortcut(QKeySequence("Up"), self, lambda: self.seek_relative(-30))
        QShortcut(QKeySequence("Down"), self, lambda: self.seek_relative(30))
        QShortcut(QKeySequence("Ctrl+A"), self, self.show_annotation_tool)

    def set_playback_speed(self, speed_text: str):
        """Set the video playback speed.

        Args:
            speed_text: Speed text from combo box (e.g., "1x", "2x")
        """
        speed = self.SPEED_OPTIONS.get(speed_text, 1.0)
        self.video_player.media_player.setPlaybackRate(speed)
        # Announce speed change for screen readers
        self.setAccessibleDescription(f"Playback speed changed to {speed_text}")

    def decrease_speed(self):
        """Decrease playback speed by one step."""
        current_idx = self.speed_combo.currentIndex()
        if current_idx > 0:
            self.speed_combo.setCurrentIndex(current_idx - 1)

    def increase_speed(self):
        """Increase playback speed by one step."""
        current_idx = self.speed_combo.currentIndex()
        if current_idx < self.speed_combo.count() - 1:
            self.speed_combo.setCurrentIndex(current_idx + 1)

    def toggle_playback(self):
        """Toggle video playback between play and pause."""
        if (
            self.video_player.media_player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.video_player.pause()
        else:
            self.video_player.play()

    def seek_relative(self, offset: int):
        """Seek relative to current position.

        Args:
            offset: Time offset in milliseconds (positive or negative)
        """
        current_pos = self.video_player.position()
        new_pos = max(0, min(current_pos + offset, self.video_player.duration()))
        self.video_player.seek(new_pos)

        # Announce seek for screen readers
        direction = "forward" if offset > 0 else "backward"
        abs_seconds = abs(offset) // 1000
        self.setAccessibleDescription(f"Seeked {direction} {abs_seconds} seconds")

    def seek_to_position(self, position: int):
        """Seek to a specific position.

        Args:
            position: Position in milliseconds
        """
        self.video_player.seek(position)
        # Announce position for screen readers
        seconds = position // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        self.setAccessibleDescription(f"Seeked to {minutes}:{seconds:02d}")

    def seek_to_annotation(self, annotation: Annotation):
        """Seek to an annotation's timestamp.

        Args:
            annotation: The annotation to seek to
        """
        self.video_player.seek(int(annotation.timestamp * 1000))

    def on_position_changed(self, position: int):
        """Handle video position changes.

        Args:
            position: Current position in milliseconds
        """
        if not self.timeline_slider.isSliderDown():
            self.timeline_slider.setValue(position)
        self.update_time_label()

    def on_duration_changed(self, duration: int):
        """Handle video duration changes.

        Args:
            duration: Video duration in milliseconds
        """
        self.timeline_slider.setRange(0, duration)
        self.annotation_display.video_duration = duration / 1000

        # Announce duration for screen readers
        total_seconds = duration // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        self.setAccessibleDescription(
            f"Video duration: {minutes} minutes and {seconds} seconds"
        )

    def on_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle video playback state changes.

        Args:
            state: New playback state
        """
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(QIcon.fromTheme("media-playback-pause"))
            self.play_button.setToolTip("Pause (Space)")
        else:
            self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
            self.play_button.setToolTip("Play (Space)")

    def update_time_label(self):
        """Update the time display label."""
        position = self.video_player.position()
        duration = self.video_player.duration()

        position_str = self.format_time(position)
        duration_str = self.format_time(duration)

        self.time_label.setText(f"{position_str} / {duration_str}")

    def format_time(self, ms: float) -> str:
        """Format milliseconds into MM:SS format.

        Args:
            ms: Time in milliseconds

        Returns:
            Time string in MM:SS format
        """
        total_seconds = int(ms / 1000)
        m = total_seconds // 60
        s = total_seconds % 60
        return f"{m:02d}:{s:02d}"

    def show_annotation_tool(self):
        """Show the annotation creation tool."""
        current_time = self.video_player.position() / 1000  # Convert to seconds
        tool = AnnotationTool(
            current_time=current_time,
            player_name=self.player_name,
            parent=self,
            on_create=self.add_annotation,
        )
        tool.show()
        self.setAccessibleDescription("Annotation tool opened")

    def add_annotation(self, annotation: Annotation):
        """Add a new annotation.

        Args:
            annotation: The annotation to add
        """
        self.annotations[str(annotation.id)] = annotation
        self.annotation_display.add_annotation(annotation)
        self.annotationAdded.emit(annotation.__dict__)
        # Announce for screen readers
        self.setAccessibleDescription(
            f"Added annotation at {annotation.timestamp:.1f} seconds: {annotation.text}"
        )

    def remove_annotation(self, annotation: Annotation):
        """Remove an annotation.

        Args:
            annotation: The annotation to remove
        """
        self.annotations.pop(str(annotation.id), None)
        self.annotation_display.remove_annotation(annotation)

    def edit_annotation(self, annotation: Annotation):
        """Show the annotation tool for editing.

        Args:
            annotation: The annotation to edit
        """
        tool = AnnotationTool(
            current_time=annotation.timestamp, player_name=self.player_name, parent=self
        )
        tool.text_input.setText(annotation.text)
        tool.duration_input.setValue(int(annotation.duration))
        tool.current_color = QColor(annotation.color)
        tool.update_color_button()

        def on_edit(new_annotation: Annotation):
            self.remove_annotation(annotation)
            self.add_annotation(new_annotation)

        tool.annotationCreated.connect(on_edit)
        tool.show()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.

        Args:
            event: Key press event
        """
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_playback()
        elif key == Qt.Key.Key_Left:
            self.seek_relative(-5)
        elif key == Qt.Key.Key_Right:
            self.seek_relative(5)
        elif (
            key == Qt.Key.Key_A
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.show_annotation_tool()
        else:
            super().keyPressEvent(event)

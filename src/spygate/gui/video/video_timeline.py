from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .timeline_manager import TimelineManager
from .timeline_preview_worker import TimelinePreviewWorker


class VideoTimeline(QWidget):
    """Video timeline widget with frame navigation controls, drag-and-drop, and marker support."""

    def __init__(self, media_player: QMediaPlayer, parent=None):
        super().__init__(parent)
        self.timeline_manager = TimelineManager(media_player)
        self.timeline_elements = []  # List of dicts: {type, position, data}
        self.markers = []  # List of dicts: {time, label, color}
        self._drag_over = False  # For visual feedback
        self.preview_cache = {}  # frame_index: QPixmap
        self._preview_worker = None
        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.layout.addWidget(self.position_slider)

        # Time labels layout
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00:00")
        self.duration_label = QLabel("00:00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.duration_label)
        self.layout.addLayout(time_layout)

        # Frame counter layout
        frame_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        frame_layout.addWidget(self.frame_label)
        frame_layout.addStretch()
        self.layout.addLayout(frame_layout)

        # Control buttons layout
        controls_layout = QHBoxLayout()

        # Previous frame button
        self.prev_frame_btn = QPushButton()
        self.prev_frame_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.prev_frame_btn.setToolTip("Previous Frame (Left Arrow)")

        # Play/Pause button
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_pause_btn.setToolTip("Play/Pause (Space)")

        # Next frame button
        self.next_frame_btn = QPushButton()
        self.next_frame_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        self.next_frame_btn.setToolTip("Next Frame (Right Arrow)")

        # Add Marker button
        self.add_marker_btn = QPushButton("Add Marker")
        self.add_marker_btn.setToolTip("Add marker at current position")
        controls_layout.addStretch()
        controls_layout.addWidget(self.prev_frame_btn)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.next_frame_btn)
        controls_layout.addWidget(self.add_marker_btn)
        controls_layout.addStretch()

        self.layout.addLayout(controls_layout)

    def _connect_signals(self):
        """Connect signals to slots."""
        # Timeline manager signals
        self.timeline_manager.position_changed.connect(self._update_position)
        self.timeline_manager.duration_changed.connect(self._update_duration)
        self.timeline_manager.frame_changed.connect(self._update_frame_display)
        self.timeline_manager.state_changed.connect(self._update_play_state)

        # UI control signals
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.prev_frame_btn.clicked.connect(self._on_prev_frame_clicked)
        self.next_frame_btn.clicked.connect(self._on_next_frame_clicked)
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self.add_marker_btn.clicked.connect(self._on_add_marker_clicked)

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._on_prev_frame_clicked)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._on_next_frame_clicked)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_playback)

    def _on_prev_frame_clicked(self):
        """Handle previous frame button click."""
        self.timeline_manager.previous_frame()

    def _on_next_frame_clicked(self):
        """Handle next frame button click."""
        self.timeline_manager.next_frame()

    def _on_slider_released(self):
        """Handle slider release event."""
        position = self.position_slider.value()
        self.timeline_manager.seek_to_position(position)

    def _toggle_playback(self):
        """Toggle between play and pause states."""
        if self.timeline_manager.is_playing:
            self.timeline_manager.is_playing = False
        else:
            self.timeline_manager.is_playing = True

    def _update_position(self, position: int):
        """Update the position display and slider."""
        # Ensure the slider range is set before updating value
        if self.position_slider.maximum() < position:
            self.position_slider.setRange(0, position)
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self.current_time_label.setText(self._format_time(position))

    def _update_duration(self, duration: int):
        """Update the duration display and slider range."""
        self.position_slider.setRange(0, duration)
        self.duration_label.setText(self._format_time(duration))

    def _update_frame_display(self, frame: int):
        """Update the frame counter display."""
        self.frame_label.setText(f"Frame: {frame} / {self.timeline_manager.total_frames}")

    def _update_play_state(self, state: QMediaPlayer.PlaybackState):
        """Update the play/pause button icon based on playback state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
        else:
            self.play_pause_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        hours = minutes // 60

        seconds %= 60
        minutes %= 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Left:
            self.timeline_manager.previous_frame()
        elif event.key() == Qt.Key.Key_Right:
            self.timeline_manager.next_frame()
        elif event.key() == Qt.Key.Key_Space:
            self._toggle_playback()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        """Accept drag if supported type."""
        if event.mimeData().hasFormat("text/plain") or event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drag_over = True
            self.update()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Accept drag move if supported type."""
        if event.mimeData().hasFormat("text/plain") or event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_over = False
        self.update()
        event.accept()

    def dropEvent(self, event):
        """Handle drop event for timeline elements."""
        self._drag_over = False
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        # Calculate timeline position (frame/time) from mouse x
        x = pos.x()
        width = self.width()
        duration = self.timeline_manager.duration or 1
        dropped_time = int((x / width) * duration)
        # Snap to nearest frame
        frame_rate = self.timeline_manager.frame_rate or 30.0
        snapped_frame = round((dropped_time / 1000) * frame_rate)
        snapped_time = int((snapped_frame / frame_rate) * 1000)
        # For demo: support text/plain (e.g., 'clip', 'effect', 'transition')
        if event.mimeData().hasFormat("text/plain"):
            element_type = event.mimeData().text()
        elif event.mimeData().hasUrls():
            element_type = "clip"
        else:
            element_type = "unknown"
        self.timeline_elements.append(
            {
                "type": element_type,
                "time": snapped_time,
                "frame": snapped_frame,
                "data": event.mimeData().text() if event.mimeData().hasText() else None,
            }
        )
        self.update()
        event.acceptProposedAction()

    def _on_add_marker_clicked(self):
        """Add a marker at the current timeline position."""
        time = self.timeline_manager.position
        label = f"Marker {len(self.markers) + 1}"
        color = "#e74c3c"  # Red for now; can be extended
        self.markers.append({"time": time, "label": label, "color": color})
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = None
        try:
            from PyQt6.QtGui import QColor, QPainter

            painter = QPainter(self)
            # Draw drag-over highlight
            if self._drag_over:
                painter.setBrush(QColor(61, 174, 233, 60))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(self.rect())
            # Draw timeline elements as colored rectangles
            for elem in self.timeline_elements:
                x = int((elem["time"] / (self.timeline_manager.duration or 1)) * self.width())
                w = 10
                h = 20
                y = self.height() - h - 10
                color = QColor(61, 174, 233) if elem["type"] == "clip" else QColor(233, 174, 61)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(x - w // 2, y, w, h)
            # Draw markers as vertical lines
            for marker in self.markers:
                x = int((marker["time"] / (self.timeline_manager.duration or 1)) * self.width())
                painter.setPen(QColor(marker["color"]))
                painter.drawLine(x, 0, x, self.height())
        finally:
            if painter:
                painter.end()

    def start_preview_generation(self, video_path, frame_indices, thumbnail_size=QSize(80, 45)):
        """Start background generation of timeline previews."""
        # Stop any existing worker
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.stop()
            self._preview_worker.wait()
        self._preview_worker = TimelinePreviewWorker(video_path, frame_indices, thumbnail_size)
        self._preview_worker.progress.connect(self._on_preview_progress)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_progress(self, current, total):
        # Optionally update a progress bar or status label
        pass

    def _on_preview_finished(self, previews):
        """Update cache and UI with generated previews."""
        self.preview_cache.update(previews)
        self.update()  # Trigger repaint if previews are shown in paintEvent

    def _on_preview_error(self, message):
        QMessageBox.warning(self, "Preview Error", message)

    def closeEvent(self, event):
        # Ensure worker is stopped on widget close
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.stop()
            self._preview_worker.wait()
        super().closeEvent(event)

    # Example usage: call this when a new video is loaded or timeline needs preview update
    def on_video_loaded(self, video_path, total_frames):
        # Generate previews for every Nth frame (e.g., every 100 frames)
        frame_indices = list(range(0, total_frames, 100))
        self.start_preview_generation(video_path, frame_indices)

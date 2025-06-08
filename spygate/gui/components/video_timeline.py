"""Implement video timeline component with markers and interactive controls."""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PyQt6.QtCore import QCache, QPoint, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPoint,
    QShortcut,
    QToolTip,
)
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.hardware import HardwareDetector
from ...core.optimizer import TierOptimizer
from ..workers.video_playback_worker import VideoPlaybackWorker

logger = logging.getLogger(__name__)


class AnnotationType(Enum):
    """Types of annotations that can be added to the timeline."""

    MARKER = auto()  # Single frame marker
    REGION = auto()  # Region selection spanning multiple frames
    TEXT = auto()  # Text annotation at a specific frame


@dataclass
class Annotation:
    """Data class for storing annotation information."""

    type: AnnotationType
    start_frame: int
    end_frame: Optional[int] = None  # For regions
    text: Optional[str] = None
    color: QColor = QColor(Qt.GlobalColor.yellow)
    metadata: Optional[Dict] = None

    def is_region(self) -> bool:
        """Check if this is a region annotation."""
        return self.type == AnnotationType.REGION and self.end_frame is not None

    def contains_frame(self, frame: int) -> bool:
        """Check if the annotation includes the given frame."""
        if self.is_region():
            return self.start_frame <= frame <= self.end_frame
        return frame == self.start_frame

    def to_dict(self) -> dict:
        """Convert annotation to dictionary for storage."""
        return {
            "type": self.type.name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "text": self.text,
            "color": self.color.name(),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Create annotation from dictionary."""
        return cls(
            type=AnnotationType[data["type"]],
            start_frame=data["start_frame"],
            end_frame=data.get("end_frame"),
            text=data.get("text"),
            color=QColor(data.get("color", Qt.GlobalColor.yellow)),
            metadata=data.get("metadata", {}),
        )


class ThumbnailWorker(QThread):
    """Worker thread for generating video thumbnails."""

    thumbnailReady = pyqtSignal(int, QPixmap)  # frame number, thumbnail
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = Queue()
        self._should_stop = False
        self.video_path = None
        self.thumbnail_size = QSize(80, 45)

    def set_video(self, video_path: str, thumbnail_size: QSize):
        """Set the video file and thumbnail size."""
        self.video_path = video_path
        self.thumbnail_size = thumbnail_size

    def add_frame(self, frame_number: int):
        """Add a frame to the thumbnail generation queue."""
        self.queue.put(frame_number)

    def stop(self):
        """Stop the worker thread."""
        self._should_stop = True

    def run(self):
        """Main worker loop."""
        try:
            if not self.video_path:
                self.error.emit("No video file specified")
                return

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Failed to open video: {self.video_path}")
                return

            while not self._should_stop:
                if self.queue.empty():
                    self.msleep(100)  # Sleep to prevent busy waiting
                    continue

                frame_number = self.queue.get()

                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    continue

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to thumbnail size
                thumbnail = cv2.resize(
                    frame_rgb,
                    (self.thumbnail_size.width(), self.thumbnail_size.height()),
                )

                # Convert to QPixmap
                height, width, channel = thumbnail.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    thumbnail.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(q_image)

                # Emit thumbnail
                self.thumbnailReady.emit(frame_number, pixmap)

            cap.release()

        except Exception as e:
            self.error.emit(f"Thumbnail generation error: {str(e)}")
            logger.error(f"Thumbnail generation error: {str(e)}", exc_info=True)


class VideoTimeline(QWidget):
    """Interactive timeline component for video navigation and annotation display."""

    # Custom signals
    positionChanged = pyqtSignal(float)  # Emits position as percentage (0-1)
    annotationSelected = pyqtSignal(dict)  # Emits selected annotation data
    frameSelected = pyqtSignal(int)  # Emits selected frame number
    playbackSpeedChanged = pyqtSignal(float)  # Emits playback speed multiplier
    frameRateChanged = pyqtSignal(int)  # Emits new frame rate
    playbackStateChanged = pyqtSignal(bool)  # Emits whether playback is active
    annotationAdded = pyqtSignal(dict)  # Emits when a new annotation is added
    annotationDeleted = pyqtSignal(int)  # Emits when an annotation is deleted

    # Constants
    TIMELINE_HEIGHT = 60
    TIMELINE_MARGIN = 10
    MARKER_WIDTH = 8
    MARKER_HEIGHT = 20
    ANNOTATION_TRACK_HEIGHT = 15
    MAX_ANNOTATION_TRACKS = 3

    def __init__(self, parent=None):
        """Initialize the video timeline component."""
        super().__init__(parent)

        # Initialize hardware-aware components
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize thumbnail cache
        self.thumbnail_cache = QCache(100)  # Cache up to 100 thumbnails

        # Create pre-computed color brushes and pens
        self.track_brushes = {
            i: QBrush(QColor(40, 40, 40)) for i in range(self.MAX_ANNOTATION_TRACKS)
        }
        self.timeline_brush = QBrush(Qt.GlobalColor.darkGray)
        self.position_pen = QPen(Qt.GlobalColor.red, 2)
        self.annotation_pens = {
            AnnotationType.MARKER: QPen(Qt.GlobalColor.yellow, 2),
            AnnotationType.REGION: QPen(Qt.GlobalColor.cyan, 2),
            AnnotationType.TEXT: QPen(Qt.GlobalColor.green, 2),
        }

        # Pre-compute fonts
        self.annotation_font = QFont()
        self.annotation_font.setPointSize(8)

        # Initialize UI and variables
        self.initUI()
        self.initVariables()
        self.setupShortcuts()

        # Initialize workers
        self._setup_workers()

        # Enable double buffering for smoother rendering
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def _setup_workers(self):
        """Initialize worker threads."""
        # Initialize thumbnail worker with optimized settings
        self.thumbnail_worker = ThumbnailWorker()
        self.thumbnail_worker.thumbnailReady.connect(self._on_thumbnail_ready)
        self.thumbnail_worker.error.connect(self._on_thumbnail_error)
        self.thumbnail_worker.start(
            QThread.Priority.LowPriority
        )  # Lower priority for thumbnails

        # Initialize playback worker
        self.playback_worker = VideoPlaybackWorker()
        self.playback_worker.frameReady.connect(self._on_frame_ready)
        self.playback_worker.error.connect(self._on_playback_error)
        self.playback_worker.finished.connect(self._on_playback_finished)
        self.playback_worker.start(
            QThread.Priority.HighPriority
        )  # Higher priority for playback

    def initUI(self):
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create controls bar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # Playback controls
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.togglePlayback)
        controls_layout.addWidget(self.play_button)

        # Frame rate control
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.onFrameRateChanged)

        fps_label = QLabel("FPS:")
        controls_layout.addWidget(fps_label)
        controls_layout.addWidget(self.fps_spinbox)

        # Playback speed control
        self.speed_combo = QComboBox()
        speeds = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
        self.speed_combo.addItems(speeds)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.onPlaybackSpeedChanged)

        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        controls_layout.addWidget(self.speed_combo)

        # Loop control
        self.loop_button = QPushButton("Loop")
        self.loop_button.setCheckable(True)
        self.loop_button.clicked.connect(self.toggleLoop)
        controls_layout.addWidget(self.loop_button)

        # Add annotation controls
        self.annotation_button = QToolButton()
        self.annotation_button.setText("Add")
        self.annotation_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )

        annotation_menu = QMenu()
        annotation_menu.addAction(
            "Marker", lambda: self.startAnnotation(AnnotationType.MARKER)
        )
        annotation_menu.addAction(
            "Region", lambda: self.startAnnotation(AnnotationType.REGION)
        )
        annotation_menu.addAction(
            "Text", lambda: self.startAnnotation(AnnotationType.TEXT)
        )

        self.annotation_button.setMenu(annotation_menu)
        controls_layout.addWidget(self.annotation_button)

        # Add spacer
        controls_layout.addStretch()

        # Add controls to main layout
        layout.addLayout(controls_layout)

        # Set fixed height based on content
        self.setMinimumHeight(
            self.TIMELINE_HEIGHT
            + self.ANNOTATION_TRACK_HEIGHT * self.MAX_ANNOTATION_TRACKS
            + 40  # Controls height
        )

    def initVariables(self):
        """Initialize instance variables."""
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.video_path = None
        self.zoom_level = 1.0
        self.is_playing = False
        self.loop_enabled = False
        self.thumbnails = {}
        self.thumbnail_size = self.optimizer.get_optimal_thumbnail_size()
        self.dragging = False
        self.last_mouse_pos = None
        self.last_paint_rect = None  # For optimizing repaints

        # Annotation variables with optimized data structures
        self.annotations = []
        self.current_annotation = None
        self.annotation_mode = False
        self.selected_annotation = None
        self.annotation_tracks = {i: [] for i in range(self.MAX_ANNOTATION_TRACKS)}

        # Frame lookup optimization
        self._frame_x_cache = {}  # Cache frame to x coordinate conversions
        self._x_frame_cache = {}  # Cache x coordinate to frame conversions
        self._cache_size = 1000  # Maximum cache size

    def setupShortcuts(self):
        """Set up keyboard shortcuts."""
        # Frame navigation
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.previousFrame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.nextFrame)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self.goToStart)
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self.goToEnd)

        # Fast navigation (10 frames)
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self.skipFrames(-10))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self.skipFrames(10))

        # Very fast navigation (100 frames)
        QShortcut(QKeySequence("Ctrl+Left"), self, lambda: self.skipFrames(-100))
        QShortcut(QKeySequence("Ctrl+Right"), self, lambda: self.skipFrames(100))

        # Playback controls
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.togglePlayback)
        QShortcut(QKeySequence("L"), self, self.toggleLoop)

        # Annotation shortcuts
        QShortcut(
            QKeySequence("M"), self, lambda: self.startAnnotation(AnnotationType.MARKER)
        )
        QShortcut(
            QKeySequence("R"), self, lambda: self.startAnnotation(AnnotationType.REGION)
        )
        QShortcut(
            QKeySequence("T"), self, lambda: self.startAnnotation(AnnotationType.TEXT)
        )
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.cancelAnnotation)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self.deleteSelectedAnnotation)

    def setVideoInfo(self, total_frames, fps, video_path=None):
        """Set video information for the timeline."""
        self.total_frames = total_frames
        self.fps = fps
        self.fps_spinbox.setValue(fps)
        self.video_path = video_path
        self.updateTimelineWidth()

        # Initialize video capture for thumbnails
        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.initializeThumbnails()

            # Set up playback worker
            self.playback_worker.set_video(video_path)

        self.update()

    def initializeThumbnails(self):
        """Initialize thumbnail generation."""
        if not self.video_capture:
            return

        # Clear existing thumbnails
        self.thumbnails.clear()
        self.thumbnail_queue.clear()

        # Calculate thumbnail interval based on total frames and timeline width
        interval = max(
            1, self.total_frames // (self.timeline_width // self.thumbnail_size.width())
        )

        # Queue thumbnails for generation
        for frame in range(0, self.total_frames, interval):
            self.thumbnail_queue.append(frame)

        # Start thumbnail generation timer
        if not self.thumbnail_timer.isActive():
            self.thumbnail_timer.start(50)  # Generate thumbnails every 50ms

    def loadNextThumbnail(self):
        """Load the next thumbnail in the queue."""
        if not self.thumbnail_queue:
            self.thumbnail_timer.stop()
            return

        frame = self.thumbnail_queue.pop(0)
        self.thumbnail_worker.add_frame(frame)

    def _on_thumbnail_ready(self, frame_number: int, thumbnail: QPixmap):
        """Handle generated thumbnail."""
        self.thumbnails[frame_number] = thumbnail
        self.update()  # Trigger repaint

    def _on_thumbnail_error(self, error_msg: str):
        """Handle thumbnail generation error."""
        logger.error(f"Thumbnail error: {error_msg}")

    def updateTimelineWidth(self):
        """Update the timeline width based on total frames and zoom level."""
        min_width = self.timeline_container.width()
        self.timeline_width = max(min_width, int(self.total_frames * self.zoom_level))
        self.timeline_container.setMinimumWidth(self.timeline_width)

    def zoom(self, factor):
        """Zoom the timeline by the given factor."""
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))

        # Maintain the center point during zoom
        if old_zoom != self.zoom_level:
            self.updateTimelineWidth()
            self.update()

    def startAnnotation(self, annotation_type: AnnotationType):
        """Start creating a new annotation."""
        self.annotation_mode = True
        self.current_annotation = Annotation(
            type=annotation_type, start_frame=self.current_frame
        )

        if annotation_type == AnnotationType.TEXT:
            text, ok = QInputDialog.getText(
                self, "Add Text Annotation", "Enter annotation text:"
            )
            if ok and text:
                self.current_annotation.text = text
                self.finishAnnotation()
            else:
                self.cancelAnnotation()
        elif annotation_type == AnnotationType.MARKER:
            color = QColorDialog.getColor(
                initial=QColor(Qt.GlobalColor.yellow),
                parent=self,
                title="Choose Marker Color",
            )
            if color.isValid():
                self.current_annotation.color = color
                self.finishAnnotation()
            else:
                self.cancelAnnotation()

    def finishAnnotation(self):
        """Finish creating the current annotation."""
        if not self.current_annotation:
            return

        if self.current_annotation.type == AnnotationType.REGION:
            self.current_annotation.end_frame = self.current_frame

            # Ensure start_frame is before end_frame
            if self.current_annotation.end_frame < self.current_annotation.start_frame:
                (
                    self.current_annotation.start_frame,
                    self.current_annotation.end_frame,
                ) = (
                    self.current_annotation.end_frame,
                    self.current_annotation.start_frame,
                )

        # Find available track
        track_assigned = False
        for track_num, track in self.annotation_tracks.items():
            # Check if this annotation overlaps with any in this track
            overlaps = any(
                self._annotations_overlap(ann, self.current_annotation) for ann in track
            )
            if not overlaps:
                track.append(self.current_annotation)
                track_assigned = True
                break

        if not track_assigned:
            logger.warning("No available tracks for annotation")
            return

        self.annotations.append(self.current_annotation)
        self.annotationAdded.emit(self.current_annotation.to_dict())
        self.annotation_mode = False
        self.current_annotation = None
        self.update()

    def cancelAnnotation(self):
        """Cancel the current annotation."""
        self.annotation_mode = False
        self.current_annotation = None
        self.update()

    def deleteSelectedAnnotation(self):
        """Delete the currently selected annotation."""
        if not self.selected_annotation:
            return

        # Remove from track
        for track in self.annotation_tracks.values():
            if self.selected_annotation in track:
                track.remove(self.selected_annotation)
                break

        # Remove from main list
        if self.selected_annotation in self.annotations:
            idx = self.annotations.index(self.selected_annotation)
            self.annotations.remove(self.selected_annotation)
            self.annotationDeleted.emit(idx)

        self.selected_annotation = None
        self.update()

    def _annotations_overlap(self, a1: Annotation, a2: Annotation) -> bool:
        """Check if two annotations overlap in time."""
        if a1.type == AnnotationType.MARKER or a2.type == AnnotationType.MARKER:
            return a1.start_frame == a2.start_frame

        if a1.type == AnnotationType.REGION and a2.type == AnnotationType.REGION:
            return not (a1.end_frame < a2.start_frame or a2.end_frame < a1.start_frame)

        # Text annotations are treated like markers
        return a1.start_frame == a2.start_frame

    def _frame_to_x(self, frame: int) -> int:
        """Convert frame number to x coordinate with caching."""
        if frame in self._frame_x_cache:
            return self._frame_x_cache[frame]

        if self.total_frames == 0:
            return self.TIMELINE_MARGIN

        timeline_width = self.width() - 2 * self.TIMELINE_MARGIN
        x = int(self.TIMELINE_MARGIN + (frame / self.total_frames) * timeline_width)

        # Cache the result
        if len(self._frame_x_cache) >= self._cache_size:
            self._frame_x_cache.clear()  # Clear cache if too large
        self._frame_x_cache[frame] = x

        return x

    def _x_to_frame(self, x: int) -> int:
        """Convert x coordinate to frame number with caching."""
        if x in self._x_frame_cache:
            return self._x_frame_cache[x]

        timeline_width = self.width() - 2 * self.TIMELINE_MARGIN
        relative_pos = max(0, min(x - self.TIMELINE_MARGIN, timeline_width))
        frame = int((relative_pos / timeline_width) * self.total_frames)
        frame = max(0, min(frame, self.total_frames - 1))

        # Cache the result
        if len(self._x_frame_cache) >= self._cache_size:
            self._x_frame_cache.clear()  # Clear cache if too large
        self._x_frame_cache[x] = frame

        return frame

    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
        # Clear coordinate caches on resize
        self._frame_x_cache.clear()
        self._x_frame_cache.clear()

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            y = event.position().y()

            # Check for annotation selection
            self.selected_annotation = None
            timeline_bottom = self.height() - self.TIMELINE_MARGIN

            for track_num, track in self.annotation_tracks.items():
                track_y = (
                    timeline_bottom - (track_num + 1) * self.ANNOTATION_TRACK_HEIGHT
                )

                if track_y <= y <= track_y + self.ANNOTATION_TRACK_HEIGHT:
                    # Check each annotation in this track
                    for ann in track:
                        x1 = self._frame_to_x(ann.start_frame)

                        if ann.type == AnnotationType.REGION:
                            x2 = self._frame_to_x(ann.end_frame)
                            if x1 <= x <= x2:
                                self.selected_annotation = ann
                                break
                        else:  # MARKER or TEXT
                            if abs(x - x1) <= self.MARKER_WIDTH / 2:
                                self.selected_annotation = ann
                                break

                    if self.selected_annotation:
                        break

            if self.selected_annotation:
                self.annotationSelected.emit(self.selected_annotation.to_dict())
            else:
                # Handle timeline click
                self.handleTimelineClick(x)

            self.dragging = True
            self.last_mouse_pos = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.dragging:
            x = event.position().x()

            if self.annotation_mode and self.current_annotation:
                if self.current_annotation.type == AnnotationType.REGION:
                    self.current_annotation.end_frame = self._x_to_frame(x)
                    self.update()
            else:
                self.handleTimelineClick(x)

            self.last_mouse_pos = event.position()
            self.updatePreviewTooltip(x)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.annotation_mode and self.current_annotation:
                if self.current_annotation.type == AnnotationType.REGION:
                    self.finishAnnotation()

            self.dragging = False
            self.last_mouse_pos = None
            QToolTip.hideText()

    def handleTimelineClick(self, x_pos):
        """Handle click or drag on timeline to seek to position."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Calculate frame number from x position
        timeline_width = self.width() - 2 * self.TIMELINE_MARGIN
        relative_pos = max(0, min(x_pos - self.TIMELINE_MARGIN, timeline_width))
        frame = int((relative_pos / timeline_width) * self.total_frames)
        frame = max(0, min(frame, self.total_frames - 1))

        # Update current frame and emit signals
        self.current_frame = frame
        self.frameSelected.emit(frame)
        self.positionChanged.emit(frame / self.total_frames)

        # Request frame preview from playback worker
        self.playback_worker.seek_frame(frame)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(frame)
        self.update()

    def updatePreviewTooltip(self, x_pos):
        """Update the preview tooltip during scrubbing."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Calculate time and frame info
        timeline_width = self.width() - 2 * self.TIMELINE_MARGIN
        relative_pos = max(0, min(x_pos - self.TIMELINE_MARGIN, timeline_width))
        frame = int((relative_pos / timeline_width) * self.total_frames)
        frame = max(0, min(frame, self.total_frames - 1))

        time_seconds = frame / self.fps_spinbox.value()
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        seconds = int(time_seconds % 60)
        frames = int(
            (time_seconds * self.fps_spinbox.value()) % self.fps_spinbox.value()
        )

        # Format tooltip text
        tooltip = f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}\nFrame: {frame}"

        # Show tooltip at cursor position
        QToolTip.showText(self.mapToGlobal(QPoint(x_pos, 0)), tooltip)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Zoom in/out with Ctrl + Mouse Wheel
            delta = event.angleDelta().y()
            self.zoom(1.1 if delta > 0 else 0.9)

    def paintEvent(self, event):
        """Paint the timeline and annotations with optimizations."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Only redraw the area that needs updating
        update_rect = event.rect()
        if self.last_paint_rect == update_rect:
            return
        self.last_paint_rect = update_rect

        # Draw timeline background
        timeline_rect = QRectF(
            self.TIMELINE_MARGIN,
            self.height() - self.TIMELINE_HEIGHT - self.TIMELINE_MARGIN,
            self.width() - 2 * self.TIMELINE_MARGIN,
            self.TIMELINE_HEIGHT,
        )
        painter.fillRect(timeline_rect, self.timeline_brush)

        # Draw thumbnails efficiently
        if self.thumbnails:
            self._draw_thumbnails(painter, timeline_rect)

        # Draw annotations efficiently
        self._draw_annotations(painter)

        # Draw current position
        if self.total_frames > 0:
            pos_x = self._frame_to_x(self.current_frame)
            painter.setPen(self.position_pen)
            painter.drawLine(pos_x, 0, pos_x, self.height())

        # Draw current annotation in progress
        if self.annotation_mode and self.current_annotation:
            self._draw_current_annotation(painter, timeline_rect)

    def _draw_thumbnails(self, painter: QPainter, timeline_rect: QRectF):
        """Draw thumbnails efficiently."""
        thumb_width = self.thumbnail_size.width()
        timeline_width = timeline_rect.width()
        frames_per_thumb = max(
            1, self.total_frames // int(timeline_width / thumb_width)
        )

        # Only draw thumbnails in the visible area
        visible_start = max(0, self._x_to_frame(self.TIMELINE_MARGIN))
        visible_end = min(
            self.total_frames, self._x_to_frame(self.width() - self.TIMELINE_MARGIN)
        )

        for frame_num in range(visible_start, visible_end, frames_per_thumb):
            if frame_num in self.thumbnails:
                x = self._frame_to_x(frame_num)
                painter.drawPixmap(
                    x - thumb_width // 2,
                    timeline_rect.top(),
                    self.thumbnails[frame_num],
                )

    def _draw_annotations(self, painter: QPainter):
        """Draw annotations efficiently."""
        timeline_bottom = self.height() - self.TIMELINE_MARGIN

        # Create paths for each annotation type
        marker_path = QPainterPath()
        region_path = QPainterPath()
        text_path = QPainterPath()

        for track_num, track in self.annotation_tracks.items():
            track_y = timeline_bottom - (track_num + 1) * self.ANNOTATION_TRACK_HEIGHT

            for ann in track:
                x1 = self._frame_to_x(ann.start_frame)

                if ann.type == AnnotationType.REGION:
                    x2 = self._frame_to_x(ann.end_frame)
                    region_path.addRect(
                        x1, track_y, x2 - x1, self.ANNOTATION_TRACK_HEIGHT
                    )
                elif ann.type == AnnotationType.MARKER:
                    marker_path.moveTo(x1, track_y)
                    marker_path.lineTo(x1, track_y + self.MARKER_HEIGHT)
                    marker_path.moveTo(x1 - self.MARKER_WIDTH // 2, track_y)
                    marker_path.lineTo(x1 + self.MARKER_WIDTH // 2, track_y)
                else:  # TEXT
                    text_path.moveTo(x1, track_y)
                    text_path.lineTo(x1, track_y + self.MARKER_HEIGHT)
                    if ann.text:
                        painter.setFont(self.annotation_font)
                        painter.drawText(
                            QRectF(x1 + 5, track_y, 100, self.ANNOTATION_TRACK_HEIGHT),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            ann.text,
                        )

        # Draw all paths at once
        painter.setPen(self.annotation_pens[AnnotationType.MARKER])
        painter.drawPath(marker_path)

        painter.setPen(self.annotation_pens[AnnotationType.REGION])
        painter.drawPath(region_path)

        painter.setPen(self.annotation_pens[AnnotationType.TEXT])
        painter.drawPath(text_path)

    def _draw_current_annotation(self, painter: QPainter, timeline_rect: QRectF):
        """Draw the current annotation being created."""
        x1 = self._frame_to_x(self.current_annotation.start_frame)
        painter.setPen(QPen(self.current_annotation.color, 2, Qt.PenStyle.DashLine))

        if self.current_annotation.type == AnnotationType.REGION:
            x2 = self._frame_to_x(self.current_frame)
            painter.drawRect(
                QRectF(
                    min(x1, x2),
                    timeline_rect.top(),
                    abs(x2 - x1),
                    timeline_rect.height(),
                )
            )
        else:
            painter.drawLine(x1, 0, x1, self.height())

    def formatTime(self, seconds):
        """Format time in seconds to HH:MM:SS:FF format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds * self.fps_spinbox.value()) % self.fps_spinbox.value())
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def togglePlayback(self):
        """Toggle video playback."""
        if not self.video_path:
            return

        self.is_playing = not self.is_playing
        self.play_button.setChecked(self.is_playing)

        if self.is_playing:
            self.playback_worker.resume()
        else:
            self.playback_worker.pause()

        self.playbackStateChanged.emit(self.is_playing)

    def toggleLoop(self):
        """Toggle video looping."""
        loop = self.loop_button.isChecked()
        self.playback_worker.set_loop(loop)

    def _on_frame_ready(self, frame: np.ndarray, timestamp: float):
        """Handle frame from playback worker."""
        # Update current frame
        frame_number = int(timestamp * self.fps)
        self.setCurrentFrame(frame_number)

        # Emit frame for display
        self.frameSelected.emit(frame_number)

    def _on_playback_error(self, error_msg: str):
        """Handle playback error."""
        logger.error(f"Playback error: {error_msg}")
        self.is_playing = False
        self.play_button.setChecked(False)
        self.playbackStateChanged.emit(False)

    def _on_playback_finished(self):
        """Handle playback finished."""
        self.is_playing = False
        self.play_button.setChecked(False)
        self.playbackStateChanged.emit(False)

    def cleanup(self):
        """Clean up resources."""
        # Stop workers
        self.thumbnail_worker.stop()
        self.thumbnail_worker.wait()
        self.playback_worker.stop()
        self.playback_worker.wait()

        # Clear caches
        self.thumbnail_cache.clear()
        self._frame_x_cache.clear()
        self._x_frame_cache.clear()
        self.thumbnails.clear()

    def ensureFrameVisible(self, frame: int):
        """Ensure the specified frame is visible in the timeline."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Calculate visible area
        visible_start = max(0, self._x_to_frame(self.TIMELINE_MARGIN))
        visible_end = min(
            self.total_frames, self._x_to_frame(self.width() - self.TIMELINE_MARGIN)
        )

        # Ensure the frame is within the visible area
        if frame < visible_start or frame > visible_end:
            # Calculate new frame to scroll to
            if frame < visible_start:
                new_frame = visible_start
            else:
                new_frame = visible_end

            # Scroll to the new frame
            self.handleTimelineClick(self._frame_to_x(new_frame))

        self.update()

    def setCurrentFrame(self, frame: int):
        """Set the current frame and update the timeline."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Update current frame
        self.current_frame = frame
        self.frameSelected.emit(frame)
        self.positionChanged.emit(frame / self.total_frames)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(frame)
        self.update()

    def skipFrames(self, frames: int):
        """Skip a specified number of frames."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Calculate new frame
        new_frame = self.current_frame + frames
        new_frame = max(0, min(new_frame, self.total_frames - 1))

        # Update current frame
        self.current_frame = new_frame
        self.frameSelected.emit(new_frame)
        self.positionChanged.emit(new_frame / self.total_frames)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(new_frame)
        self.update()

    def previousFrame(self):
        """Go to the previous frame."""
        self.skipFrames(-1)

    def nextFrame(self):
        """Go to the next frame."""
        self.skipFrames(1)

    def goToStart(self):
        """Go to the start of the video."""
        self.current_frame = 0
        self.frameSelected.emit(0)
        self.positionChanged.emit(0.0)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(0)
        self.update()

    def goToEnd(self):
        """Go to the end of the video."""
        self.current_frame = self.total_frames - 1
        self.frameSelected.emit(self.current_frame)
        self.positionChanged.emit(1.0)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(self.current_frame)
        self.update()

    def onFrameRateChanged(self, new_fps: int):
        """Handle frame rate change."""
        self.fps = new_fps
        self.fps_spinbox.setValue(new_fps)
        self.update()

    def onPlaybackSpeedChanged(self, speed: str):
        """Handle playback speed change."""
        # Implementation of this method is not provided in the original file or the new code
        pass

    def setCurrentFrame(self, frame: int):
        """Set the current frame and update the timeline."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Update current frame
        self.current_frame = frame
        self.frameSelected.emit(frame)
        self.positionChanged.emit(frame / self.total_frames)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(frame)
        self.update()

    def setCurrentFrame(self, frame: int):
        """Set the current frame and update the timeline."""
        if not hasattr(self, "total_frames") or self.total_frames == 0:
            return

        # Update current frame
        self.current_frame = frame
        self.frameSelected.emit(frame)
        self.positionChanged.emit(frame / self.total_frames)

        # Ensure the frame is visible in the timeline
        self.ensureFrameVisible(frame)
        self.update()

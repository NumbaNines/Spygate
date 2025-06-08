from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..services.motion_service import MotionService
from ..visualization.motion_visualizer import MotionVisualizer


class MotionAnalysisView(QWidget):
    """Widget for displaying motion analysis results."""

    # Signals
    pattern_selected = pyqtSignal(dict)  # Emitted when a pattern is selected
    event_selected = pyqtSignal(dict)  # Emitted when an event is selected

    def __init__(self, motion_service: MotionService, parent: Optional[QWidget] = None):
        """Initialize the motion analysis view."""
        super().__init__(parent)
        self.motion_service = motion_service
        self.visualizer = MotionVisualizer()

        # State
        self.current_video_id = None
        self.current_frame = None
        self.current_time = 0.0
        self.fps = 30.0
        self.update_interval = 1000  # ms
        self.auto_update = True

        # Initialize UI
        self._init_ui()

        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(self.update_interval)

    def _init_ui(self):
        """Initialize the UI layout and components."""
        # Main layout
        layout = QVBoxLayout(self)

        # Controls
        controls = QHBoxLayout()

        # Time range controls
        self.time_range = QSpinBox()
        self.time_range.setRange(1, 300)  # 1-300 seconds
        self.time_range.setValue(30)
        self.time_range.valueChanged.connect(self._update_displays)

        # Pattern type filter
        self.pattern_filter = QComboBox()
        self.pattern_filter.addItems(["All", "Rapid", "Sustained", "Brief"])
        self.pattern_filter.currentTextChanged.connect(self._update_patterns)

        # Auto-update toggle
        self.auto_update_cb = QCheckBox("Auto Update")
        self.auto_update_cb.setChecked(True)
        self.auto_update_cb.stateChanged.connect(self._toggle_auto_update)

        # Update button
        self.update_btn = QPushButton("Update Now")
        self.update_btn.clicked.connect(self._update_displays)

        # Add controls to layout
        controls.addWidget(QLabel("Time Window (s):"))
        controls.addWidget(self.time_range)
        controls.addWidget(QLabel("Pattern Type:"))
        controls.addWidget(self.pattern_filter)
        controls.addWidget(self.auto_update_cb)
        controls.addWidget(self.update_btn)
        controls.addStretch()

        layout.addLayout(controls)

        # Main content area
        content = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Heatmap and current frame
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Heatmap display
        self.heatmap_label = QLabel()
        self.heatmap_label.setMinimumSize(320, 180)
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.heatmap_label)

        # Current frame display
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(320, 180)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.frame_label)

        content.addWidget(left_widget)

        # Right side - Events and patterns
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Motion events list
        events_frame = QFrame()
        events_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        events_layout = QVBoxLayout(events_frame)
        events_layout.addWidget(QLabel("<b>Motion Events</b>"))

        self.events_scroll = QScrollArea()
        self.events_scroll.setWidgetResizable(True)
        self.events_scroll.setMinimumHeight(200)
        self.events_content = QWidget()
        self.events_layout = QVBoxLayout(self.events_content)
        self.events_scroll.setWidget(self.events_content)
        events_layout.addWidget(self.events_scroll)

        right_layout.addWidget(events_frame)

        # Motion patterns list
        patterns_frame = QFrame()
        patterns_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        patterns_layout = QVBoxLayout(patterns_frame)
        patterns_layout.addWidget(QLabel("<b>Motion Patterns</b>"))

        self.patterns_scroll = QScrollArea()
        self.patterns_scroll.setWidgetResizable(True)
        self.patterns_scroll.setMinimumHeight(200)
        self.patterns_content = QWidget()
        self.patterns_layout = QVBoxLayout(self.patterns_content)
        self.patterns_scroll.setWidget(self.patterns_content)
        patterns_layout.addWidget(self.patterns_scroll)

        right_layout.addWidget(patterns_frame)

        content.addWidget(right_widget)
        layout.addWidget(content)

        # Set stretch factors
        content.setStretchFactor(0, 1)  # Left side
        content.setStretchFactor(1, 1)  # Right side

    def update_video(self, video_id: int, frame: np.ndarray, time: float, fps: float):
        """Update the current video context."""
        self.current_video_id = video_id
        self.current_frame = frame.copy()
        self.current_time = time
        self.fps = fps
        self._update_displays()

    def _update_displays(self):
        """Update all display components."""
        if self.current_video_id is None or self.current_frame is None:
            return

        # Get time range
        time_window = self.time_range.value()
        start_time = max(0, self.current_time - time_window)
        end_time = self.current_time

        # Update heatmap
        self._update_heatmap(start_time, end_time)

        # Update frame display
        self._update_frame_display()

        # Update events and patterns
        self._update_events(start_time, end_time)
        self._update_patterns(start_time, end_time)

    def _update_heatmap(self, start_time: float, end_time: float):
        """Update the motion heatmap display."""
        heatmap = self.motion_service.get_motion_heatmap(
            self.current_video_id, start_time, end_time
        )

        if heatmap is not None:
            # Convert heatmap to color
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Resize to match frame size if needed
            if self.current_frame is not None:
                heatmap_color = cv2.resize(
                    heatmap_color,
                    (self.current_frame.shape[1], self.current_frame.shape[0]),
                )

            # Convert to QPixmap
            height, width = heatmap_color.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(
                heatmap_color.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_img)

            # Scale to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                self.heatmap_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.heatmap_label.setPixmap(pixmap)

    def _update_frame_display(self):
        """Update the current frame display."""
        if self.current_frame is None:
            return

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap
        height, width = frame_rgb.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(
            frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            self.frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.frame_label.setPixmap(pixmap)

    def _update_events(self, start_time: float, end_time: float):
        """Update the motion events list."""
        # Clear existing events
        for i in reversed(range(self.events_layout.count())):
            self.events_layout.itemAt(i).widget().setParent(None)

        # Get events
        events = self.motion_service.get_motion_events(
            self.current_video_id, start_time, end_time
        )

        # Add event widgets
        for event in events:
            event_widget = self._create_event_widget(event)
            self.events_layout.addWidget(event_widget)

        # Add stretch to bottom
        self.events_layout.addStretch()

    def _update_patterns(self, start_time: float, end_time: float):
        """Update the motion patterns list."""
        # Clear existing patterns
        for i in reversed(range(self.patterns_layout.count())):
            self.patterns_layout.itemAt(i).widget().setParent(None)

        # Get pattern type filter
        pattern_type = self.pattern_filter.currentText().lower()
        pattern_types = None
        if pattern_type != "all":
            pattern_types = [f"{pattern_type}_movement"]

        # Get patterns
        patterns = self.motion_service.get_motion_patterns(
            self.current_video_id, start_time, end_time, pattern_types
        )

        # Add pattern widgets
        for pattern in patterns:
            pattern_widget = self._create_pattern_widget(pattern)
            self.patterns_layout.addWidget(pattern_widget)

        # Add stretch to bottom
        self.patterns_layout.addStretch()

    def _create_event_widget(self, event: Dict[str, Any]) -> QFrame:
        """Create a widget to display a motion event."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)

        # Event header
        header = QHBoxLayout()
        header.addWidget(QLabel(f'Frame {event["frame_number"]}'))
        header.addWidget(QLabel(f'Score: {event["motion_score"]:.2f}'))
        header.addStretch()
        layout.addLayout(header)

        # Situations
        for situation in event["situations"]:
            sit_text = (
                f"{situation['type']} "
                f"({situation['confidence']:.2f}) "
                f"in {situation['field_region']}"
            )
            layout.addWidget(QLabel(sit_text))

        # Make clickable
        frame.mousePressEvent = lambda e: self.event_selected.emit(event)

        return frame

    def _create_pattern_widget(self, pattern: Dict[str, Any]) -> QFrame:
        """Create a widget to display a motion pattern."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)

        # Pattern header
        header = QHBoxLayout()
        header.addWidget(QLabel(pattern["type"].replace("_", " ").title()))
        header.addWidget(QLabel(f'Confidence: {pattern["confidence"]:.2f}'))
        header.addStretch()
        layout.addLayout(header)

        # Pattern details
        details = (
            f"Direction: {pattern['direction']}, "
            f"Speed: {pattern['speed']:.1f}, "
            f"Duration: {pattern['duration']:.2f}s"
        )
        layout.addWidget(QLabel(details))

        # Make clickable
        frame.mousePressEvent = lambda e: self.pattern_selected.emit(pattern)

        return frame

    def _toggle_auto_update(self, state: int):
        """Toggle automatic updates."""
        self.auto_update = bool(state)
        if self.auto_update:
            self.update_timer.start(self.update_interval)
        else:
            self.update_timer.stop()

    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
        self._update_frame_display()  # Update frame scaling
        self._update_displays()  # Update other displays

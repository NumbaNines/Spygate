"""Widget for displaying video annotations on the timeline."""

from typing import Dict, List, Optional

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QKeySequence, QMouseEvent, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from ..models.annotation import Annotation


class AnnotationMarker(QFrame):
    """A visual marker representing an annotation on the timeline."""

    clicked = pyqtSignal(Annotation)

    def __init__(self, annotation: Annotation, parent: Optional[QWidget] = None):
        """Initialize the annotation marker.

        Args:
            annotation: The annotation this marker represents
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.annotation = annotation
        self.setFixedSize(8, 16)
        self.setToolTip(f"{annotation.text} at {annotation.timestamp:.1f} seconds")
        self.setStyleSheet(f"background-color: {annotation.color};")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set accessibility
        self.setAccessibleName(f"Annotation marker")
        self.setAccessibleDescription(
            f"Annotation at {annotation.timestamp:.1f} seconds: {annotation.text}. "
            "Click to show options menu."
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.annotation)

    def keyPressEvent(self, event):
        """Handle keyboard events for accessibility."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Space):
            self.clicked.emit(self.annotation)
        else:
            super().keyPressEvent(event)


class AnnotationDisplay(QWidget):
    """Widget for displaying annotations on the video timeline."""

    # Signals
    annotationSelected = pyqtSignal(Annotation)
    annotationDeleted = pyqtSignal(Annotation)
    annotationEdited = pyqtSignal(Annotation)

    def __init__(self, video_duration: float, parent: Optional[QWidget] = None):
        """Initialize the annotation display.

        Args:
            video_duration: Duration of the video in seconds
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.video_duration = video_duration
        self.annotations: dict[str, Annotation] = {}
        self.markers: dict[str, AnnotationMarker] = {}
        self.setup_ui()
        self.setup_shortcuts()

        # Set accessibility
        self.setAccessibleName("Annotation Display")
        self.setAccessibleDescription(
            "Timeline display of video annotations. "
            "Use Tab to navigate between annotations, "
            "Enter or Space to open annotation menu, "
            "and arrow keys to navigate menu options."
        )

    def setup_ui(self):
        """Set up the UI components."""
        self.setMinimumHeight(20)
        self.setStyleSheet("background-color: #1F2937;")  # Dark background

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for accessibility."""
        # Navigation between markers
        QShortcut(QKeySequence("Tab"), self, self.focus_next_marker)
        QShortcut(QKeySequence("Shift+Tab"), self, self.focus_prev_marker)

    def focus_next_marker(self):
        """Focus the next annotation marker."""
        current = self.focusWidget()
        if not current or not isinstance(current, AnnotationMarker):
            # Focus first marker if none focused
            if self.markers:
                next_marker = min(self.markers.values(), key=lambda m: m.annotation.timestamp)
                next_marker.setFocus()
            return

        # Find next marker by timestamp
        current_time = current.annotation.timestamp
        next_markers = [m for m in self.markers.values() if m.annotation.timestamp > current_time]
        if next_markers:
            next_marker = min(next_markers, key=lambda m: m.annotation.timestamp)
            next_marker.setFocus()

    def focus_prev_marker(self):
        """Focus the previous annotation marker."""
        current = self.focusWidget()
        if not current or not isinstance(current, AnnotationMarker):
            # Focus last marker if none focused
            if self.markers:
                prev_marker = max(self.markers.values(), key=lambda m: m.annotation.timestamp)
                prev_marker.setFocus()
            return

        # Find previous marker by timestamp
        current_time = current.annotation.timestamp
        prev_markers = [m for m in self.markers.values() if m.annotation.timestamp < current_time]
        if prev_markers:
            prev_marker = max(prev_markers, key=lambda m: m.annotation.timestamp)
            prev_marker.setFocus()

    def add_annotation(self, annotation: Annotation):
        """Add a new annotation to the display.

        Args:
            annotation: The annotation to add
        """
        self.annotations[str(annotation.id)] = annotation
        marker = AnnotationMarker(annotation, self)
        marker.clicked.connect(self.show_annotation_menu)
        self.markers[str(annotation.id)] = marker
        self.update_marker_positions()

        # Announce for screen readers
        self.setAccessibleDescription(
            f"Added annotation at {annotation.timestamp:.1f} seconds: {annotation.text}"
        )

    def remove_annotation(self, annotation: Annotation):
        """Remove an annotation from the display.

        Args:
            annotation: The annotation to remove
        """
        marker = self.markers.pop(str(annotation.id), None)
        if marker:
            marker.deleteLater()
        self.annotations.pop(str(annotation.id), None)

        # Announce for screen readers
        self.setAccessibleDescription(f"Removed annotation: {annotation.text}")

    def update_marker_positions(self):
        """Update the positions of all annotation markers."""
        width = self.width()
        for annotation_id, marker in self.markers.items():
            annotation = self.annotations[annotation_id]
            x_pos = (annotation.timestamp / self.video_duration) * width
            marker.move(int(x_pos) - 4, 2)  # Center the marker

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_marker_positions()

    def show_annotation_menu(self, annotation: Annotation):
        """Show the context menu for an annotation.

        Args:
            annotation: The annotation to show the menu for
        """
        menu = QMenu(self)

        # Edit action
        edit_action = menu.addAction("Edit")
        edit_action.setToolTip("Edit annotation text and properties")
        edit_action.triggered.connect(lambda: self.edit_annotation(annotation))

        # Delete action
        delete_action = menu.addAction("Delete")
        delete_action.setToolTip("Remove this annotation")
        delete_action.triggered.connect(lambda: self.delete_annotation(annotation))

        # Show menu at cursor position
        menu.exec(QCursor.pos())

        # Announce menu closure for screen readers
        self.setAccessibleDescription(f"Closed menu for annotation: {annotation.text}")

    def edit_annotation(self, annotation: Annotation):
        """Emit signal to edit an annotation.

        Args:
            annotation: The annotation to edit
        """
        self.annotationEdited.emit(annotation)
        self.setAccessibleDescription(f"Editing annotation: {annotation.text}")

    def delete_annotation(self, annotation: Annotation):
        """Delete an annotation.

        Args:
            annotation: The annotation to delete
        """
        self.remove_annotation(annotation)
        self.annotationDeleted.emit(annotation)

    def paintEvent(self, event):
        """Paint the timeline background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw timeline line
        pen = QPen(QColor("#4B5563"), 2)  # Gray line
        painter.setPen(pen)
        y = self.height() // 2
        painter.drawLine(0, y, self.width(), y)

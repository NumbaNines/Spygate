"""Widget for creating and editing video annotations."""

from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..models.annotation import Annotation


class AnnotationTool(QWidget):
    """Tool for creating and editing video annotations."""

    # Add the missing signal
    annotationCreated = pyqtSignal(Annotation)

    def __init__(
        self,
        current_time: float,
        player_name: str,
        parent: Optional[QWidget] = None,
        on_create: Optional[Callable[[Annotation], None]] = None,
    ):
        """Initialize the annotation tool.

        Args:
            current_time: Current video time in seconds
            player_name: Name of the player being analyzed
            parent: Optional parent widget
            on_create: Callback for when an annotation is created
        """
        super().__init__(parent)
        self.current_time = current_time
        self.player_name = player_name
        self.on_create = on_create
        self.color = QColor("#FF0000")  # Default to red

        # Create UI elements
        self.text_input = QLineEdit()
        self.duration_input = QSpinBox()
        self.color_button = QPushButton()

        self.setup_ui()

        # Set accessibility
        self.setAccessibleName("Annotation Tool")
        self.setAccessibleDescription("Create and edit video annotations")

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Text input
        text_layout = QHBoxLayout()
        text_label = QLabel("Annotation:")
        text_label.setBuddy(self.text_input)  # For accessibility
        self.text_input.setPlaceholderText("Enter annotation text")
        self.text_input.setAccessibleName("Annotation Text")
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_input)

        # Duration input
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration (seconds):")
        duration_label.setBuddy(self.duration_input)  # For accessibility
        self.duration_input.setRange(1, 60)
        self.duration_input.setValue(5)  # Default duration
        self.duration_input.setAccessibleName("Annotation Duration")
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_input)

        # Color picker
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        self.color_button.setFixedSize(32, 32)
        self.update_color_button()
        self.color_button.clicked.connect(self.show_color_dialog)
        self.color_button.setAccessibleName("Color Picker")
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()

        # Create button
        create_button = QPushButton("Create Annotation")
        create_button.clicked.connect(self.create_annotation)
        create_button.setAccessibleName("Create Annotation Button")

        # Add all layouts
        layout.addLayout(text_layout)
        layout.addLayout(duration_layout)
        layout.addLayout(color_layout)
        layout.addWidget(create_button)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Enter key to create annotation
        self.create_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        self.create_shortcut.activated.connect(self.create_annotation)
        self.create_shortcut.setWhatsThis("Create annotation")

        # Escape key to close tool
        self.close_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.close_shortcut.activated.connect(self.close)
        self.close_shortcut.setWhatsThis("Close annotation tool")

    def update_color_button(self):
        """Update the color button's appearance."""
        self.color_button.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid #666;"
        )
        self.color_button.setAccessibleDescription(
            f"Current color: {self.color.name()}"
        )

    def show_color_dialog(self):
        """Show the color picker dialog."""
        color = QColorDialog.getColor(
            self.color,
            self,
            "Choose Annotation Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self.color = color
            self.update_color_button()

    def create_annotation(self):
        """Create a new annotation with the current settings."""
        text = self.text_input.text().strip()
        if not text:
            self.text_input.setFocus()
            return

        duration = self.duration_input.value()
        annotation = Annotation(
            start_time=self.current_time,
            end_time=self.current_time + duration,
            text=text,
            color=self.color,
            player_name=self.player_name,
        )

        if self.on_create:
            self.on_create(annotation)

        self.close()

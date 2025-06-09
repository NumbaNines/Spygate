"""
Base Panel Component
"""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtProperty, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class BasePanel(QFrame):
    """Base panel class for all custom panels."""

    theme_changed = pyqtSignal(str)

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        """Initialize the base panel.

        Args:
            title: Optional panel title
            parent: Optional parent widget
        """
        super().__init__(parent)

        # Create main layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)

        # Create title label if provided
        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("panel-title")
            self._layout.addWidget(self.title_label)

            # Add separator
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setObjectName("panel-separator")
            self._layout.addWidget(separator)

        # Create content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.content.setLayout(self.content_layout)
        self.content.setObjectName("panel-content")
        self._layout.addWidget(self.content)

        # Set panel properties
        self.setObjectName("base-panel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        # Initialize theme
        self._theme = "dark_teal"
        self._theme_data = {}

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the panel content.

        Args:
            widget: Widget to add
        """
        self.content_layout.addWidget(widget)

    def clear_content(self) -> None:
        """Remove all widgets from the panel content."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_theme(self, theme_name):
        """Set the panel's theme name."""
        self._theme = theme_name
        self.theme_changed.emit(theme_name)
        self.update()

    def get_theme(self):
        """Get the current theme name."""
        return self._theme

    def update_theme(self, theme_data):
        """Update the panel's theme data.

        Args:
            theme_data: Dictionary containing theme colors and styles
        """
        self._theme_data = theme_data
        self.update()

    @pyqtProperty(dict)
    def theme(self):
        """Get the current theme data."""
        return self._theme_data

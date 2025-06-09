"""
Base Dialog Component
"""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtProperty, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class BaseDialog(QDialog):
    """Base class for all dialog components providing consistent behavior."""

    theme_changed = pyqtSignal(str)

    def __init__(
        self,
        title: str = "",
        parent: Optional[QWidget] = None,
        modal: bool = True,
        buttons: QDialogButtonBox.StandardButton = QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel,
    ):
        """Initialize the base dialog.

        Args:
            title: Dialog title
            parent: Optional parent widget
            modal: Whether the dialog is modal
            buttons: Standard buttons to include
        """
        super().__init__(parent)

        # Set dialog properties
        self.setWindowTitle(title)
        self.setModal(modal)
        self.setObjectName("base-dialog")
        self._theme = "dark_teal"
        self._theme_data = {}

        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(16)

        # Create content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.content.setLayout(self.content_layout)
        self.layout.addWidget(self.content)

        # Create button box
        self.button_box = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the dialog content.

        Args:
            widget: Widget to add
        """
        self.content_layout.addWidget(widget)

    def add_custom_button(
        self,
        text: str,
        role: QDialogButtonBox.ButtonRole = QDialogButtonBox.ButtonRole.ActionRole,
    ) -> QPushButton:
        """Add a custom button to the button box.

        Args:
            text: Button text
            role: Button role

        Returns:
            The created button
        """
        button = QPushButton(text)
        self.button_box.addButton(button, role)
        return button

    def clear_content(self) -> None:
        """Remove all widgets from the dialog content."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_theme(self, theme_name):
        """Set the dialog's theme name."""
        self._theme = theme_name
        self.theme_changed.emit(theme_name)
        self.update()

    def get_theme(self):
        """Get the current theme name."""
        return self._theme

    def update_theme(self, theme_data):
        """Update the dialog's theme data.

        Args:
            theme_data: Dictionary containing theme colors and styles
        """
        self._theme_data = theme_data
        self.update()

    @pyqtProperty(dict)
    def theme(self):
        """Get the current theme data."""
        return self._theme_data

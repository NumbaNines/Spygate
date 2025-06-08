"""Dialog component for displaying modal content."""

from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..base import BaseDialog


class Dialog(BaseDialog):
    """A material design dialog component."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the dialog.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._content: Optional[QWidget] = None
        self._buttons: list[QPushButton] = []
        self.content_widgets: list[QWidget] = []
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        super()._setup_ui()

        # Create button layout
        self._button_layout = QHBoxLayout()
        self._button_layout.setContentsMargins(8, 8, 8, 8)
        self._button_layout.setSpacing(8)
        self.layout.addLayout(self._button_layout)

    def set_content(self, widget: QWidget) -> None:
        """Set the main content widget.

        Args:
            widget: Widget to use as content
        """
        if self._content:
            self.layout.removeWidget(self._content)
        self._content = widget
        self.layout.insertWidget(0, widget)  # Insert before button layout

    def content(self) -> Optional[QWidget]:
        """Get the main content widget.

        Returns:
            The content widget or None
        """
        return self._content

    def add_button(self, text: str, callback: Callable[[], None]) -> None:
        """Add a button to the dialog.

        Args:
            text: Button text
            callback: Function to call when button is clicked
        """
        button = QPushButton(text)
        button.clicked.connect(callback)
        self._buttons.append(button)
        self._button_layout.addWidget(button)

    def buttons(self) -> list[QPushButton]:
        """Get all dialog buttons.

        Returns:
            List of buttons
        """
        return self._buttons

    def set_modal(self, modal: bool) -> None:
        """Set dialog modality.

        Args:
            modal: Whether the dialog should be modal
        """
        self.setModal(modal)

    def add_content(self, widget: QWidget) -> None:
        """Add a widget to the dialog's content area.

        Args:
            widget: Widget to add
        """
        self.layout.insertWidget(self.layout.count() - 1, widget)  # Insert before button layout
        self.content_widgets.append(widget)

    def update_theme(self, theme: dict[str, Any]) -> None:
        """Update the dialog's theme.

        Args:
            theme: Theme dictionary with style properties
        """
        super().update_theme(theme)
        for widget in self.content_widgets:
            if hasattr(widget, "update_theme"):
                widget.update_theme(theme)
        for button in self._buttons:
            if hasattr(button, "update_theme"):
                button.update_theme(theme)

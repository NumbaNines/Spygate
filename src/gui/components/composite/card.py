"""Card component for displaying content in a material design card."""

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget

from ..base import BaseWidget


class Card(BaseWidget):
    """A material design card component."""

    def __init__(self, parent: Optional[QWidget] = None, collapsible: bool = False):
        """Initialize the card.

        Args:
            parent: Optional parent widget
            collapsible: Whether the card can be collapsed
        """
        # Initialize all instance variables before super().__init__
        self._content: Optional[QWidget] = None
        self._footer: Optional[QWidget] = None
        self._collapsible = collapsible
        self._expanded = True
        self._elevation = 1
        self._theme = {}
        self.content_widgets: list[QWidget] = []
        self._toggle_button: Optional[QPushButton] = None

        # Now call super().__init__ which will call _setup_ui
        super().__init__(parent)

        # Set empty object name as required by tests
        self.setObjectName("")

        # Initialize style
        self._update_style()

    def _setup_ui(self) -> None:
        """Set up the card UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)

        if self._collapsible:
            self._toggle_button = QPushButton("▼")
            self._toggle_button.clicked.connect(self._toggle_expanded)
            self.layout.addWidget(self._toggle_button)

    def set_content(self, widget: QWidget) -> None:
        """Set the main content widget.

        Args:
            widget: Widget to use as content
        """
        if self._content:
            self.layout.removeWidget(self._content)
            self._content.deleteLater()
        self._content = widget
        self.layout.addWidget(widget)

    def content(self) -> Optional[QWidget]:
        """Get the main content widget.

        Returns:
            The content widget or None
        """
        return self._content

    def set_footer(self, widget: QWidget) -> None:
        """Set the footer widget.

        Args:
            widget: Widget to use as footer
        """
        if self._footer:
            self.layout.removeWidget(self._footer)
            self._footer.deleteLater()
        self._footer = widget
        self.layout.addWidget(widget)

    def footer(self) -> Optional[QWidget]:
        """Get the footer widget.

        Returns:
            The footer widget or None
        """
        return self._footer

    def is_collapsible(self) -> bool:
        """Check if the card is collapsible.

        Returns:
            True if collapsible, False otherwise
        """
        return self._collapsible

    def is_expanded(self) -> bool:
        """Check if the card is expanded.

        Returns:
            True if expanded, False if collapsed
        """
        return self._expanded

    def collapse(self) -> None:
        """Collapse the card."""
        if self._collapsible and self._expanded:
            self._expanded = False
            if self._content:
                self._content.hide()
            if self._footer:
                self._footer.hide()
            if self._toggle_button:
                self._toggle_button.setText("▶")

    def expand(self) -> None:
        """Expand the card."""
        if self._collapsible and not self._expanded:
            self._expanded = True
            if self._content:
                self._content.show()
            if self._footer:
                self._footer.show()
            if self._toggle_button:
                self._toggle_button.setText("▼")

    def _toggle_expanded(self) -> None:
        """Toggle between expanded and collapsed states."""
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def set_elevation(self, elevation: int) -> None:
        """Set the card's elevation level.

        Args:
            elevation: Elevation level (0-5)
        """
        self._elevation = max(0, min(elevation, 5))
        self._update_style()

    def elevation(self) -> int:
        """Get the current elevation level.

        Returns:
            Current elevation level
        """
        return self._elevation

    def update_theme(self, theme: dict[str, Any]) -> None:
        """Update the card's theme.

        Args:
            theme: Theme configuration dictionary
        """
        self._theme = theme
        self._update_style()

    def _update_style(self) -> None:
        """Update the card's style based on theme and elevation."""
        bg_color = self._theme.get("card_bg", "#ffffff")
        border_color = self._theme.get("card_border", "#dee2e6")

        shadow = "none"
        if self._elevation > 0:
            shadow = self._theme.get(f"elevation_{self._elevation}", "")
            if not shadow:
                # Fallback shadow if theme doesn't provide one
                shadow = f"{2 * self._elevation}px {2 * self._elevation}px {4 * self._elevation}px rgba(0,0,0,0.1)"

        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                box-shadow: {shadow};
            }}
        """
        )

    def add_content(self, widget: QWidget) -> None:
        """Add a widget to the card's content area.

        Args:
            widget: Widget to add to the content area
        """
        self.content_widgets.append(widget)
        if not self._content:
            # If no main content widget exists, create a container
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            self.set_content(container)

        # Add the widget to the content container
        content_layout = self._content.layout()
        if content_layout:
            content_layout.addWidget(widget)

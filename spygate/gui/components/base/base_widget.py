"""
Base Widget Component
"""

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, pyqtProperty, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...layouts import LayoutConfig, LayoutManager


class BaseWidget(QWidget):
    """Base widget class that provides common functionality."""

    # Signal emitted when the widget state changes
    state_changed = pyqtSignal(dict)

    theme_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        layout_config: Optional[LayoutConfig] = None,
    ):
        """Initialize the base widget.

        Args:
            parent: Optional parent widget
            layout_config: Optional layout configuration
        """
        super().__init__(parent)

        # Initialize state
        self._state: dict[str, Any] = {}

        # Set default properties
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # Create layout with configuration
        config = layout_config or LayoutConfig()
        self._layout = LayoutManager.create_vertical(parent=self, config=config)

        # Apply size constraints and policies
        if config.min_size:
            self.setMinimumSize(config.min_size)
        if config.max_size:
            self.setMaximumSize(config.max_size)
        self.setSizePolicy(config.horizontal_policy, config.vertical_policy)

        self._theme = "dark_teal"
        self._theme_data = {}
        self.show()  # Make widget visible by default

    @property
    def layout(self):
        """Get the widget's layout."""
        return self._layout

    def set_state(self, key: str, value: Any) -> None:
        """Set a state value.

        Args:
            key: State key
            value: State value
        """
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            The state value or default
        """
        return self._state.get(key, default)

    def remove_state(self, key: str):
        """Remove a state value.

        Args:
            key: State key to remove
        """
        self._state.pop(key, None)

    def _on_state_changed(self, key: str, value: Any) -> None:
        """Handle state changes. Override in subclasses.

        Args:
            key: State key that changed
            value: New value for the state
        """
        pass

    def set_loading(self, loading: bool) -> None:
        """Set the loading state of the widget.

        Args:
            loading: Whether the widget is loading
        """
        self.setEnabled(not loading)
        self.set_state("loading", loading)

    def set_error(self, error: Optional[str]) -> None:
        """Set an error message for the widget.

        Args:
            error: Error message or None to clear
        """
        self.set_state("error", error)

    def create_horizontal_layout(self, spacing: int = 0) -> QHBoxLayout:
        """Create a horizontal layout with default settings.

        Args:
            spacing: Spacing between widgets

        Returns:
            The created layout
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        return layout

    def create_vertical_layout(self, spacing: int = 0) -> QVBoxLayout:
        """Create a vertical layout with default settings.

        Args:
            spacing: Spacing between widgets

        Returns:
            The created layout
        """
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(spacing)
        return layout

    def set_theme(self, theme_name):
        """Set the widget's theme name."""
        self._theme = theme_name
        self.theme_changed.emit(theme_name)
        self.update()

    def get_theme(self):
        """Get the current theme name."""
        return self._theme

    def update_theme(self, theme_data):
        """Update the widget's theme data.

        Args:
            theme_data: Dictionary containing theme colors and styles
        """
        self._theme_data = theme_data
        self.update()

    @pyqtProperty(dict)
    def theme(self):
        """Get the current theme data."""
        return self._theme_data

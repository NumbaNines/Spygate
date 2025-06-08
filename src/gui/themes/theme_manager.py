"""Theme manager for handling application themes."""

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QApplication, QWidget
from qt_material import apply_stylesheet, list_themes


class ThemeManager:
    """Manages application themes using qt-material."""

    def __init__(self, app: Optional[QApplication] = None):
        """Initialize the theme manager.

        Args:
            app: Optional QApplication instance
        """
        self.app = app
        self._current_theme = "dark_teal"
        self._available_themes = list_themes()
        self._custom_themes: dict[str, dict[str, Any]] = {}

    @property
    def current_theme(self) -> str:
        """Get the current theme name."""
        return self._current_theme

    @property
    def available_themes(self) -> list:
        """Get list of available themes."""
        return self._available_themes + list(self._custom_themes.keys())

    def register_theme(self, name: str, theme: dict[str, Any]) -> None:
        """Register a custom theme.

        Args:
            name: Theme name
            theme: Theme properties dictionary
        """
        self._custom_themes[name] = theme

    def apply_theme(self, widget: QWidget, theme_name: str) -> None:
        """Apply a theme to a widget.

        Args:
            widget: Widget to apply theme to
            theme_name: Name of the theme to apply
        """
        if theme_name in self._custom_themes:
            if hasattr(widget, "update_theme"):
                widget.update_theme(self._custom_themes[theme_name])
        else:
            if self.app:
                apply_stylesheet(self.app, theme=theme_name)
            else:
                apply_stylesheet(widget, theme=theme_name)
        self._current_theme = theme_name

    def apply_theme_to_app(self, theme_name: str) -> None:
        """Apply a theme to the entire application.

        Args:
            theme_name: Name of the theme to apply

        Raises:
            ValueError: If no QApplication instance was provided
        """
        if not self.app:
            raise ValueError("No QApplication instance provided")

        if theme_name in self._custom_themes:
            # Apply custom theme to all widgets
            for widget in self.app.allWidgets():
                if hasattr(widget, "update_theme"):
                    widget.update_theme(self._custom_themes[theme_name])
        else:
            apply_stylesheet(self.app, theme=theme_name)
        self._current_theme = theme_name

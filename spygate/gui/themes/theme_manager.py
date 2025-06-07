"""
Spygate Theme Manager
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet, list_themes


class ThemeManager:
    """Manages application theming including Qt Material themes and custom QSS."""

    def __init__(self):
        """Initialize the theme manager."""
        self.app = QApplication.instance()
        self.current_theme = "dark_teal"
        self.custom_styles: Dict[str, str] = {}

        # Load built-in Qt Material themes
        self.material_themes = {
            theme.replace(".xml", ""): theme for theme in list_themes()
        }

        # Create themes directory if it doesn't exist
        self.themes_dir = Path(__file__).parent / "custom"
        self.themes_dir.mkdir(exist_ok=True)

        # Load custom themes
        self._load_custom_themes()

        # Apply default theme
        self.apply_theme("dark_teal")

    def _load_custom_themes(self):
        """Load custom QSS themes from the themes directory."""
        for theme_file in self.themes_dir.glob("*.json"):
            try:
                with open(theme_file, "r") as f:
                    theme_data = json.load(f)
                    name = theme_data.get("name")
                    if name:
                        self.custom_styles[name] = theme_data.get("styles", "")
            except Exception as e:
                print(f"Error loading theme {theme_file}: {e}")

    def apply_theme(self, theme_name: str) -> bool:
        """Apply a theme to the application.

        Args:
            theme_name: Name of the theme to apply

        Returns:
            bool: True if theme was applied successfully
        """
        try:
            # Handle custom themes
            if theme_name in self.custom_styles:
                self.app.setStyleSheet(self.custom_styles[theme_name])
                self.current_theme = theme_name
                return True

            # Handle Qt Material themes
            if theme_name in self.material_themes:
                apply_stylesheet(
                    self.app,
                    theme=self.material_themes[theme_name],
                    invert_secondary=True,
                )
                self.current_theme = theme_name
                return True

            # If theme not found, try applying as a direct Qt Material theme name
            try:
                apply_stylesheet(self.app, theme=theme_name)
                self.current_theme = theme_name
                return True
            except Exception:
                pass

            return False
        except Exception as e:
            print(f"Error applying theme {theme_name}: {e}")
            return False

    def save_custom_theme(self, name: str, styles: str) -> bool:
        """Save a custom theme to disk.

        Args:
            name: Name of the theme
            styles: QSS styles for the theme

        Returns:
            bool: True if theme was saved successfully
        """
        try:
            theme_data = {"name": name, "styles": styles}

            file_path = self.themes_dir / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(theme_data, f, indent=2)

            self.custom_styles[name] = styles
            return True
        except Exception as e:
            print(f"Error saving theme {name}: {e}")
            return False

    def get_available_themes(self) -> Dict[str, str]:
        """Get all available themes.

        Returns:
            Dict[str, str]: Dictionary of theme names and their types
        """
        themes = {name: "material" for name in self.material_themes.keys()}

        themes.update({name: "custom" for name in self.custom_styles.keys()})

        return themes

    def get_current_theme(self) -> str:
        """Get the name of the current theme.

        Returns:
            str: Name of the current theme
        """
        return self.current_theme

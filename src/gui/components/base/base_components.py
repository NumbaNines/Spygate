"""Base GUI components for Spygate."""

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QDialog, QFrame, QVBoxLayout, QWidget


class BaseWidget(QWidget):
    """Base widget class."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize base widget."""
        super().__init__(parent)
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

    def update_theme(self, theme: dict[str, Any]):
        """Update widget theme."""
        style = []
        if "background" in theme:
            style.append(f"background-color: {theme['background']};")
        if "color" in theme:
            style.append(f"color: {theme['color']};")
        if "border" in theme:
            style.append(f"border: {theme['border']};")
        if "radius" in theme:
            style.append(f"border-radius: {theme['radius']}px;")
        self.setStyleSheet(" ".join(style))


class BaseDialog(QDialog):
    """Base dialog class."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize base dialog."""
        super().__init__(parent)
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(8)

    def update_theme(self, theme: dict[str, Any]):
        """Update dialog theme."""
        style = []
        if "background" in theme:
            style.append(f"background-color: {theme['background']};")
        if "color" in theme:
            style.append(f"color: {theme['color']};")
        if "border" in theme:
            style.append(f"border: {theme['border']};")
        if "radius" in theme:
            style.append(f"border-radius: {theme['radius']}px;")
        self.setStyleSheet(" ".join(style))


class BasePanel(QFrame):
    """Base panel class."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize base panel."""
        super().__init__(parent)
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(4)

    def update_theme(self, theme: dict[str, Any]):
        """Update panel theme."""
        style = []
        if "background" in theme:
            style.append(f"background-color: {theme['background']};")
        if "color" in theme:
            style.append(f"color: {theme['color']};")
        if "border" in theme:
            style.append(f"border: {theme['border']};")
        if "radius" in theme:
            style.append(f"border-radius: {theme['radius']}px;")
        self.setStyleSheet(" ".join(style))

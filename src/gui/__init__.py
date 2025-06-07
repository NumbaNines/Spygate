"""Spygate GUI package."""

from .components.base import BaseDialog, BasePanel, BaseWidget
from .components.composite import Card, Dialog, FormGroup, NavigationBar
from .themes import ThemeManager

__all__ = [
    # Base components
    "BaseWidget",
    "BaseDialog",
    "BasePanel",
    # Composite components
    "Card",
    "Dialog",
    "FormGroup",
    "NavigationBar",
    # Theme management
    "ThemeManager",
]

"""Navigation bar component for application navigation."""

from typing import Any, Dict, List, Literal, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..base import BaseWidget


class NavigationItem(QPushButton):
    """Navigation item with icon, text and badge support."""

    def __init__(self, id_: str, text: str, icon_path: Optional[str] = None):
        """Initialize navigation item.

        Args:
            id_: Item ID
            text: Item text
            icon_path: Optional path to icon
        """
        super().__init__()
        self.id = id_
        self.setText(text)
        if icon_path:
            self.setIcon(QIcon(icon_path))
        self._badge: Optional[str] = None
        self._badge_label: Optional[QLabel] = None
        self.setCheckable(True)
        self.setAutoExclusive(True)

    def set_badge(self, text: Optional[str]) -> None:
        """Set badge text.

        Args:
            text: Badge text or None to hide badge
        """
        if text:
            if not self._badge_label:
                self._badge_label = QLabel(text)
                self._badge_label.setStyleSheet(
                    "background: red; color: white; border-radius: 10px; padding: 2px 6px;"
                )
                # Add badge to layout
        else:
            if self._badge_label:
                self._badge_label.hide()
        self._badge = text

    def get_badge(self) -> Optional[str]:
        """Get badge text.

        Returns:
            Current badge text or None if no badge
        """
        return self._badge


class NavigationBar(BaseWidget):
    """A navigation bar component."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the navigation bar.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._items: Dict[str, NavigationItem] = {}
        self._selected_item: Optional[str] = None
        self._orientation = "horizontal"
        self._style = "light"
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the navigation bar UI."""
        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(8, 8, 8, 8)
        self.layout().setSpacing(8)

    def add_item(
        self, id_: str, text: str, icon_path: Optional[str] = None
    ) -> NavigationItem:
        """Add an item to the navigation bar.

        Args:
            id_: Item ID
            text: Item text
            icon_path: Optional path to icon

        Returns:
            The created navigation item
        """
        item = NavigationItem(id_, text, icon_path)
        item.clicked.connect(lambda: self.select_item(id_))
        self._items[id_] = item
        self.layout().addWidget(item)
        return item

    def items(self) -> List[QWidget]:
        """Get all navigation items.

        Returns:
            List of navigation items
        """
        return list(self._items.values())

    def get_item(self, id_: str) -> Optional[NavigationItem]:
        """Get an item by ID.

        Args:
            id_: Item ID

        Returns:
            The navigation item or None if not found
        """
        return self._items.get(id_)

    def remove_item(self, id_: str) -> None:
        """Remove an item.

        Args:
            id_: Item ID

        Raises:
            ValueError: If item not found
        """
        if id_ not in self._items:
            raise ValueError(f"Item '{id_}' not found")
        item = self._items.pop(id_)
        self.layout().removeWidget(item)
        item.deleteLater()
        if self._selected_item == id_:
            self._selected_item = None

    def update_item(
        self, id_: str, text: Optional[str] = None, icon_path: Optional[str] = None
    ) -> None:
        """Update an item.

        Args:
            id_: Item ID
            text: Optional new text
            icon_path: Optional new icon path

        Raises:
            ValueError: If item not found
        """
        item = self.get_item(id_)
        if not item:
            raise ValueError(f"Item '{id_}' not found")
        if text:
            item.setText(text)
        if icon_path:
            item.setIcon(QIcon(icon_path))

    def select_item(self, id_: str) -> None:
        """Select an item.

        Args:
            id_: Item ID

        Raises:
            ValueError: If item not found
        """
        if id_ not in self._items:
            raise ValueError(f"Item '{id_}' not found")
        self._items[id_].setChecked(True)
        self._selected_item = id_

    def selected_item(self) -> Optional[str]:
        """Get selected item ID.

        Returns:
            Selected item ID or None if none selected
        """
        return self._selected_item

    def select_next(self) -> None:
        """Select next item."""
        items = list(self._items.keys())
        if not items:
            return
        if not self._selected_item:
            self.select_item(items[0])
        else:
            try:
                current_idx = items.index(self._selected_item)
                next_idx = (current_idx + 1) % len(items)
                self.select_item(items[next_idx])
            except ValueError:
                self.select_item(items[0])

    def select_previous(self) -> None:
        """Select previous item."""
        items = list(self._items.keys())
        if not items:
            return
        if not self._selected_item:
            self.select_item(items[-1])
        else:
            try:
                current_idx = items.index(self._selected_item)
                prev_idx = (current_idx - 1) % len(items)
                self.select_item(items[prev_idx])
            except ValueError:
                self.select_item(items[-1])

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]) -> None:
        """Set navigation bar orientation.

        Args:
            orientation: Either "horizontal" or "vertical"

        Raises:
            ValueError: If invalid orientation
        """
        if orientation not in ["horizontal", "vertical"]:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'")

        self._orientation = orientation
        old_layout = self.layout()

        if orientation == "horizontal":
            new_layout = QHBoxLayout()
        else:
            new_layout = QVBoxLayout()

        new_layout.setContentsMargins(8, 8, 8, 8)
        new_layout.setSpacing(8)

        # Move items to new layout
        while old_layout.count():
            item = old_layout.takeAt(0)
            if item.widget():
                new_layout.addWidget(item.widget())

        # Delete old layout and set new one
        old_layout.deleteLater()
        self.setLayout(new_layout)

    def orientation(self) -> str:
        """Get current orientation.

        Returns:
            Current orientation ("horizontal" or "vertical")
        """
        return self._orientation

    def set_style(self, style: Literal["light", "dark"]) -> None:
        """Set navigation bar style.

        Args:
            style: Either "light" or "dark"

        Raises:
            ValueError: If invalid style
        """
        if style not in ["light", "dark"]:
            raise ValueError("Style must be 'light' or 'dark'")
        self._style = style
        self._update_style()

    def style(self) -> str:
        """Get current style.

        Returns:
            Current style ("light" or "dark")
        """
        return self._style

    def _update_style(self) -> None:
        """Update navigation bar style."""
        if self._style == "light":
            self.setStyleSheet(
                """
                QPushButton {
                    background: white;
                    color: black;
                    border: none;
                    padding: 8px;
                }
                QPushButton:checked {
                    background: #e0e0e0;
                }
            """
            )
        else:
            self.setStyleSheet(
                """
                QPushButton {
                    background: #333333;
                    color: white;
                    border: none;
                    padding: 8px;
                }
                QPushButton:checked {
                    background: #555555;
                }
            """
            )

    def set_badge(self, id_: str, text: Optional[str]) -> None:
        """Set badge text for an item.

        Args:
            id_: Item ID
            text: Badge text or None to hide badge

        Raises:
            ValueError: If item not found
        """
        item = self.get_item(id_)
        if not item:
            raise ValueError(f"Item '{id_}' not found")
        item.set_badge(text)

    def get_badge(self, id_: str) -> Optional[str]:
        """Get badge text for an item.

        Args:
            id_: Item ID

        Returns:
            Badge text or None if no badge

        Raises:
            ValueError: If item not found
        """
        item = self.get_item(id_)
        if not item:
            raise ValueError(f"Item '{id_}' not found")
        return item.get_badge()

    def update_theme(self, theme: Dict[str, Any]) -> None:
        """Update the navigation bar's theme.

        Args:
            theme: Theme dictionary with style properties
        """
        super().update_theme(theme)
        for item in self._items.values():
            if hasattr(item, "update_theme"):
                item.update_theme(theme)

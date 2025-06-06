from typing import Optional, List, Dict, Callable
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout,
    QVBoxLayout, QLabel, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

from ..base.base_widget import BaseWidget
from ...layouts.layout_manager import LayoutManager
from ...layouts.layout_config import LayoutConfig

class NavigationItem:
    """Configuration for a navigation item."""
    def __init__(
        self,
        text: str,
        icon: Optional[QIcon] = None,
        badge_text: str = "",
        badge_color: str = "#dc3545",  # Default to red
        callback: Optional[Callable[[], None]] = None,
        tooltip: str = ""
    ):
        self.text = text
        self.icon = icon
        self.badge_text = badge_text
        self.badge_color = badge_color
        self.callback = callback
        self.tooltip = tooltip or text

class NavigationBar(BaseWidget):
    """A navigation component for switching between views."""
    
    # Signals
    item_clicked = pyqtSignal(str)  # Emits the text of the clicked item
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        items: List[NavigationItem] = None,
        orientation: str = "horizontal",  # horizontal, vertical
        layout_config: Optional[LayoutConfig] = None,
        selected_item: str = "",
        style: str = "default"  # default, tabs, pills
    ):
        super().__init__(parent)
        self.items = items or []
        self.orientation = orientation
        self.layout_config = layout_config or LayoutConfig(
            margins=(0, 0, 0, 0),
            spacing=5
        )
        self.selected_item = selected_item
        self.style = style
        self.item_widgets: Dict[str, QPushButton] = {}
        self.badge_widgets: Dict[str, QLabel] = {}
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Initialize the navigation bar layout and items."""
        # Main layout
        if self.orientation == "horizontal":
            main_layout = LayoutManager.create_horizontal(self, self.layout_config)
        else:
            main_layout = LayoutManager.create_vertical(self, self.layout_config)
        
        # Style the navigation bar
        if self.style == "tabs":
            self.setStyleSheet("""
                NavigationBar {
                    border-bottom: 1px solid palette(mid);
                }
            """)
        elif self.style == "pills":
            self.setStyleSheet("""
                NavigationBar {
                    background-color: palette(base);
                    border-radius: 4px;
                }
            """)
        
        # Add items
        for item in self.items:
            # Create button container for badge positioning
            container = QWidget(self)
            if self.orientation == "horizontal":
                container_layout = QHBoxLayout(container)
            else:
                container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(5)
            
            # Create button
            btn = QPushButton(item.text, container)
            if item.icon:
                btn.setIcon(item.icon)
            btn.setToolTip(item.tooltip)
            
            # Style the button based on navigation style
            if self.style == "tabs":
                btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        border-bottom: 2px solid transparent;
                        padding: 8px 16px;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: palette(alternate-base);
                    }
                    QPushButton[selected="true"] {
                        border-bottom: 2px solid palette(highlight);
                        color: palette(highlight);
                    }
                """)
            elif self.style == "pills":
                btn.setStyleSheet("""
                    QPushButton {
                        border: none;
                        border-radius: 4px;
                        padding: 8px 16px;
                        background: transparent;
                    }
                    QPushButton:hover {
                        background-color: palette(alternate-base);
                    }
                    QPushButton[selected="true"] {
                        background-color: palette(highlight);
                        color: palette(bright-text);
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        padding: 8px 16px;
                    }
                    QPushButton[selected="true"] {
                        background-color: palette(highlight);
                        color: palette(bright-text);
                    }
                """)
            
            # Set selected state
            if item.text == self.selected_item:
                btn.setProperty("selected", True)
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            
            # Create badge if needed
            if item.badge_text:
                badge = QLabel(item.badge_text, container)
                badge.setStyleSheet(f"""
                    QLabel {{
                        background-color: {item.badge_color};
                        color: white;
                        border-radius: 10px;
                        padding: 2px 6px;
                        font-size: 10px;
                    }}
                """)
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                container_layout.addWidget(badge)
                self.badge_widgets[item.text] = badge
            
            # Connect signals
            def create_click_handler(item_text: str, callback: Optional[Callable[[], None]]):
                def handle_click():
                    self.set_selected(item_text)
                    if callback:
                        callback()
                    self.item_clicked.emit(item_text)
                return handle_click
            
            btn.clicked.connect(create_click_handler(item.text, item.callback))
            
            # Store widget reference
            self.item_widgets[item.text] = btn
            
            # Add to layout
            container_layout.addWidget(btn)
            main_layout.addWidget(container)
        
        # Add stretch at the end
        if self.orientation == "horizontal":
            main_layout.addStretch()
    
    def set_selected(self, text: str) -> None:
        """Set the selected item."""
        if text not in self.item_widgets:
            return
            
        # Update old selection
        if self.selected_item in self.item_widgets:
            old_btn = self.item_widgets[self.selected_item]
            old_btn.setProperty("selected", False)
            old_btn.style().unpolish(old_btn)
            old_btn.style().polish(old_btn)
        
        # Update new selection
        self.selected_item = text
        new_btn = self.item_widgets[text]
        new_btn.setProperty("selected", True)
        new_btn.style().unpolish(new_btn)
        new_btn.style().polish(new_btn)
    
    def set_badge_text(self, item_text: str, badge_text: str) -> None:
        """Update the badge text for an item."""
        if item_text in self.badge_widgets:
            self.badge_widgets[item_text].setText(badge_text)
    
    def set_badge_color(self, item_text: str, color: str) -> None:
        """Update the badge color for an item."""
        if item_text in self.badge_widgets:
            badge = self.badge_widgets[item_text]
            badge.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 10px;
                    padding: 2px 6px;
                    font-size: 10px;
                }}
            """) 
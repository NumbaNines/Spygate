from typing import Optional, List, Callable, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..base.base_widget import BaseWidget
from ...layouts.layout_manager import LayoutManager
from ...layouts.layout_config import LayoutConfig

class DialogButton:
    """Configuration for a dialog button."""
    def __init__(
        self,
        text: str,
        role: str = "default",  # default, primary, danger
        callback: Optional[Callable[[], Any]] = None,
        close_on_click: bool = True
    ):
        self.text = text
        self.role = role
        self.callback = callback
        self.close_on_click = close_on_click

class Dialog(QDialog, BaseWidget):
    """A customizable modal dialog component."""
    
    # Signals
    closed = pyqtSignal(str)  # Emits the role of the button that closed the dialog
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        content: Optional[QWidget] = None,
        buttons: List[DialogButton] = None,
        layout_config: Optional[LayoutConfig] = None,
        modal: bool = True,
        width: int = 400,
        height: Optional[int] = None
    ):
        super().__init__(parent)
        BaseWidget.__init__(self, parent)
        
        self.dialog_title = title
        self.content = content
        self.buttons = buttons or [
            DialogButton("OK", "primary"),
            DialogButton("Cancel", "default")
        ]
        self.layout_config = layout_config or LayoutConfig(
            margins=(20, 20, 20, 20),
            spacing=15
        )
        
        # Configure dialog properties
        self.setModal(modal)
        self.setMinimumWidth(width)
        if height:
            self.setMinimumHeight(height)
        
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Initialize the dialog layout and components."""
        # Main layout
        main_layout = LayoutManager.create_vertical(self, self.layout_config)
        
        # Title
        if self.dialog_title:
            title_label = QLabel(self.dialog_title, self)
            title_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: palette(text);
                }
            """)
            main_layout.addWidget(title_label)
            
            # Add separator
            separator = QFrame(self)
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            main_layout.addWidget(separator)
        
        # Content
        if self.content:
            self.content.setParent(self)
            main_layout.addWidget(self.content)
        
        # Buttons
        if self.buttons:
            button_container = QWidget(self)
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.setSpacing(10)
            button_layout.addStretch()
            
            for button in self.buttons:
                btn = QPushButton(button.text, button_container)
                
                # Style based on role
                if button.role == "primary":
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: palette(highlight);
                            color: palette(bright-text);
                            min-width: 80px;
                            padding: 8px 16px;
                        }
                        QPushButton:hover {
                            background-color: palette(dark);
                        }
                    """)
                elif button.role == "danger":
                    btn.setStyleSheet("""
                        QPushButton {
                            background-color: #dc3545;
                            color: white;
                            min-width: 80px;
                            padding: 8px 16px;
                        }
                        QPushButton:hover {
                            background-color: #c82333;
                        }
                    """)
                else:
                    btn.setStyleSheet("""
                        QPushButton {
                            min-width: 80px;
                            padding: 8px 16px;
                        }
                    """)
                
                # Connect button actions
                def create_click_handler(button: DialogButton):
                    def handle_click():
                        if button.callback:
                            button.callback()
                        if button.close_on_click:
                            self.close()
                            self.closed.emit(button.role)
                    return handle_click
                
                btn.clicked.connect(create_click_handler(button))
                button_layout.addWidget(btn)
            
            main_layout.addWidget(button_container)
    
    def set_content(self, content: QWidget) -> None:
        """Set or update the dialog's content widget."""
        if self.content:
            self.content.deleteLater()
        
        self.content = content
        self.content.setParent(self)
        layout = self.findChild(QVBoxLayout)
        
        # Insert before buttons (last widget)
        layout.insertWidget(layout.count() - 1, content)
    
    def get_button(self, text: str) -> Optional[QPushButton]:
        """Get a button widget by its text."""
        for button in self.findChildren(QPushButton):
            if button.text() == text:
                return button
        return None
    
    def set_button_enabled(self, text: str, enabled: bool) -> None:
        """Enable or disable a button by its text."""
        button = self.get_button(text)
        if button:
            button.setEnabled(enabled)
    
    def set_button_text(self, old_text: str, new_text: str) -> None:
        """Update a button's text."""
        button = self.get_button(old_text)
        if button:
            button.setText(new_text) 
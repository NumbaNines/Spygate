from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QLabel, QFrame, QVBoxLayout,
    QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt

from ..base.base_widget import BaseWidget
from ...layouts.layout_manager import LayoutManager
from ...layouts.layout_config import LayoutConfig

class Card(BaseWidget):
    """A themed container component with header, content, and optional footer."""
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        content: Optional[QWidget] = None,
        footer: Optional[QWidget] = None,
        layout_config: Optional[LayoutConfig] = None,
        collapsible: bool = False,
        elevated: bool = True
    ):
        super().__init__(parent)
        self.title = title
        self.content = content
        self.footer = footer
        self.layout_config = layout_config or LayoutConfig(
            margins=(15, 15, 15, 15),
            spacing=10
        )
        self.collapsible = collapsible
        self.elevated = elevated
        self.collapsed = False
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Initialize the card layout and components."""
        # Main layout
        main_layout = LayoutManager.create_vertical(self, self.layout_config)
        
        # Set up card frame
        self.setFrameStyle(QFrame.Shape.Box)
        if self.elevated:
            self.setStyleSheet("""
                Card {
                    background-color: palette(base);
                    border: 1px solid palette(mid);
                    border-radius: 4px;
                    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                }
            """)
        
        # Header
        if self.title or self.collapsible:
            header = QWidget(self)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            
            # Title
            title_label = QLabel(self.title, header)
            title_label.setStyleSheet("font-weight: bold;")
            header_layout.addWidget(title_label)
            
            # Collapse button
            if self.collapsible:
                self.collapse_btn = QPushButton("▼" if not self.collapsed else "▶", header)
                self.collapse_btn.setFixedSize(20, 20)
                self.collapse_btn.clicked.connect(self.toggle_collapse)
                header_layout.addWidget(self.collapse_btn, alignment=Qt.AlignmentFlag.AlignRight)
            
            main_layout.addWidget(header)
        
        # Content
        if self.content:
            self.content.setParent(self)
            main_layout.addWidget(self.content)
        
        # Footer
        if self.footer:
            self.footer.setParent(self)
            main_layout.addWidget(self.footer)
    
    def toggle_collapse(self) -> None:
        """Toggle the collapsed state of the card."""
        if not self.collapsible or not self.content:
            return
            
        self.collapsed = not self.collapsed
        self.content.setVisible(not self.collapsed)
        self.collapse_btn.setText("▼" if not self.collapsed else "▶")
        
        if self.footer:
            self.footer.setVisible(not self.collapsed)
    
    def set_content(self, content: QWidget) -> None:
        """Set or update the card's content widget."""
        if self.content:
            self.content.deleteLater()
        
        self.content = content
        self.content.setParent(self)
        layout = self.findChild(QVBoxLayout)
        layout.insertWidget(1 if self.title else 0, content)
        
        if self.collapsed:
            content.hide()
    
    def set_footer(self, footer: QWidget) -> None:
        """Set or update the card's footer widget."""
        if self.footer:
            self.footer.deleteLater()
        
        self.footer = footer
        self.footer.setParent(self)
        layout = self.findChild(QVBoxLayout)
        layout.addWidget(footer)
        
        if self.collapsed:
            footer.hide() 
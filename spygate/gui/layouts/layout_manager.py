"""
Layout Management System
"""

from typing import Optional, Tuple, Union
from PyQt6.QtWidgets import (
    QWidget,
    QLayout,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedLayout,
    QSizePolicy,
    QSpacerItem,
    QSplitter
)
from PyQt6.QtCore import Qt, QSize
from .layout_config import LayoutConfig


class LayoutManager:
    """Manages layout creation and configuration."""
    
    def __init__(self, container: QWidget):
        """Initialize layout manager.
        
        Args:
            container: Widget to manage layouts for
        """
        self.container = container
        self.grid_layout = None
        self.flow_layout = None
        self.stack_layout = None
    
    @staticmethod
    def create_vertical(
        parent: Optional[QWidget] = None,
        margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
        spacing: int = 0,
        config: Optional[LayoutConfig] = None,
    ) -> QVBoxLayout:
        """Create a vertical layout with specified settings.
        
        Args:
            parent: Optional parent widget
            margins: Tuple of (left, top, right, bottom) margins
            spacing: Spacing between widgets
            config: Optional layout configuration
            
        Returns:
            The configured vertical layout
        """
        layout = QVBoxLayout(parent) if parent else QVBoxLayout()
        if config:
            if config.margins:
                margins = config.margins
            spacing = config.spacing
            if config.alignment:
                layout.setAlignment(config.alignment)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        return layout
    
    @staticmethod
    def create_horizontal(
        parent: Optional[QWidget] = None,
        margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
        spacing: int = 0,
        config: Optional[LayoutConfig] = None,
    ) -> QHBoxLayout:
        """Create a horizontal layout with specified settings.
        
        Args:
            parent: Optional parent widget
            margins: Tuple of (left, top, right, bottom) margins
            spacing: Spacing between widgets
            config: Optional layout configuration
            
        Returns:
            The configured horizontal layout
        """
        layout = QHBoxLayout(parent) if parent else QHBoxLayout()
        if config:
            if config.margins:
                margins = config.margins
            spacing = config.spacing
            if config.alignment:
                layout.setAlignment(config.alignment)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)
        return layout
    
    def create_vertical_layout(self, spacing: int = 0) -> QVBoxLayout:
        """Create a vertical layout.
        
        Args:
            spacing: Space between widgets
            
        Returns:
            The created vertical layout
        """
        layout = QVBoxLayout()
        layout.setSpacing(spacing)
        return layout
    
    def create_horizontal_layout(self, spacing: int = 0) -> QHBoxLayout:
        """Create a horizontal layout.
        
        Args:
            spacing: Space between widgets
            
        Returns:
            The created horizontal layout
        """
        layout = QHBoxLayout()
        layout.setSpacing(spacing)
        return layout
    
    def create_grid_layout(self, spacing: int = 0) -> QGridLayout:
        """Create a grid layout.
        
        Args:
            spacing: Space between widgets
            
        Returns:
            The created grid layout
        """
        layout = QGridLayout()
        layout.setSpacing(spacing)
        return layout
    
    def create_stacked_layout(self) -> QStackedLayout:
        """Create a stacked layout.
        
        Returns:
            The created stacked layout
        """
        return QStackedLayout()
    
    def add_widget(
        self,
        layout: QLayout,
        widget: QWidget,
        stretch: int = 0,
        alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft
    ) -> None:
        """Add a widget to a layout.
        
        Args:
            layout: Layout to add widget to
            widget: Widget to add
            stretch: Stretch factor
            alignment: Widget alignment
        """
        if isinstance(layout, (QVBoxLayout, QHBoxLayout)):
            layout.addWidget(widget, stretch, alignment)
        elif isinstance(layout, QGridLayout):
            # For grid layouts, we need row and column
            # This is a simplified version that adds to the next available cell
            row = layout.rowCount()
            col = layout.columnCount()
            layout.addWidget(widget, row, col, alignment)
        elif isinstance(layout, QStackedLayout):
            layout.addWidget(widget)
    
    def add_layout(
        self,
        parent_layout: QLayout,
        child_layout: QLayout,
        stretch: int = 0
    ) -> None:
        """Add a layout to another layout.
        
        Args:
            parent_layout: Layout to add to
            child_layout: Layout to add
            stretch: Stretch factor
        """
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
            parent_layout.addLayout(child_layout, stretch)
        elif isinstance(parent_layout, QGridLayout):
            row = parent_layout.rowCount()
            col = parent_layout.columnCount()
            parent_layout.addLayout(child_layout, row, col)
    
    def apply_layout_config(self, widget: QWidget, config: LayoutConfig) -> None:
        """Apply layout configuration to a widget.
        
        Args:
            widget: Widget to configure
            config: Layout configuration
        """
        if config.min_size:
            if isinstance(config.min_size, tuple):
                widget.setMinimumSize(*config.min_size)
            else:
                widget.setMinimumSize(config.min_size)
        if config.max_size:
            if isinstance(config.max_size, tuple):
                widget.setMaximumSize(*config.max_size)
            else:
                widget.setMaximumSize(config.max_size)
        if config.margins and widget.layout():
            widget.layout().setContentsMargins(*config.margins)
        if widget.layout():
            widget.layout().setSpacing(config.spacing)
        widget.setSizePolicy(config.horizontal_policy, config.vertical_policy)
        if config.alignment and widget.layout():
            widget.layout().setAlignment(config.alignment)
    
    def add_to_grid(self, widget: QWidget, row: int, col: int, rowspan: int = 1, colspan: int = 1) -> None:
        """Add a widget to the grid layout at specified position.
        
        Args:
            widget: Widget to add
            row: Row index
            col: Column index
            rowspan: Number of rows to span
            colspan: Number of columns to span
        """
        if not self.grid_layout:
            self.grid_layout = self.create_grid_layout()
            self.container.setLayout(self.grid_layout)
        self.grid_layout.addWidget(widget, row, col, rowspan, colspan)
    
    def add_to_flow(self, widget: QWidget) -> None:
        """Add a widget to the flow layout.
        
        Args:
            widget: Widget to add
        """
        if not self.flow_layout:
            self.flow_layout = self.create_horizontal_layout()
            self.container.setLayout(self.flow_layout)
        self.flow_layout.addWidget(widget)
    
    def add_to_stack(self, widget: QWidget) -> None:
        """Add a widget to the stack layout.
        
        Args:
            widget: Widget to add
        """
        if not self.stack_layout:
            self.stack_layout = self.create_stacked_layout()
            self.container.setLayout(self.stack_layout)
        self.stack_layout.addWidget(widget)
    
    def switch_to_stack_index(self, index: int) -> None:
        """Switch to a specific widget in the stack layout.
        
        Args:
            index: Index of the widget to show
        """
        if self.stack_layout:
            self.stack_layout.setCurrentIndex(index)
    
    def configure_grid(self, config: LayoutConfig) -> None:
        """Configure the grid layout.
        
        Args:
            config: Layout configuration
        """
        if not self.grid_layout:
            self.grid_layout = self.create_grid_layout()
            self.container.setLayout(self.grid_layout)
        self.grid_layout.setSpacing(config.spacing)
        if config.margins:
            self.grid_layout.setContentsMargins(*config.margins)
        if config.alignment:
            self.grid_layout.setAlignment(config.alignment)
    
    def configure_flow(self, config: LayoutConfig) -> None:
        """Configure the flow layout.
        
        Args:
            config: Layout configuration
        """
        if not self.flow_layout:
            self.flow_layout = self.create_horizontal_layout()
            self.container.setLayout(self.flow_layout)
        self.flow_layout.setSpacing(config.spacing)
        if config.margins:
            self.flow_layout.setContentsMargins(*config.margins)
        if config.alignment:
            self.flow_layout.setAlignment(config.alignment)
    
    def configure_stack(self, config: LayoutConfig) -> None:
        """Configure the stack layout.
        
        Args:
            config: Layout configuration
        """
        if not self.stack_layout:
            self.stack_layout = self.create_stacked_layout()
            self.container.setLayout(self.stack_layout)
        self.stack_layout.setSpacing(config.spacing)
        if config.margins:
            self.stack_layout.setContentsMargins(*config.margins)
        if config.alignment:
            self.stack_layout.setAlignment(config.alignment)
    
    @staticmethod
    def create_stack(parent: Optional[QWidget] = None) -> QStackedLayout:
        """Create a stacked layout.
        
        Args:
            parent: Optional parent widget
            
        Returns:
            QStackedLayout: The created stacked layout
        """
        layout = QStackedLayout()
        if parent:
            parent.setLayout(layout)
        return layout
    
    @staticmethod
    def create_horizontal(parent: Optional[QWidget] = None, config: Optional[LayoutConfig] = None) -> QHBoxLayout:
        """Create a horizontal layout.
        
        Args:
            parent: Optional parent widget
            config: Optional layout configuration
            
        Returns:
            QHBoxLayout: The created horizontal layout
        """
        layout = QHBoxLayout()
        if parent:
            parent.setLayout(layout)
        if config:
            if config.margins:
                layout.setContentsMargins(*config.margins)
            if config.spacing:
                layout.setSpacing(config.spacing)
            if config.alignment:
                layout.setAlignment(config.alignment)
        return layout
    
    @staticmethod
    def create_vertical(parent: Optional[QWidget] = None, config: Optional[LayoutConfig] = None) -> QVBoxLayout:
        """Create a vertical layout.
        
        Args:
            parent: Optional parent widget
            config: Optional layout configuration
            
        Returns:
            QVBoxLayout: The created vertical layout
        """
        layout = QVBoxLayout()
        if parent:
            parent.setLayout(layout)
        if config:
            if config.margins:
                layout.setContentsMargins(*config.margins)
            if config.spacing:
                layout.setSpacing(config.spacing)
            if config.alignment:
                layout.setAlignment(config.alignment)
        return layout
    
    @staticmethod
    def create_grid(parent: Optional[QWidget] = None, config: Optional[LayoutConfig] = None) -> QGridLayout:
        """Create a grid layout.
        
        Args:
            parent: Optional parent widget
            config: Optional layout configuration
            
        Returns:
            QGridLayout: The created grid layout
        """
        layout = QGridLayout()
        if parent:
            parent.setLayout(layout)
        if config:
            if config.margins:
                layout.setContentsMargins(*config.margins)
            if config.spacing:
                layout.setSpacing(config.spacing)
            if config.alignment:
                layout.setAlignment(config.alignment)
        return layout 
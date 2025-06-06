"""
Layout Manager Tests
"""

import pytest
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
from spygate.gui.layouts import LayoutManager, LayoutConfig

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def container(app, qtbot):
    """Create a container widget."""
    container = QWidget()
    qtbot.addWidget(container)
    return container

@pytest.fixture
def layout_manager(container):
    """Create a LayoutManager instance."""
    return LayoutManager(container)

def test_layout_manager_creation(layout_manager, container):
    """Test that LayoutManager is created properly."""
    assert layout_manager is not None
    assert isinstance(layout_manager, LayoutManager)
    assert layout_manager.container == container

def test_vertical_layout_creation(layout_manager):
    """Test creating a vertical layout."""
    layout = layout_manager.create_vertical_layout(spacing=10)
    assert isinstance(layout, QVBoxLayout)
    assert layout.spacing() == 10

def test_horizontal_layout_creation(layout_manager):
    """Test creating a horizontal layout."""
    layout = layout_manager.create_horizontal_layout(spacing=8)
    assert isinstance(layout, QHBoxLayout)
    assert layout.spacing() == 8

def test_layout_config_application(layout_manager, container):
    """Test applying layout configuration."""
    config = LayoutConfig(
        min_size=(100, 100),
        max_size=(500, 500),
        margins=(10, 10, 10, 10),
        spacing=5
    )
    
    layout_manager.apply_layout_config(container, config)
    assert container.minimumSize().width() == 100
    assert container.minimumSize().height() == 100
    assert container.maximumSize().width() == 500
    assert container.maximumSize().height() == 500

def test_widget_addition(layout_manager, container):
    """Test adding widgets to layouts."""
    layout = layout_manager.create_vertical_layout()
    container.setLayout(layout)
    
    # Add some widgets
    label1 = QLabel("Test 1")
    label2 = QLabel("Test 2")
    layout_manager.add_widget(layout, label1)
    layout_manager.add_widget(layout, label2)
    
    assert layout.count() == 2
    assert layout.itemAt(0).widget() == label1
    assert layout.itemAt(1).widget() == label2

def test_layout_nesting(layout_manager, container):
    """Test nesting layouts."""
    main_layout = layout_manager.create_vertical_layout()
    container.setLayout(main_layout)
    
    # Create nested horizontal layout
    nested_layout = layout_manager.create_horizontal_layout()
    layout_manager.add_layout(main_layout, nested_layout)
    
    # Add widgets to nested layout
    label1 = QLabel("Nested 1")
    label2 = QLabel("Nested 2")
    layout_manager.add_widget(nested_layout, label1)
    layout_manager.add_widget(nested_layout, label2)
    
    assert main_layout.count() == 1
    assert isinstance(main_layout.itemAt(0).layout(), QHBoxLayout)
    assert nested_layout.count() == 2

def test_layout_stretching(layout_manager, container):
    """Test layout stretching configuration."""
    layout = layout_manager.create_horizontal_layout()
    container.setLayout(layout)
    
    # Add widgets with different stretch factors
    label1 = QLabel("Stretch 1")
    label2 = QLabel("Stretch 2")
    layout_manager.add_widget(layout, label1, stretch=1)
    layout_manager.add_widget(layout, label2, stretch=2)
    
    assert layout.stretch(0) == 1
    assert layout.stretch(1) == 2

def test_layout_alignment(layout_manager, container):
    """Test widget alignment in layouts."""
    layout = layout_manager.create_vertical_layout()
    container.setLayout(layout)
    
    label = QLabel("Aligned")
    layout_manager.add_widget(layout, label, alignment=Qt.AlignmentFlag.AlignRight)
    
    assert layout.itemAt(0).alignment() == Qt.AlignmentFlag.AlignRight

def test_grid_layout(layout_manager):
    """Test grid layout functionality."""
    # Create test widgets
    widget1 = QLabel("Widget 1")
    widget2 = QLabel("Widget 2")
    widget3 = QLabel("Widget 3")
    widget4 = QLabel("Widget 4")
    
    # Add widgets to grid
    layout_manager.add_to_grid(widget1, 0, 0)
    layout_manager.add_to_grid(widget2, 0, 1)
    layout_manager.add_to_grid(widget3, 1, 0)
    layout_manager.add_to_grid(widget4, 1, 1)
    
    # Verify grid positions
    assert layout_manager.grid_layout.itemAtPosition(0, 0).widget() == widget1
    assert layout_manager.grid_layout.itemAtPosition(0, 1).widget() == widget2
    assert layout_manager.grid_layout.itemAtPosition(1, 0).widget() == widget3
    assert layout_manager.grid_layout.itemAtPosition(1, 1).widget() == widget4

def test_flow_layout(layout_manager):
    """Test flow layout functionality."""
    # Create test widgets
    widgets = [QLabel(f"Widget {i}") for i in range(5)]
    
    # Add widgets to flow layout
    for widget in widgets:
        layout_manager.add_to_flow(widget)
    
    # Verify widgets are in flow layout
    for i, widget in enumerate(widgets):
        assert layout_manager.flow_layout.itemAt(i).widget() == widget

def test_stack_layout(layout_manager):
    """Test stack layout functionality."""
    # Create test widgets
    widgets = [QLabel(f"Widget {i}") for i in range(3)]
    
    # Add widgets to stack
    for widget in widgets:
        layout_manager.add_to_stack(widget)
    
    # Verify stack operations
    assert layout_manager.stack_layout.count() == len(widgets)
    
    # Test switching between widgets
    for i in range(len(widgets)):
        layout_manager.switch_to_stack_index(i)
        assert layout_manager.stack_layout.currentIndex() == i
        assert layout_manager.stack_layout.currentWidget() == widgets[i]

def test_layout_config(layout_manager):
    """Test layout configuration."""
    config = LayoutConfig(
        margins=(10, 10, 10, 10),
        spacing=5,
        alignment=Qt.AlignmentFlag.AlignCenter
    )
    
    # Apply config to grid layout
    layout_manager.configure_grid(config)
    assert layout_manager.grid_layout.spacing() == config.spacing
    assert layout_manager.grid_layout.contentsMargins().left() == config.margins[0]
    
    # Apply config to flow layout
    layout_manager.configure_flow(config)
    assert layout_manager.flow_layout.spacing() == config.spacing
    
    # Apply config to stack layout
    layout_manager.configure_stack(config)
    assert layout_manager.stack_layout.spacing() == config.spacing

def test_responsive_behavior(layout_manager, container, qtbot):
    """Test responsive layout behavior."""
    # Add some widgets
    widgets = [QLabel(f"Widget {i}") for i in range(4)]
    for widget in widgets:
        layout_manager.add_to_grid(widget, 0, widgets.index(widget))
    
    # Test different container sizes
    sizes = [(400, 300), (800, 600), (1200, 900)]
    for width, height in sizes:
        container.resize(width, height)
        qtbot.wait(100)  # Allow layout to update
        assert container.width() == width
        assert container.height() == height
        # Add specific layout checks based on your responsive implementation 
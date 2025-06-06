import pytest
from PyQt6.QtWidgets import QApplication
from spygate.gui.components.base import BaseWidget, BaseDialog, BasePanel

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def base_widget(app, qtbot):
    """Create a BaseWidget instance."""
    widget = BaseWidget()
    qtbot.addWidget(widget)
    return widget

@pytest.fixture
def base_dialog(app, qtbot):
    """Create a BaseDialog instance."""
    dialog = BaseDialog(title="Test Dialog")
    qtbot.addWidget(dialog)
    return dialog

@pytest.fixture
def base_panel(app, qtbot):
    """Create a BasePanel instance."""
    panel = BasePanel()
    qtbot.addWidget(panel)
    return panel

def test_base_widget_creation(base_widget):
    """Test that BaseWidget is created properly."""
    assert base_widget is not None
    assert isinstance(base_widget, BaseWidget)
    assert hasattr(base_widget, "theme")

def test_base_widget_theme_update(base_widget):
    """Test BaseWidget theme update functionality."""
    test_theme = {
        "bg_primary": "#ffffff",
        "text_primary": "#000000"
    }
    base_widget.update_theme(test_theme)
    assert base_widget.theme == test_theme

def test_base_dialog_creation(base_dialog):
    """Test that BaseDialog is created properly."""
    assert base_dialog is not None
    assert isinstance(base_dialog, BaseDialog)
    assert base_dialog.windowTitle() == "Test Dialog"
    assert hasattr(base_dialog, "theme")

def test_base_dialog_theme_update(base_dialog):
    """Test BaseDialog theme update functionality."""
    test_theme = {
        "dialog_bg": "#ffffff",
        "dialog_border": "#000000"
    }
    base_dialog.update_theme(test_theme)
    assert base_dialog.theme == test_theme

def test_base_panel_creation(base_panel):
    """Test that BasePanel is created properly."""
    assert base_panel is not None
    assert isinstance(base_panel, BasePanel)
    assert hasattr(base_panel, "theme")

def test_base_panel_theme_update(base_panel):
    """Test BasePanel theme update functionality."""
    test_theme = {
        "panel_bg": "#ffffff",
        "panel_border": "#000000"
    }
    base_panel.update_theme(test_theme)
    assert base_panel.theme == test_theme

def test_base_widget_visibility(base_widget, qtbot):
    """Test BaseWidget visibility controls."""
    assert base_widget.isVisible()
    base_widget.hide()
    assert not base_widget.isVisible()
    base_widget.show()
    assert base_widget.isVisible()

def test_base_panel_layout(base_panel):
    """Test BasePanel layout management."""
    assert base_panel.layout() is not None
    # Test adding widgets would go here based on your layout implementation 
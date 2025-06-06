import pytest
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt
from spygate.gui.components.composite import Dialog

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def dialog(app, qtbot):
    """Create a Dialog instance."""
    dialog = Dialog(title="Test Dialog")
    qtbot.addWidget(dialog)
    return dialog

def test_dialog_creation(dialog):
    """Test that Dialog is created properly."""
    assert dialog is not None
    assert isinstance(dialog, Dialog)
    assert dialog.windowTitle() == "Test Dialog"

def test_set_content(dialog):
    """Test setting dialog content."""
    content = QLabel("Test Content")
    dialog.set_content(content)
    assert dialog.content is not None
    assert isinstance(dialog.content, QLabel)
    assert dialog.content.text() == "Test Content"

def test_add_buttons(dialog):
    """Test adding buttons to dialog."""
    # Add default button
    dialog.add_button("Cancel", role="default", callback=lambda: None)
    assert len(dialog.buttons) == 1
    assert dialog.buttons[0].text() == "Cancel"
    
    # Add primary button
    dialog.add_button("OK", role="primary", callback=lambda: None)
    assert len(dialog.buttons) == 2
    assert dialog.buttons[1].text() == "OK"
    
    # Add danger button
    dialog.add_button("Delete", role="danger", callback=lambda: None)
    assert len(dialog.buttons) == 3
    assert dialog.buttons[2].text() == "Delete"

def test_button_callbacks(dialog, qtbot):
    """Test button callbacks."""
    callback_triggered = False
    
    def on_click():
        nonlocal callback_triggered
        callback_triggered = True
    
    dialog.add_button("Test", callback=on_click)
    qtbot.mouseClick(dialog.buttons[0], Qt.MouseButton.LeftButton)
    
    assert callback_triggered

def test_dialog_size(dialog):
    """Test dialog size management."""
    # Test default size
    assert dialog.width() > 0
    assert dialog.height() > 0
    
    # Test custom size
    new_size = (400, 300)
    dialog.resize(*new_size)
    assert dialog.width() == new_size[0]
    assert dialog.height() == new_size[1]

def test_dialog_modal(dialog):
    """Test dialog modal property."""
    # Test default (should be modal)
    assert dialog.isModal()
    
    # Test non-modal
    dialog.setModal(False)
    assert not dialog.isModal()

def test_dialog_theme(dialog):
    """Test dialog theme awareness."""
    # Test default theme
    assert hasattr(dialog, "theme")
    
    # Test theme update
    dialog.update_theme({
        "dialog_bg": "#ffffff",
        "dialog_border": "#000000",
        "button_primary": "#007bff",
        "button_danger": "#dc3545"
    })
    # Add assertions based on your theme implementation 
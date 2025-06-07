"""Test dialog component."""

import pytest
from PyQt6.QtWidgets import QLabel

from src.gui.components.composite import Dialog


@pytest.mark.gui
def test_dialog_creation():
    """Test creating a dialog."""
    dialog = Dialog()
    assert dialog is not None
    assert dialog.objectName() == ""


@pytest.mark.gui
def test_set_content():
    """Test setting dialog content."""
    dialog = Dialog()
    content = QLabel("Test content")
    dialog.set_content(content)
    assert dialog.content() == content


@pytest.mark.gui
def test_add_buttons():
    """Test adding buttons to dialog."""
    dialog = Dialog()
    dialog.add_button("OK", lambda: None)
    dialog.add_button("Cancel", lambda: None)
    assert len(dialog.buttons()) == 2


@pytest.mark.gui
def test_button_callbacks():
    """Test button callbacks."""
    dialog = Dialog()
    callback_called = False

    def callback():
        nonlocal callback_called
        callback_called = True

    dialog.add_button("Test", callback)
    dialog.buttons()[0].click()
    assert callback_called


@pytest.mark.gui
def test_dialog_size():
    """Test dialog size control."""
    dialog = Dialog()
    dialog.resize(300, 200)
    assert dialog.width() == 300
    assert dialog.height() == 200


@pytest.mark.gui
def test_dialog_modal():
    """Test dialog modality."""
    dialog = Dialog()
    dialog.set_modal(True)
    assert dialog.isModal()


@pytest.mark.gui
def test_dialog_theme():
    """Test dialog theme awareness."""
    dialog = Dialog()
    dialog.update_theme({"background": "#ffffff"})
    assert dialog.styleSheet() != ""


@pytest.mark.gui
def test_dialog_add_content():
    """Test adding content to the dialog."""
    dialog = Dialog()
    label = QLabel("Test")
    dialog.add_content(label)
    assert label in dialog.content_widgets

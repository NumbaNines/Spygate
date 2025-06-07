"""Test card component."""

import pytest
from PyQt6.QtWidgets import QApplication, QLabel

from src.gui.components.composite import Card


@pytest.fixture
def qapp():
    """Create a QApplication instance for the tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()


@pytest.mark.gui
def test_card_creation(qapp):
    """Test creating a card."""
    card = Card()
    assert card is not None
    assert card.objectName() == ""


@pytest.mark.gui
def test_set_content(qapp):
    """Test setting card content."""
    card = Card()
    content = QLabel("Test content")
    card.set_content(content)
    assert card.content() == content


@pytest.mark.gui
def test_set_footer(qapp):
    """Test setting card footer."""
    card = Card()
    footer = QLabel("Test footer")
    card.set_footer(footer)
    assert card.footer() == footer


@pytest.mark.gui
def test_collapsible(qapp):
    """Test card collapsible behavior."""
    card = Card(collapsible=True)
    assert card.is_collapsible()
    assert card.is_expanded()
    card.collapse()
    assert not card.is_expanded()
    card.expand()
    assert card.is_expanded()


@pytest.mark.gui
def test_elevation(qapp):
    """Test card elevation."""
    card = Card()
    card.set_elevation(2)
    assert card.elevation() == 2
    assert "box-shadow" in card.styleSheet()


@pytest.mark.gui
def test_theme_awareness(qapp):
    """Test card theme awareness."""
    card = Card()
    card.update_theme({"background": "#ffffff"})
    assert card.styleSheet() != ""


@pytest.mark.gui
def test_card_add_content(qapp):
    """Test adding content to the card."""
    card = Card()
    label = QLabel("Test")
    card.add_content(label)
    assert label in card.content_widgets

import pytest
from PyQt6.QtWidgets import QApplication, QLabel
from spygate.gui.components.composite import Card

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def card(app, qtbot):
    """Create a Card instance."""
    card = Card(title="Test Card")
    qtbot.addWidget(card)
    return card

def test_card_creation(card):
    """Test that Card is created properly."""
    assert card is not None
    assert isinstance(card, Card)
    assert card.title == "Test Card"

def test_set_content(card):
    """Test setting card content."""
    content = QLabel("Test Content")
    card.set_content(content)
    assert card.content is not None
    assert isinstance(card.content, QLabel)
    assert card.content.text() == "Test Content"

def test_set_footer(card):
    """Test setting card footer."""
    footer = QLabel("Test Footer")
    card.set_footer(footer)
    assert card.footer is not None
    assert isinstance(card.footer, QLabel)
    assert card.footer.text() == "Test Footer"

def test_collapsible(card, qtbot):
    """Test card collapsible functionality."""
    # Enable collapsible
    card.set_collapsible(True)
    assert card.is_collapsible
    
    # Set content
    content = QLabel("Test Content")
    card.set_content(content)
    
    # Test collapse
    card.collapse()
    assert card.is_collapsed
    assert not card.content.isVisible()
    
    # Test expand
    card.expand()
    assert not card.is_collapsed
    assert card.content.isVisible()

def test_elevation(card):
    """Test card elevation."""
    # Test default elevation
    assert card.elevation == 1
    
    # Set new elevation
    card.set_elevation(3)
    assert card.elevation == 3
    
    # Test invalid elevation
    with pytest.raises(ValueError):
        card.set_elevation(-1)
    with pytest.raises(ValueError):
        card.set_elevation(6)

def test_theme_awareness(card):
    """Test card theme awareness."""
    # Test default theme
    assert hasattr(card, "theme")
    
    # Test theme update
    card.update_theme({"card_bg": "#ffffff", "card_border": "#000000"})
    # Add assertions based on your theme implementation 
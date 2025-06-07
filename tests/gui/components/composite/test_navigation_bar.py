"""Test navigation bar component."""

import pytest
from PyQt6.QtWidgets import QLabel

from src.gui.components.composite import NavigationBar


@pytest.mark.gui
def test_navigation_bar_creation():
    """Test creating a navigation bar."""
    nav = NavigationBar()
    assert nav is not None


@pytest.mark.gui
def test_navigation_bar_add_item():
    """Test adding an item to the navigation bar."""
    nav = NavigationBar()
    label = QLabel("Test")
    nav.add_item(label)
    assert label in nav.items()


@pytest.mark.gui
def test_add_item():
    """Test adding an item."""
    navbar = NavigationBar()
    item = navbar.add_item("home", "Home", "home.png")
    assert item is not None
    assert item.text() == "Home"


@pytest.mark.gui
def test_item_selection():
    """Test item selection."""
    navbar = NavigationBar()
    item1 = navbar.add_item("home", "Home", "home.png")
    item2 = navbar.add_item("settings", "Settings", "settings.png")
    navbar.select_item("home")
    assert navbar.selected_item() == "home"


@pytest.mark.gui
def test_active_item():
    """Test active item styling."""
    navbar = NavigationBar()
    navbar.add_item("home", "Home", "home.png")
    navbar.select_item("home")
    assert "active" in navbar.get_item("home").styleSheet()


@pytest.mark.gui
def test_orientation_change():
    """Test orientation change."""
    navbar = NavigationBar()
    navbar.set_orientation("vertical")
    assert navbar.orientation() == "vertical"
    navbar.set_orientation("horizontal")
    assert navbar.orientation() == "horizontal"


@pytest.mark.gui
def test_style_change():
    """Test style change."""
    navbar = NavigationBar()
    navbar.set_style("dark")
    assert navbar.style() == "dark"
    navbar.set_style("light")
    assert navbar.style() == "light"


@pytest.mark.gui
def test_badge_update():
    """Test badge update."""
    navbar = NavigationBar()
    navbar.add_item("messages", "Messages", "messages.png")
    navbar.set_badge("messages", "5")
    assert navbar.get_badge("messages") == "5"


@pytest.mark.gui
def test_remove_item():
    """Test removing an item."""
    navbar = NavigationBar()
    navbar.add_item("home", "Home", "home.png")
    navbar.remove_item("home")
    assert navbar.get_item("home") is None


@pytest.mark.gui
def test_update_item():
    """Test updating an item."""
    navbar = NavigationBar()
    navbar.add_item("home", "Home", "home.png")
    navbar.update_item("home", text="New Home")
    assert navbar.get_item("home").text() == "New Home"


@pytest.mark.gui
def test_keyboard_navigation():
    """Test keyboard navigation."""
    navbar = NavigationBar()
    navbar.add_item("home", "Home", "home.png")
    navbar.add_item("settings", "Settings", "settings.png")
    navbar.select_next()
    assert navbar.selected_item() == "settings"
    navbar.select_previous()
    assert navbar.selected_item() == "home"


@pytest.mark.gui
def test_error_handling():
    """Test error handling."""
    navbar = NavigationBar()
    with pytest.raises(ValueError):
        navbar.select_item("nonexistent")
    with pytest.raises(ValueError):
        navbar.set_orientation("invalid")


@pytest.mark.gui
def test_layout_behavior():
    """Test layout behavior."""
    navbar = NavigationBar()
    navbar.add_item("home", "Home", "home.png")
    navbar.add_item("settings", "Settings", "settings.png")
    assert navbar.layout().count() == 2

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from spygate.gui.components.composite import NavigationBar

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def navbar(app, qtbot):
    """Create a NavigationBar instance."""
    navbar = NavigationBar(orientation="horizontal", style="tabs")
    qtbot.addWidget(navbar)
    return navbar

def test_navbar_creation(navbar):
    """Test that NavigationBar is created properly."""
    assert navbar is not None
    assert isinstance(navbar, NavigationBar)
    assert navbar.orientation == "horizontal"
    assert navbar.style == "tabs"

def test_add_item(navbar):
    """Test adding navigation items."""
    navbar.add_item("home", "Home", icon="home.png")
    assert "home" in navbar.items
    
    # Add item without icon or badge
    navbar.add_item("home", "Home")
    assert "home" in navbar.items
    assert navbar.items["home"].text() == "Home"
    
    # Add item with icon
    navbar.add_item("stats", "Statistics", icon="stats.png")
    assert "stats" in navbar.items
    assert navbar.items["stats"].icon() is not None
    
    # Add item with badge
    navbar.add_item("messages", "Messages", badge="3")
    assert "messages" in navbar.items
    assert navbar.items["messages"].badge_text == "3"

def test_item_selection(navbar, qtbot):
    """Test item selection handling."""
    selection_made = False
    selected_id = None
    
    def on_select(item_id):
        nonlocal selection_made, selected_id
        selection_made = True
        selected_id = item_id
    
    navbar.on_select = on_select
    
    # Add and click item
    navbar.add_item("home", "Home")
    qtbot.mouseClick(navbar.items["home"], Qt.MouseButton.LeftButton)
    
    assert selection_made
    assert selected_id == "home"

def test_active_item(navbar):
    """Test active item management."""
    # Add items
    navbar.add_item("home", "Home")
    navbar.add_item("stats", "Statistics")
    
    # Set active item
    navbar.set_active("home")
    assert navbar.active_item == "home"
    assert navbar.items["home"].is_active
    assert not navbar.items["stats"].is_active
    
    # Change active item
    navbar.set_active("stats")
    assert navbar.active_item == "stats"
    assert navbar.items["stats"].is_active
    assert not navbar.items["home"].is_active

def test_orientation_change(navbar):
    """Test orientation change."""
    # Test horizontal (default)
    assert navbar.orientation == "horizontal"
    
    # Change to vertical
    navbar.set_orientation("vertical")
    assert navbar.orientation == "vertical"
    
    # Test invalid orientation
    with pytest.raises(ValueError):
        navbar.set_orientation("invalid")

def test_style_change(navbar):
    """Test navigation style change."""
    # Test tabs (default)
    assert navbar.style == "tabs"
    
    # Change to pills
    navbar.set_style("pills")
    assert navbar.style == "pills"
    
    # Change to default
    navbar.set_style("default")
    assert navbar.style == "default"
    
    # Test invalid style
    with pytest.raises(ValueError):
        navbar.set_style("invalid")

def test_badge_update(navbar):
    """Test badge update functionality."""
    # Add item with badge
    navbar.add_item("messages", "Messages", badge="3")
    assert navbar.items["messages"].badge_text == "3"
    
    # Update badge
    navbar.update_badge("messages", "5")
    assert navbar.items["messages"].badge_text == "5"
    
    # Clear badge
    navbar.update_badge("messages", None)
    assert navbar.items["messages"].badge_text is None

def test_remove_item(navbar):
    """Test removing navigation items."""
    # Add items
    navbar.add_item("home", "Home")
    navbar.add_item("stats", "Statistics")
    assert len(navbar.items) == 2
    
    # Remove item
    navbar.remove_item("home")
    assert "home" not in navbar.items
    assert len(navbar.items) == 1
    
    # Try removing non-existent item
    with pytest.raises(KeyError):
        navbar.remove_item("nonexistent")

def test_update_item(navbar):
    """Test updating navigation item properties."""
    # Add initial item
    navbar.add_item("home", "Home", icon="home.png")
    
    # Update text
    navbar.update_item("home", text="Homepage")
    assert navbar.items["home"].text() == "Homepage"
    
    # Update icon
    navbar.update_item("home", icon="house.png")
    assert navbar.items["home"].icon() is not None
    
    # Update both
    navbar.update_item("home", text="Main", icon="main.png")
    assert navbar.items["home"].text() == "Main"
    assert navbar.items["home"].icon() is not None
    
    # Try updating non-existent item
    with pytest.raises(KeyError):
        navbar.update_item("nonexistent", text="Error")

def test_keyboard_navigation(navbar, qtbot):
    """Test keyboard navigation between items."""
    # Add items
    navbar.add_item("home", "Home")
    navbar.add_item("stats", "Statistics")
    navbar.add_item("settings", "Settings")
    
    # Set initial focus
    navbar.items["home"].setFocus()
    assert navbar.items["home"].hasFocus()
    
    # Test Tab key navigation
    qtbot.keyClick(navbar.items["home"], Qt.Key.Key_Tab)
    assert navbar.items["stats"].hasFocus()
    
    qtbot.keyClick(navbar.items["stats"], Qt.Key.Key_Tab)
    assert navbar.items["settings"].hasFocus()
    
    # Test Shift+Tab navigation
    qtbot.keyClick(navbar.items["settings"], Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
    assert navbar.items["stats"].hasFocus()

def test_error_handling(navbar):
    """Test error handling for invalid operations."""
    # Test setting active item with invalid ID
    with pytest.raises(KeyError):
        navbar.set_active("nonexistent")
    
    # Test updating badge with invalid ID
    with pytest.raises(KeyError):
        navbar.update_badge("nonexistent", "5")
    
    # Test adding item with duplicate ID
    navbar.add_item("home", "Home")
    with pytest.raises(ValueError):
        navbar.add_item("home", "Duplicate")

def test_layout_behavior(navbar):
    """Test layout behavior in different orientations."""
    # Add test items
    navbar.add_item("home", "Home")
    navbar.add_item("stats", "Statistics")
    navbar.add_item("settings", "Settings")
    
    # Test horizontal layout (default)
    assert navbar.orientation == "horizontal"
    first_item_pos = navbar.items["home"].pos()
    second_item_pos = navbar.items["stats"].pos()
    assert second_item_pos.x() > first_item_pos.x()  # Items should be laid out horizontally
    assert second_item_pos.y() == first_item_pos.y()  # Items should be at same vertical position
    
    # Test vertical layout
    navbar.set_orientation("vertical")
    first_item_pos = navbar.items["home"].pos()
    second_item_pos = navbar.items["stats"].pos()
    assert second_item_pos.y() > first_item_pos.y()  # Items should be laid out vertically
    assert second_item_pos.x() == first_item_pos.x()  # Items should be at same horizontal position 
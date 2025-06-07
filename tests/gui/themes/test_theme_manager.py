"""Test theme manager."""

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from src.gui.themes import ThemeManager


@pytest.fixture
def app():
    """Create a QApplication instance."""
    app = QApplication([])
    yield app
    app.quit()


@pytest.fixture
def theme_manager(app):
    """Create a ThemeManager instance."""
    return ThemeManager(app)


@pytest.mark.gui
def test_theme_manager_creation(theme_manager):
    """Test that ThemeManager is created properly."""
    assert theme_manager is not None
    assert isinstance(theme_manager, ThemeManager)
    assert theme_manager.current_theme == "dark_teal"  # Default theme


@pytest.mark.gui
def test_material_themes(theme_manager):
    """Test Qt Material themes functionality."""
    # Check that material themes are loaded
    assert len(theme_manager.material_themes) > 0

    # Test applying a material theme
    success = theme_manager.apply_theme("dark_teal", custom=False)
    assert success
    assert theme_manager.current_theme == "dark_teal"

    # Test applying invalid theme
    success = theme_manager.apply_theme("invalid_theme", custom=False)
    assert not success


@pytest.mark.gui
def test_custom_themes(theme_manager):
    """Test custom themes functionality."""
    # Create and save a custom theme
    test_theme = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    """
    success = theme_manager.save_custom_theme("test_theme", test_theme)
    assert success
    assert "test_theme" in theme_manager.custom_styles

    # Apply custom theme
    success = theme_manager.apply_theme("test_theme", custom=True)
    assert success
    assert theme_manager.current_theme == "test_theme"


@pytest.mark.gui
def test_theme_listing(theme_manager):
    """Test getting available themes."""
    # Get all themes
    themes = theme_manager.get_available_themes(include_custom=True)
    assert len(themes) > 0
    assert all(theme_type in ["material", "custom"] for theme_type in themes.values())

    # Get only material themes
    material_themes = theme_manager.get_available_themes(include_custom=False)
    assert len(material_themes) > 0
    assert all(theme_type == "material" for theme_type in material_themes.values())

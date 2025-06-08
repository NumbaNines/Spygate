"""
Tests for the Dashboard component.
"""

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from spygate.gui.components.dashboard import ClipCard, Dashboard


@pytest.fixture(scope="session")
def app():
    """Create a QApplication instance."""
    app = QApplication([])
    yield app
    app.quit()


@pytest.fixture
def dashboard(app, qtbot):
    """Create a Dashboard instance."""
    dashboard = Dashboard()
    dashboard.show()
    qtbot.addWidget(dashboard)
    return dashboard


def test_dashboard_initialization(dashboard):
    """Test that the dashboard initializes correctly."""
    # Check window properties
    assert dashboard.windowTitle() == "Spygate"
    assert dashboard.size().width() >= 1200
    assert dashboard.size().height() >= 800

    # Check theme
    assert dashboard.styleSheet() == "background: #1E1E1E;"

    # Check main components exist
    assert dashboard.sidebar is not None
    assert dashboard.search_bar is not None
    assert dashboard.player_combo is not None
    assert dashboard.tag_list is not None
    assert dashboard.grid_layout is not None


def test_sidebar_navigation(dashboard):
    """Test sidebar navigation buttons."""
    # Check all navigation buttons exist
    nav_buttons = ["Home", "Upload", "Clips", "Analytics", "Settings"]

    for button_text in nav_buttons:
        button = dashboard.sidebar.nav_buttons.get(button_text)
        assert button is not None, f"Button '{button_text}' not found"
        assert button.text() == button_text


def test_clip_management(dashboard, qtbot):
    """Test clip management functionality."""
    # Add a test clip
    clip_data = {"title": "Test Clip", "player": "Self", "tags": ["Test", "Demo"]}
    dashboard.add_clip(clip_data)
    qtbot.wait(100)  # Wait for UI to update

    # Verify clip was added
    assert len(dashboard.clips) == 1
    clip = dashboard.clips[0]
    assert clip.title == "Test Clip"
    assert clip.player_name == "Self"

    # Check tags were added
    tag_texts = [dashboard.tag_list.item(i).text() for i in range(dashboard.tag_list.count())]
    assert "Test" in tag_texts
    assert "Demo" in tag_texts

    # Check player was added to combo
    assert "Self" in [
        dashboard.player_combo.itemText(i) for i in range(dashboard.player_combo.count())
    ]


def test_search_and_filtering(dashboard, qtbot):
    """Test search and filtering functionality."""
    # Add test clips
    clips = [
        {"title": "Offensive Play", "player": "Self", "tags": ["Offense", "Strategy"]},
        {
            "title": "Defensive Play",
            "player": "Opponent: John",
            "tags": ["Defense", "Strategy"],
        },
        {
            "title": "Tournament Game",
            "player": "Self",
            "tags": ["Tournament", "Offense"],
        },
    ]
    for clip_data in clips:
        dashboard.add_clip(clip_data)
    qtbot.wait(100)  # Wait for UI to update

    # Test search by title
    qtbot.keyClicks(dashboard.search_bar, "offensive")
    qtbot.wait(100)  # Wait for filter to apply
    assert dashboard.clips[0].isVisible()  # "Offensive Play" should be visible
    assert not dashboard.clips[1].isVisible()  # "Defensive Play" should be hidden
    assert not dashboard.clips[2].isVisible()  # "Tournament Game" should be hidden

    # Clear search
    dashboard.search_bar.clear()
    qtbot.wait(100)

    # Test player filter
    dashboard.player_combo.setCurrentText("Self")
    qtbot.wait(100)  # Wait for filter to apply
    assert dashboard.clips[0].isVisible()  # Self's clip should be visible
    assert not dashboard.clips[1].isVisible()  # Opponent's clip should be hidden
    assert dashboard.clips[2].isVisible()  # Self's tournament clip should be visible

    # Test "All Players" filter
    dashboard.player_combo.setCurrentText("All Players")
    qtbot.wait(100)
    assert all(clip.isVisible() for clip in dashboard.clips)  # All clips should be visible

    # Test tag filter
    strategy_tag = dashboard.tag_list.findItems("Strategy", Qt.MatchFlag.MatchExactly)[0]
    strategy_tag.setSelected(True)
    qtbot.wait(100)  # Wait for filter to apply
    assert dashboard.clips[0].isVisible()  # Has "Strategy" tag
    assert dashboard.clips[1].isVisible()  # Has "Strategy" tag
    assert not dashboard.clips[2].isVisible()  # Doesn't have "Strategy" tag

    # Test multiple tag selection
    offense_tag = dashboard.tag_list.findItems("Offense", Qt.MatchFlag.MatchExactly)[0]
    offense_tag.setSelected(True)
    qtbot.wait(100)  # Wait for filter to apply
    assert dashboard.clips[0].isVisible()  # Has both tags
    assert dashboard.clips[1].isVisible()  # Has "Strategy" tag
    assert dashboard.clips[2].isVisible()  # Has "Offense" tag

    # Test gameplan filtering
    dashboard.create_new_gameplan()
    dashboard.add_to_gameplan("New Gameplan", dashboard.clips[0])
    dashboard.on_gameplan_selected("New Gameplan")
    qtbot.wait(100)  # Wait for filter to apply
    assert dashboard.clips[0].isVisible()  # In gameplan
    assert not dashboard.clips[1].isVisible()  # Not in gameplan
    assert not dashboard.clips[2].isVisible()  # Not in gameplan

    # Clear gameplan filter
    dashboard.clear_gameplan_filter()
    qtbot.wait(100)  # Wait for filter to apply
    assert all(clip.isVisible() for clip in dashboard.clips)  # All clips should be visible


def test_gameplan_management(dashboard, qtbot):
    """Test gameplan management functionality."""
    # Create a new gameplan
    gameplan = dashboard.create_new_gameplan()
    qtbot.wait(100)  # Wait for UI to update

    # Add a clip to the gameplan
    clip_data = {"title": "Gameplan Clip", "player": "Self", "tags": ["Strategy"]}
    dashboard.add_clip(clip_data)
    clip = dashboard.clips[0]
    dashboard.add_to_gameplan("New Gameplan", clip)
    qtbot.wait(100)  # Wait for UI to update

    # Verify clip is in gameplan
    assert clip in dashboard.gameplan_clips["New Gameplan"]

    # Test gameplan filtering
    dashboard.on_gameplan_selected("New Gameplan")
    qtbot.wait(100)  # Wait for filter to apply
    assert clip.isVisible()

    # Clear gameplan filter
    dashboard.clear_gameplan_filter()
    qtbot.wait(100)  # Wait for filter to apply
    assert clip.isVisible()


def test_player_name_handling(dashboard, qtbot):
    """Test player name handling throughout the dashboard."""
    # Add clips with different player names
    clips = [
        {"title": "Self Clip", "player": "Self", "tags": ["Test"]},
        {"title": "Opponent Clip", "player": "Opponent: Jane", "tags": ["Test"]},
        {"title": "Another Opponent", "player": "Opponent: Bob", "tags": ["Test"]},
    ]
    for clip_data in clips:
        dashboard.add_clip(clip_data)
    qtbot.wait(100)  # Wait for UI to update

    # Check all players are in combo box
    players = {dashboard.player_combo.itemText(i) for i in range(dashboard.player_combo.count())}
    assert "Self" in players
    assert "Opponent: Jane" in players
    assert "Opponent: Bob" in players

    # Test filtering by each player
    for player in players:
        dashboard.player_combo.setCurrentText(player)
        qtbot.wait(100)  # Wait for filter to apply
        visible_clips = [clip for clip in dashboard.clips if clip.isVisible()]
        assert all(clip.player_name == player for clip in visible_clips)


def test_theme_consistency(dashboard, qtbot):
    """Test theme consistency across components."""
    # Check main colors
    assert "#1E1E1E" in dashboard.styleSheet()  # Background
    assert "#3B82F6" in dashboard.search_bar.styleSheet()  # Accent color

    # Check sidebar theme
    assert "#2A2A2A" in dashboard.sidebar.styleSheet()

    # Check clip cards theme
    clip_data = {"title": "Theme Test", "player": "Self", "tags": ["Test"]}
    dashboard.add_clip(clip_data)
    qtbot.wait(100)  # Wait for UI to update
    clip = dashboard.clips[0]
    assert "#2A2A2A" in clip.styleSheet()  # Card background

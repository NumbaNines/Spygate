"""Tests for player name dialog."""

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from spygate.gui.components.dialogs.player_name_dialog import PlayerNameDialog


@pytest.fixture
def dialog(qtbot):
    """Create a player name dialog."""
    dialog = PlayerNameDialog()
    qtbot.addWidget(dialog)
    return dialog


def test_initial_state(dialog):
    """Test initial dialog state."""
    assert dialog.windowTitle() == "Select Player"
    assert dialog.self_radio.isChecked()  # Self should be default
    assert not dialog.opponent_radio.isChecked()
    assert not dialog.opponent_name.isEnabled()


def test_radio_toggle(dialog, qtbot):
    """Test radio button toggling."""
    # Click opponent radio
    qtbot.mouseClick(dialog.opponent_radio, None)
    assert not dialog.self_radio.isChecked()
    assert dialog.opponent_radio.isChecked()
    assert dialog.opponent_name.isEnabled()

    # Click self radio
    qtbot.mouseClick(dialog.self_radio, None)
    assert dialog.self_radio.isChecked()
    assert not dialog.opponent_radio.isChecked()
    assert not dialog.opponent_name.isEnabled()


def test_get_player_name_self(dialog):
    """Test getting player name when 'Self' is selected."""
    dialog.self_radio.setChecked(True)
    assert dialog.get_player_name() == "Self"


def test_get_player_name_opponent(dialog):
    """Test getting player name when 'Opponent' is selected."""
    dialog.opponent_radio.setChecked(True)
    dialog.opponent_name.setText("John Doe")
    assert dialog.get_player_name() == "Opponent: John Doe"


def test_accept_self(dialog, qtbot):
    """Test accepting dialog with 'Self' selected."""
    dialog.self_radio.setChecked(True)

    # Click OK button
    ok_button = dialog.findChild(type(dialog.self_radio), "OK")
    qtbot.mouseClick(ok_button, None)

    assert dialog.result()  # Should be accepted


def test_accept_opponent_valid(dialog, qtbot):
    """Test accepting dialog with valid opponent name."""
    dialog.opponent_radio.setChecked(True)
    dialog.opponent_name.setText("John Doe")

    # Click OK button
    ok_button = dialog.findChild(type(dialog.self_radio), "OK")
    qtbot.mouseClick(ok_button, None)

    assert dialog.result()  # Should be accepted


def test_accept_opponent_empty(dialog, qtbot, monkeypatch):
    """Test accepting dialog with empty opponent name."""
    # Mock QMessageBox
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    dialog.opponent_radio.setChecked(True)
    dialog.opponent_name.setText("")  # Empty name

    # Click OK button
    ok_button = dialog.findChild(type(dialog.self_radio), "OK")
    qtbot.mouseClick(ok_button, None)

    # Should show warning and not accept
    mock_warning.assert_called_once()
    assert not dialog.result()


def test_accept_opponent_whitespace(dialog, qtbot, monkeypatch):
    """Test accepting dialog with whitespace opponent name."""
    # Mock QMessageBox
    mock_warning = MagicMock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    dialog.opponent_radio.setChecked(True)
    dialog.opponent_name.setText("   ")  # Only whitespace

    # Click OK button
    ok_button = dialog.findChild(type(dialog.self_radio), "OK")
    qtbot.mouseClick(ok_button, None)

    # Should show warning and not accept
    mock_warning.assert_called_once()
    assert not dialog.result()


def test_cancel(dialog, qtbot):
    """Test canceling the dialog."""
    # Click Cancel button
    cancel_button = dialog.findChild(type(dialog.self_radio), "Cancel")
    qtbot.mouseClick(cancel_button, None)

    assert not dialog.result()  # Should be rejected


def test_opponent_name_focus(dialog, qtbot):
    """Test opponent name gets focus when radio selected."""
    # Click opponent radio
    qtbot.mouseClick(dialog.opponent_radio, None)
    assert dialog.opponent_name.hasFocus()

    # Click self radio
    qtbot.mouseClick(dialog.self_radio, None)
    assert not dialog.opponent_name.hasFocus()

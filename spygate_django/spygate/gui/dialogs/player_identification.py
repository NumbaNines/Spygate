"""
Dialog for identifying players in imported videos.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


class PlayerIdentificationDialog(QDialog):
    """Dialog for identifying players in imported videos."""

    def __init__(self, parent=None):
        """Initialize the player identification dialog."""
        super().__init__(parent)
        self.setWindowTitle("Player Identification")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Player type selection
        type_layout = QHBoxLayout()
        self.self_radio = QRadioButton("Self")
        self.self_radio.setChecked(True)
        self.self_radio.toggled.connect(self._on_player_type_changed)
        self.opponent_radio = QRadioButton("Opponent")
        self.opponent_radio.toggled.connect(self._on_player_type_changed)

        type_layout.addWidget(self.self_radio)
        type_layout.addWidget(self.opponent_radio)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # Opponent name input (hidden by default)
        self.name_label = QLabel("Opponent Name:")
        self.name_label.setVisible(False)
        layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter opponent name")
        self.name_input.setVisible(False)
        layout.addWidget(self.name_input)

        # Recent opponents dropdown (hidden by default)
        self.recent_label = QLabel("Recent Opponents:")
        self.recent_label.setVisible(False)
        layout.addWidget(self.recent_label)

        self.recent_combo = QComboBox()
        self.recent_combo.setVisible(False)
        self.recent_combo.currentTextChanged.connect(self._on_recent_selected)
        layout.addWidget(self.recent_combo)

        # Buttons
        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """
        )

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """
        )

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Load recent opponents
        self._load_recent_opponents()

    def _on_player_type_changed(self):
        """Handle player type radio button changes."""
        is_opponent = self.opponent_radio.isChecked()
        self.name_label.setVisible(is_opponent)
        self.name_input.setVisible(is_opponent)
        self.recent_label.setVisible(is_opponent)
        self.recent_combo.setVisible(is_opponent)

    def _on_recent_selected(self, name: str):
        """Handle selection from recent opponents dropdown."""
        if name:
            self.name_input.setText(name)

    def _load_recent_opponents(self):
        """Load recent opponents into the dropdown."""
        # TODO: Load from database
        recent_opponents = ["John Doe", "Jane Smith", "Bob Wilson"]
        self.recent_combo.addItems(recent_opponents)

    def get_player_name(self) -> str:
        """
        Get the selected player name.

        Returns:
            str: "Self" or "Opponent: {name}"
        """
        if self.self_radio.isChecked():
            return "Self"
        else:
            name = self.name_input.text().strip()
            return f"Opponent: {name}" if name else "Unknown Opponent"

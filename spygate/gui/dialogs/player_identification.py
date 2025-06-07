"""
Dialog for identifying players in video clips.
"""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)


class PlayerIdentificationDialog(QDialog):
    """Dialog for identifying whether a clip is from self or an opponent."""

    def __init__(self, parent=None):
        """Initialize the dialog.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.player_name: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Player Identification")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Who's in this clip?")
        title.setStyleSheet("font-size: 16px; color: white;")
        layout.addWidget(title)

        # Radio buttons frame
        radio_frame = QFrame()
        radio_frame.setStyleSheet(
            "QFrame {"
            "   background-color: #2A2A2A;"
            "   border-radius: 8px;"
            "   padding: 16px;"
            "}"
        )
        radio_layout = QVBoxLayout(radio_frame)

        # Radio button group
        self.button_group = QButtonGroup(self)

        # Self radio button
        self.self_radio = QRadioButton("My gameplay")
        self.self_radio.setStyleSheet("color: #D1D5DB;")
        self.button_group.addButton(self.self_radio)
        radio_layout.addWidget(self.self_radio)

        # Opponent radio button
        self.opponent_radio = QRadioButton("Opponent's gameplay")
        self.opponent_radio.setStyleSheet("color: #D1D5DB;")
        self.button_group.addButton(self.opponent_radio)
        radio_layout.addWidget(self.opponent_radio)

        layout.addWidget(radio_frame)

        # Opponent name input (hidden initially)
        self.opponent_frame = QFrame()
        self.opponent_frame.setVisible(False)
        self.opponent_frame.setStyleSheet(
            "QFrame {"
            "   background-color: #2A2A2A;"
            "   border-radius: 8px;"
            "   padding: 16px;"
            "   margin-top: 8px;"
            "}"
        )
        opponent_layout = QVBoxLayout(self.opponent_frame)

        opponent_label = QLabel("Opponent's Name:")
        opponent_label.setStyleSheet("color: #D1D5DB;")
        opponent_layout.addWidget(opponent_label)

        self.opponent_input = QLineEdit()
        self.opponent_input.setStyleSheet(
            "QLineEdit {"
            "   background-color: #1E1E1E;"
            "   color: white;"
            "   border: 1px solid #3B82F6;"
            "   border-radius: 4px;"
            "   padding: 8px;"
            "}"
        )
        opponent_layout.addWidget(self.opponent_input)

        layout.addWidget(self.opponent_frame)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #374151;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px 16px;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #4B5563;"
            "}"
        )
        cancel_button.clicked.connect(self.reject)

        confirm_button = QPushButton("Confirm")
        confirm_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #3B82F6;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px 16px;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #2563EB;"
            "}"
        )
        confirm_button.clicked.connect(self._handle_confirm)

        button_layout.addWidget(cancel_button)
        button_layout.addWidget(confirm_button)

        layout.addLayout(button_layout)

        # Connect radio button signals
        self.self_radio.toggled.connect(self._handle_radio_toggle)
        self.opponent_radio.toggled.connect(self._handle_radio_toggle)

        # Set window properties
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )
        self.setStyleSheet("background-color: #1E1E1E;")

    def _handle_radio_toggle(self, checked: bool):
        """Handle radio button toggle events.

        Args:
            checked: Whether the radio button was checked
        """
        self.opponent_frame.setVisible(self.opponent_radio.isChecked())

    def _handle_confirm(self):
        """Handle confirm button click."""
        if self.self_radio.isChecked():
            self.player_name = "Self"
            self.accept()
        elif self.opponent_radio.isChecked():
            opponent_name = self.opponent_input.text().strip()
            if opponent_name:
                self.player_name = f"Opponent: {opponent_name}"
                self.accept()
            else:
                self.opponent_input.setStyleSheet(
                    "QLineEdit {"
                    "   background-color: #1E1E1E;"
                    "   color: white;"
                    "   border: 1px solid #EF4444;"
                    "   border-radius: 4px;"
                    "   padding: 8px;"
                    "}"
                )
                self.opponent_input.setPlaceholderText("Please enter opponent's name")

    def get_player_name(self) -> Optional[str]:
        """Get the selected player name.

        Returns:
            The player name if confirmed, None if cancelled
        """
        return self.player_name

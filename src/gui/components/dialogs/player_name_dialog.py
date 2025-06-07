"""Dialog for entering player names."""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class PlayerNameDialog(QDialog):
    """Dialog for entering player names."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the dialog.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Player Name")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Player type selection
        type_group = QButtonGroup(self)
        self.self_radio = QRadioButton("Self")
        self.opponent_radio = QRadioButton("Opponent")
        type_group.addButton(self.self_radio)
        type_group.addButton(self.opponent_radio)

        # Default to "Self"
        self.self_radio.setChecked(True)

        # Add radio buttons to a horizontal layout
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.self_radio)
        radio_layout.addWidget(self.opponent_radio)
        layout.addLayout(radio_layout)

        # Name input (initially hidden)
        self.name_label = QLabel("Opponent Name:")
        self.name_input = QLineEdit()
        self.name_label.hide()
        self.name_input.hide()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.opponent_radio.toggled.connect(self._on_opponent_toggled)

        # Set minimum size
        self.setMinimumWidth(300)

    def _on_opponent_toggled(self, checked: bool):
        """Handle opponent radio button toggle.

        Args:
            checked: Whether the opponent radio button is checked
        """
        self.name_label.setVisible(checked)
        self.name_input.setVisible(checked)
        if checked:
            self.name_input.setFocus()

    def get_player_name(self) -> str:
        """Get the entered player name.

        Returns:
            str: "Self" or "Opponent: {name}"
        """
        if self.self_radio.isChecked():
            return "Self"
        return f"Opponent: {self.name_input.text()}"

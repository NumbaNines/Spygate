"""Dialog for specifying player name during video import."""

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
    """Dialog for getting player name during video import."""

    def __init__(self, parent=None):
        """Initialize the dialog.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Player Identification")
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Add description label
        desc_label = QLabel("Is this your own gameplay or from another player?", self)
        layout.addWidget(desc_label)

        # Radio buttons for player type
        self.player_type_group = QButtonGroup(self)

        # Self radio button
        self.self_radio = QRadioButton("My Own Gameplay", self)
        self.self_radio.setChecked(True)  # Default selection
        self.player_type_group.addButton(self.self_radio)
        layout.addWidget(self.self_radio)

        # Opponent radio button and name input
        opponent_widget = QWidget(self)
        opponent_layout = QHBoxLayout(opponent_widget)
        opponent_layout.setContentsMargins(0, 0, 0, 0)

        self.opponent_radio = QRadioButton("Opponent:", self)
        self.player_type_group.addButton(self.opponent_radio)
        opponent_layout.addWidget(self.opponent_radio)

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter opponent's name")
        self.name_input.setEnabled(False)  # Disabled by default
        opponent_layout.addWidget(self.name_input)

        layout.addWidget(opponent_widget)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        cancel_button = QPushButton("Cancel", self)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        self.self_radio.toggled.connect(self._on_radio_toggled)
        self.opponent_radio.toggled.connect(self._on_radio_toggled)

        # Set dialog properties
        self.setModal(True)
        self.setMinimumWidth(300)

    def _on_radio_toggled(self, checked):
        """Handle radio button toggle."""
        if self.opponent_radio.isChecked():
            self.name_input.setEnabled(True)
            self.name_input.setFocus()
        else:
            self.name_input.setEnabled(False)

    def get_player_name(self) -> str:
        """Get the specified player name.

        Returns:
            str: "Self" if own gameplay, or "Opponent: Name" if opponent
        """
        if self.self_radio.isChecked():
            return "Self"
        else:
            name = self.name_input.text().strip()
            return f"Opponent: {name}" if name else "Opponent"

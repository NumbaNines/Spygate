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

        # Add description
        desc = QLabel(
            "Is this your own gameplay or from another player?\n"
            "This helps organize and analyze clips effectively."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Radio buttons for selection
        self.self_radio = QRadioButton("My Own Gameplay")
        self.self_radio.setObjectName("self_radio")
        self.opponent_radio = QRadioButton("Other Player's Gameplay")
        self.opponent_radio.setObjectName("opponent_radio")

        # Group radio buttons
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.self_radio)
        self.button_group.addButton(self.opponent_radio)

        layout.addWidget(self.self_radio)
        layout.addWidget(self.opponent_radio)

        # Opponent name input (hidden initially)
        self.opponent_widget = QWidget()
        opponent_layout = QHBoxLayout(self.opponent_widget)
        opponent_layout.addWidget(QLabel("Player Name:"))
        self.opponent_name = QLineEdit()
        self.opponent_name.setObjectName("opponent_name")
        opponent_layout.addWidget(self.opponent_name)
        self.opponent_widget.hide()
        layout.addWidget(self.opponent_widget)

        # Connect radio button signals
        self.self_radio.toggled.connect(self.on_radio_toggled)
        self.opponent_radio.toggled.connect(self.on_radio_toggled)

        # Buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setObjectName("ok_button")
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("cancel_button")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        # Connect buttons
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        # Select self by default
        self.self_radio.setChecked(True)

    def on_radio_toggled(self, checked):
        """Handle radio button toggle events.

        Args:
            checked: Whether the button was checked
        """
        if self.opponent_radio.isChecked():
            self.opponent_widget.show()
            self.opponent_name.setFocus()
        else:
            self.opponent_widget.hide()

    def get_player_name(self) -> str:
        """Get the selected player name.

        Returns:
            str: "Self" for own gameplay, or "Opponent: Name" for other players
        """
        if self.self_radio.isChecked():
            return "Self"
        else:
            name = self.opponent_name.text().strip()
            return f"Opponent: {name}" if name else ""

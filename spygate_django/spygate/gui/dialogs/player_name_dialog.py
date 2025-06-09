"""Dialog for entering Madden player details."""

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class PlayerNameDialog(QDialog):
    """Dialog for entering Madden player gamertag/username."""

    # Signal emitted when player details are confirmed
    player_confirmed = Signal(str)  # player_name (e.g., "Self" or "Opponent: JonBeast")

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Enter Player Details")
        self.setModal(True)

        # Create layout
        layout = QVBoxLayout(self)

        # Player type selection
        type_label = QLabel("Who's gameplay is this?")
        layout.addWidget(type_label)

        # Radio buttons for player type
        self.type_group = QButtonGroup(self)
        self.self_radio = QRadioButton("My gameplay")
        self.opponent_radio = QRadioButton("Opponent's gameplay")
        self.type_group.addButton(self.self_radio)
        self.type_group.addButton(self.opponent_radio)
        layout.addWidget(self.self_radio)
        layout.addWidget(self.opponent_radio)

        # Default to self gameplay
        self.self_radio.setChecked(True)

        # Opponent name input (hidden initially)
        self.opponent_layout = QHBoxLayout()
        opponent_label = QLabel("Opponent's Gamertag:")
        self.opponent_input = QLineEdit()
        self.opponent_layout.addWidget(opponent_label)
        self.opponent_layout.addWidget(self.opponent_input)
        layout.addLayout(self.opponent_layout)

        # Hide opponent input initially
        opponent_label.setVisible(False)
        self.opponent_input.setVisible(False)

        # Connect radio buttons to toggle opponent name visibility
        self.self_radio.toggled.connect(lambda checked: self._toggle_opponent_input(not checked))

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Set OK as default button and improve keyboard navigation
        ok_button.setDefault(True)
        ok_button.setAutoDefault(True)
        self.opponent_input.returnPressed.connect(ok_button.click)

        # Set dark theme style
        self.setStyleSheet(
            """
            QDialog {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
            QRadioButton {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #444;
                color: #ffffff;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #444;
                color: #ffffff;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """
        )

    def _toggle_opponent_input(self, show: bool):
        """Toggle visibility of opponent name input."""
        for widget in [
            self.opponent_layout.itemAt(i).widget() for i in range(self.opponent_layout.count())
        ]:
            if widget:
                widget.setVisible(show)

        # Set focus to opponent input when it becomes visible
        if show:
            self.opponent_input.setFocus()
            self.opponent_input.selectAll()  # Select any existing text for easy replacement

    def accept(self):
        """Handle dialog acceptance."""
        if self.self_radio.isChecked():
            player_name = "Self"
        else:
            opponent_name = self.opponent_input.text().strip()
            if not opponent_name:
                QMessageBox.warning(
                    self, "Missing Information", "Please enter the opponent's gamertag."
                )
                return
            player_name = f"Opponent: {opponent_name}"

        self.player_confirmed.emit(player_name)
        super().accept()

    def get_player_name(self) -> str:
        """Get the entered player name.

        Returns:
            str: "Self" or "Opponent: <gamertag>"
        """
        if self.self_radio.isChecked():
            return "Self"
        return f"Opponent: {self.opponent_input.text().strip()}"

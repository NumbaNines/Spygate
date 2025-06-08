"""Dialog for entering player name and details."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class PlayerNameDialog(QDialog):
    """Dialog for entering player name and details."""

    # Signal emitted when player details are confirmed
    player_confirmed = pyqtSignal(str, str, str)  # name, team, position

    def __init__(self, parent=None):
        """Initialize the dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Enter Player Details")
        self.setModal(True)

        # Create layout
        layout = QVBoxLayout(self)

        # Name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Team input
        team_layout = QHBoxLayout()
        team_label = QLabel("Team:")
        self.team_input = QLineEdit()
        team_layout.addWidget(team_label)
        team_layout.addWidget(self.team_input)
        layout.addLayout(team_layout)

        # Position selection
        position_layout = QHBoxLayout()
        position_label = QLabel("Position:")
        self.position_combo = QComboBox()
        self.position_combo.addItems(["Forward", "Midfielder", "Defender", "Goalkeeper"])
        position_layout.addWidget(position_label)
        position_layout.addWidget(self.position_combo)
        layout.addLayout(position_layout)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def accept(self):
        """Handle dialog acceptance."""
        name = self.name_input.text().strip()
        team = self.team_input.text().strip()
        position = self.position_combo.currentText()

        if not name:
            QMessageBox.warning(self, "Missing Information", "Please enter the player's name.")
            return

        if not team:
            QMessageBox.warning(self, "Missing Information", "Please enter the player's team.")
            return

        self.player_confirmed.emit(name, team, position)
        super().accept()

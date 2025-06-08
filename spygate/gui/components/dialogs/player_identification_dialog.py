"""
Enhanced player identification dialog for video import.
Supports multiple player tagging and profile creation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ....database.schema import Player
from ....database.video_manager import VideoManager


@dataclass
class PlayerProfile:
    """Data class for player profile information with validation."""

    name: str
    team: str = field(default="")
    is_self: bool = field(default=False)
    gamertag: Optional[str] = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate player profile data after initialization."""
        if not self.name:
            raise ValueError("Player name cannot be empty")

        if self.is_self and self.gamertag:
            raise ValueError("Self player cannot have a gamertag")

        if not self.is_self and not self.team:
            # For opponents, team is required
            raise ValueError("Team is required for opponent players")

        self.name = self.name.strip()
        if self.team:
            self.team = self.team.strip()
        if self.gamertag:
            self.gamertag = self.gamertag.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary format."""
        return {
            "name": self.name,
            "team": self.team,
            "is_self": self.is_self,
            "gamertag": self.gamertag,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerProfile":
        """Create profile from dictionary data."""
        return cls(
            name=data["name"],
            team=data.get("team", ""),
            is_self=data.get("is_self", False),
            gamertag=data.get("gamertag"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.utcnow()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.utcnow()
            ),
        )

    @classmethod
    def from_db_model(cls, player: Player) -> "PlayerProfile":
        """Create profile from database Player model."""
        return cls(
            name=player.name,
            team=player.team,
            is_self=player.is_self,
            gamertag=player.gamertag,
            created_at=player.created_at,
            updated_at=player.updated_at,
        )


class PlayerIdentificationDialog(QDialog):
    """Enhanced dialog for identifying players in imported videos."""

    # Signal emitted when a new player profile is created
    player_profile_created = pyqtSignal(PlayerProfile)

    def __init__(self, video_manager: VideoManager, parent: Optional[QWidget] = None):
        """Initialize the player identification dialog."""
        super().__init__(parent)
        self.video_manager = video_manager
        self.setWindowTitle("Player Identification")
        self.setModal(True)
        self.selected_players: List[PlayerProfile] = []
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Primary player section
        primary_group = QWidget()
        primary_layout = QVBoxLayout(primary_group)
        primary_layout.setContentsMargins(0, 0, 0, 0)

        primary_label = QLabel("Primary Player:")
        primary_label.setStyleSheet("font-weight: bold; color: #3B82F6;")
        primary_layout.addWidget(primary_label)

        # Player type selection
        type_layout = QHBoxLayout()
        self.self_radio = QRadioButton("Self")
        self.opponent_radio = QRadioButton("Opponent")

        # Group radio buttons
        self.player_type_group = QButtonGroup()
        self.player_type_group.addButton(self.self_radio)
        self.player_type_group.addButton(self.opponent_radio)

        # Default to "Self"
        self.self_radio.setChecked(True)

        type_layout.addWidget(self.self_radio)
        type_layout.addWidget(self.opponent_radio)
        primary_layout.addLayout(type_layout)

        # Opponent details (initially hidden)
        self.opponent_widget = QWidget()
        opponent_layout = QVBoxLayout(self.opponent_widget)
        opponent_layout.setContentsMargins(0, 0, 0, 0)

        # Existing opponent selection
        existing_layout = QHBoxLayout()
        existing_label = QLabel("Select Existing:")
        self.opponent_combo = QComboBox()
        self.opponent_combo.setEditable(True)
        self.opponent_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)

        existing_layout.addWidget(existing_label)
        existing_layout.addWidget(self.opponent_combo)
        opponent_layout.addLayout(existing_layout)

        # Team input
        team_layout = QHBoxLayout()
        team_label = QLabel("Team:")
        self.team_input = QLineEdit()
        self.team_input.setPlaceholderText("Enter team name")

        team_layout.addWidget(team_label)
        team_layout.addWidget(self.team_input)
        opponent_layout.addLayout(team_layout)

        # Gamertag input
        gamertag_layout = QHBoxLayout()
        gamertag_label = QLabel("Gamertag:")
        self.gamertag_input = QLineEdit()
        self.gamertag_input.setPlaceholderText("Enter gamertag (optional)")

        gamertag_layout.addWidget(gamertag_label)
        gamertag_layout.addWidget(self.gamertag_input)
        opponent_layout.addLayout(gamertag_layout)

        primary_layout.addWidget(self.opponent_widget)
        layout.addWidget(primary_group)

        # Additional players section
        additional_group = QWidget()
        additional_layout = QVBoxLayout(additional_group)
        additional_layout.setContentsMargins(0, 0, 0, 0)

        additional_label = QLabel("Additional Players (Optional):")
        additional_label.setStyleSheet("font-weight: bold; color: #3B82F6;")
        additional_layout.addWidget(additional_label)

        # Player list
        self.players_list = QListWidget()
        additional_layout.addWidget(self.players_list)

        # Add player controls
        add_layout = QHBoxLayout()
        self.add_player_combo = QComboBox()
        self.add_player_combo.setEditable(True)
        self.add_player_combo.setInsertPolicy(
            QComboBox.InsertPolicy.InsertAlphabetically
        )

        add_button = QPushButton("Add")
        add_button.clicked.connect(self._on_add_player)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._on_remove_players)

        add_layout.addWidget(self.add_player_combo)
        add_layout.addWidget(add_button)
        add_layout.addWidget(remove_button)
        additional_layout.addLayout(add_layout)

        layout.addWidget(additional_group)

        # Dialog buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        self.setLayout(layout)

        # Connect signals
        self.self_radio.toggled.connect(self._on_player_type_changed)
        self.opponent_radio.toggled.connect(self._on_player_type_changed)

        # Initial state
        self._on_player_type_changed()
        self._load_existing_players()

    def _on_player_type_changed(self):
        """Handle player type radio button changes."""
        is_opponent = self.opponent_radio.isChecked()
        self.opponent_widget.setVisible(is_opponent)

    def _on_add_player(self):
        """Add selected player to the additional players list."""
        player_name = self.add_player_combo.currentText().strip()
        if player_name and player_name not in [
            self.players_list.item(i).text() for i in range(self.players_list.count())
        ]:
            self.players_list.addItem(player_name)
            self.add_player_combo.setCurrentText("")

    def _on_remove_players(self):
        """Remove selected players from the list."""
        for item in self.players_list.selectedItems():
            self.players_list.takeItem(self.players_list.row(item))

    def _load_existing_players(self):
        """Load existing players from the database."""
        try:
            # Get all non-self players from database
            players = (
                self.video_manager.session.query(Player).filter_by(is_self=False).all()
            )

            # Add to combo boxes
            for player in players:
                display_text = (
                    f"{player.name} ({player.team})" if player.team else player.name
                )
                self.opponent_combo.addItem(display_text, player)
                self.add_player_combo.addItem(display_text, player)

        except Exception as e:
            QMessageBox.warning(
                self, "Database Error", f"Failed to load existing players: {str(e)}"
            )

    def get_selected_players(self) -> List[Dict[str, Any]]:
        """Get the selected players in database-compatible format."""
        players = []

        # Add primary player
        if self.self_radio.isChecked():
            players.append(
                {
                    "name": "Self",
                    "team": None,
                    "is_self": True,
                    "gamertag": None,
                    "is_primary": True,
                }
            )
        else:
            # Get opponent details
            name = self.opponent_combo.currentText().split(" (")[0].strip()
            team = self.team_input.text().strip()
            gamertag = self.gamertag_input.text().strip() or None

            players.append(
                {
                    "name": name,
                    "team": team,
                    "is_self": False,
                    "gamertag": gamertag,
                    "is_primary": True,
                }
            )

        # Add additional players
        for i in range(self.players_list.count()):
            item = self.players_list.item(i)
            name = item.text().split(" (")[0].strip()

            # Try to get player from database
            player = (
                self.video_manager.session.query(Player).filter_by(name=name).first()
            )

            players.append(
                {
                    "name": name,
                    "team": player.team if player else None,
                    "is_self": False,
                    "gamertag": player.gamertag if player else None,
                    "is_primary": False,
                }
            )

        return players

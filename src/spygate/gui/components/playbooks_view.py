"""
Spygate - Playbooks View Component
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class PlaybookCard(QFrame):
    """Card widget for displaying playbook information."""

    clicked = pyqtSignal()

    def __init__(self, playbook_name, description, parent=None):
        super().__init__(parent)
        self.setObjectName("playbook_card")
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setup_ui(playbook_name, description)

    def setup_ui(self, playbook_name, description):
        """Set up the card UI."""
        layout = QVBoxLayout(self)

        # Playbook name
        name_label = QLabel(playbook_name)
        name_label.setObjectName("playbook_name")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setObjectName("playbook_description")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(desc_label)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        super().mousePressEvent(event)
        self.clicked.emit()


class PlaybooksView(QWidget):
    """View component for managing playbooks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the playbooks view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Game selection
        game_layout = QHBoxLayout()
        game_label = QLabel("Game Version:")
        game_label.setObjectName("section_label")
        self.game_combo = QComboBox()
        self.game_combo.setObjectName("game_selector")
        self.game_combo.addItem("Madden NFL 25")  # Default game
        game_layout.addWidget(game_label)
        game_layout.addWidget(self.game_combo)
        game_layout.addStretch()
        layout.addLayout(game_layout)

        # Player filter
        player_layout = QHBoxLayout()
        player_label = QLabel("Filter by Player:")
        player_label.setObjectName("section_label")
        self.player_combo = QComboBox()
        self.player_combo.setObjectName("player_selector")
        self.player_combo.addItem("All Players")
        self.player_combo.addItem("Self")
        player_layout.addWidget(player_label)
        player_layout.addWidget(self.player_combo)
        player_layout.addStretch()
        layout.addLayout(player_layout)

        # Playbooks grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        playbooks_widget = QWidget()
        self.playbooks_grid = QGridLayout(playbooks_widget)
        self.playbooks_grid.setContentsMargins(0, 0, 0, 0)
        self.playbooks_grid.setSpacing(20)

        scroll_area.setWidget(playbooks_widget)
        layout.addWidget(scroll_area)

        # Add some example playbooks
        self.add_example_playbooks()

        # Connect signals
        self.game_combo.currentTextChanged.connect(self.on_game_changed)
        self.player_combo.currentTextChanged.connect(self.on_player_changed)

    def add_example_playbooks(self):
        """Add example playbooks to the grid."""
        playbooks = [
            (
                "West Coast Offense",
                "Classic west coast offensive playbook with quick passes and timing routes.",
            ),
            (
                "Run 'n' Gun",
                "Aggressive offensive playbook focused on deep passes and misdirection runs.",
            ),
            (
                "4-3 Defense",
                "Traditional 4-3 defensive playbook with multiple coverage schemes.",
            ),
            (
                "3-4 Defense",
                "Versatile 3-4 defensive playbook with complex blitz packages.",
            ),
        ]

        for i, (name, desc) in enumerate(playbooks):
            row = i // 2
            col = i % 2
            card = PlaybookCard(name, desc)
            card.clicked.connect(lambda n=name: self.on_playbook_selected(n))
            self.playbooks_grid.addWidget(card, row, col)

    def on_game_changed(self, game_version):
        """Handle game version change."""
        # TODO: Update playbooks based on selected game version
        pass

    def on_player_changed(self, player_filter):
        """Handle player filter change."""
        # TODO: Filter playbooks based on selected player
        pass

    def on_playbook_selected(self, playbook_name):
        """Handle playbook selection."""
        # TODO: Implement playbook selection logic
        pass

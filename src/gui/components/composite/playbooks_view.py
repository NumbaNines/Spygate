"""
Spygate - Playbooks View Component
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PlaybookCard(QFrame):
    """A card widget displaying playbook information."""

    clicked = pyqtSignal(str)  # Signal emitted when card is clicked with playbook ID

    def __init__(self, playbook_id, title, description, formation_count, parent=None):
        super().__init__(parent)
        self.playbook_id = playbook_id
        self.title = title
        self.description = description
        self.formation_count = formation_count
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(200)
        self.setStyleSheet(
            """
            PlaybookCard {
                background-color: #2c2c2c;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 12px;
            }
            PlaybookCard:hover {
                background-color: #363636;
                border-color: #4c4c4c;
            }
            QLabel {
                color: #ffffff;
            }
            QLabel#title {
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#description {
                color: #b0b0b0;
            }
            QLabel#stats {
                color: #808080;
                font-size: 12px;
            }
            QPushButton {
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #2563EB;
            }
            QPushButton#delete {
                background: #DC2626;
            }
            QPushButton#delete:hover {
                background: #B91C1C;
            }
        """
        )

        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(self.title)
        title_label.setObjectName("title")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self.description)
        desc_label.setObjectName("description")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Stats
        stats_label = QLabel(f"{self.formation_count} formations")
        stats_label.setObjectName("stats")
        layout.addWidget(stats_label)

        layout.addStretch()

        # Action buttons
        button_layout = QHBoxLayout()

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview_playbook)
        button_layout.addWidget(preview_btn)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_playbook)
        button_layout.addWidget(edit_btn)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_playbook)
        button_layout.addWidget(export_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("delete")
        delete_btn.clicked.connect(self._delete_playbook)
        button_layout.addWidget(delete_btn)

        layout.addLayout(button_layout)

    def mousePressEvent(self, event):
        """Handle mouse press events to emit clicked signal."""
        super().mousePressEvent(event)
        self.clicked.emit(self.playbook_id)

    def _preview_playbook(self):
        """Preview the playbook."""
        # TODO: Implement playbook preview
        QMessageBox.information(
            self,
            "Preview Playbook",
            f"Previewing playbook: {self.title}\n(Coming soon)",
        )

    def _edit_playbook(self):
        """Edit the playbook."""
        # TODO: Implement playbook editor
        QMessageBox.information(
            self, "Edit Playbook", f"Opening editor for: {self.title}\n(Coming soon)"
        )

    def _export_playbook(self):
        """Export the playbook."""
        # TODO: Implement playbook export
        QMessageBox.information(
            self, "Export Playbook", f"Exporting playbook: {self.title}\n(Coming soon)"
        )

    def _delete_playbook(self):
        """Delete the playbook."""
        confirm = QMessageBox.question(
            self,
            "Delete Playbook",
            f"Are you sure you want to delete '{self.title}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            # TODO: Implement actual deletion
            self.deleteLater()


class PlaybooksView(QWidget):
    """View for browsing and managing playbooks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_example_playbooks()

    def _setup_ui(self):
        """Set up the playbooks view UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header with title and search
        header_layout = QHBoxLayout()

        title = QLabel("Playbooks")
        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: #D1D5DB;
        """
        )
        header_layout.addWidget(title)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search playbooks...")
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background: #2A2A2A;
                color: #D1D5DB;
                border: 2px solid #3B82F6;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
        """
        )
        self.search_bar.textChanged.connect(self._filter_playbooks)
        header_layout.addWidget(self.search_bar)

        # New playbook button
        new_btn = QPushButton("New Playbook")
        new_btn.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )
        new_btn.clicked.connect(self._create_new_playbook)
        header_layout.addWidget(new_btn)

        main_layout.addLayout(header_layout)

        # Controls bar
        controls_layout = QHBoxLayout()

        # Game version selector
        game_label = QLabel("Game Version:")
        game_label.setStyleSheet("color: #ffffff;")
        self.game_combo = QComboBox()
        self.game_combo.addItems(
            [
                "Madden NFL 25",
                "Madden NFL 24",
                "Madden NFL 23",
                "Madden NFL 22",
            ]
        )
        self.game_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px;
                min-width: 150px;
            }
            QComboBox:hover {
                background-color: #363636;
                border-color: #4c4c4c;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(resources/icons/down_arrow.png);
            }
        """
        )
        controls_layout.addWidget(game_label)
        controls_layout.addWidget(self.game_combo)

        controls_layout.addStretch()

        # Player filter
        player_label = QLabel("Show:")
        player_label.setStyleSheet("color: #ffffff;")
        self.player_combo = QComboBox()
        self.player_combo.addItems(["All Players", "Self", "Opponent"])
        self.player_combo.setStyleSheet(self.game_combo.styleSheet())
        controls_layout.addWidget(player_label)
        controls_layout.addWidget(self.player_combo)

        main_layout.addLayout(controls_layout)

        # Scroll area for playbook grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #2c2c2c;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #4c4c4c;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """
        )

        # Container widget for grid
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(20)
        scroll.setWidget(container)

        main_layout.addWidget(scroll)

    def _load_example_playbooks(self):
        """Load example playbooks into the grid."""
        example_playbooks = [
            {
                "id": "wco",
                "title": "West Coast Offense",
                "description": "A balanced offensive system emphasizing short, high-percentage passes to control the ball and set up the run.",
                "formations": 40,
            },
            {
                "id": "rng",
                "title": "Run 'n' Gun",
                "description": "An aggressive offensive scheme featuring spread formations and option plays to keep the defense guessing.",
                "formations": 35,
            },
            {
                "id": "43d",
                "title": "4-3 Defense",
                "description": "A defensive formation with four linemen and three linebackers, offering balanced run and pass defense.",
                "formations": 30,
            },
            {
                "id": "34d",
                "title": "3-4 Defense",
                "description": "A versatile defensive scheme with three linemen and four linebackers for multiple blitz packages.",
                "formations": 32,
            },
        ]

        for i, playbook in enumerate(example_playbooks):
            row = i // 2
            col = i % 2
            card = PlaybookCard(
                playbook["id"],
                playbook["title"],
                playbook["description"],
                playbook["formations"],
            )
            card.clicked.connect(self._on_playbook_selected)
            self.grid_layout.addWidget(card, row, col)

    def _create_new_playbook(self):
        """Create a new playbook."""
        name, ok = QInputDialog.getText(
            self, "New Playbook", "Enter playbook name:", QLineEdit.EchoMode.Normal
        )

        if ok and name:
            # Create a new playbook card with default values
            card = PlaybookCard(
                f"new_{len(self._get_all_playbooks()) + 1}",
                name,
                "New playbook - click Edit to add formations",
                0,
            )
            card.clicked.connect(self._on_playbook_selected)

            # Add to the grid in the next available position
            count = self.grid_layout.count()
            row = count // 2
            col = count % 2
            self.grid_layout.addWidget(card, row, col)

    def _filter_playbooks(self):
        """Filter playbooks based on search text."""
        search_text = self.search_bar.text().lower()

        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and isinstance(item.widget(), PlaybookCard):
                playbook = item.widget()
                matches = (
                    search_text in playbook.title.lower()
                    or search_text in playbook.description.lower()
                )
                playbook.setVisible(matches)

    def _on_playbook_selected(self, playbook_id):
        """Handle playbook selection."""
        # TODO: Implement playbook detail view
        print(f"Selected playbook: {playbook_id}")

    def _get_all_playbooks(self):
        """Get all playbook cards in the grid."""
        playbooks = []
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and isinstance(item.widget(), PlaybookCard):
                playbooks.append(item.widget())
        return playbooks

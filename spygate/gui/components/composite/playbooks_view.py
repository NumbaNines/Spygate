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
        self.title = title  # Store for filtering
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
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """
        )

        self._setup_ui(title, description, formation_count)

    def _setup_ui(self, title, description, formation_count):
        """Set up the card UI."""
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("title")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setObjectName("description")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Stats
        stats_label = QLabel(f"{formation_count} formations")
        stats_label.setObjectName("stats")
        layout.addWidget(stats_label)

        # Action buttons
        button_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(lambda: self._preview_playbook())
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(lambda: self._edit_playbook())
        button_layout.addWidget(preview_btn)
        button_layout.addWidget(edit_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _preview_playbook(self):
        """Show playbook preview."""
        QMessageBox.information(
            self, "Preview Playbook", f"Preview of '{self.title}' (Coming soon)"
        )

    def _edit_playbook(self):
        """Edit playbook."""
        QMessageBox.information(
            self, "Edit Playbook", f"Edit '{self.title}' (Coming soon)"
        )

    def mousePressEvent(self, event):
        """Handle mouse press events to emit clicked signal."""
        super().mousePressEvent(event)
        self.clicked.emit(self.playbook_id)


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

        # Header section
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
                min-width: 200px;
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
        self.game_combo.currentTextChanged.connect(self._on_game_changed)
        controls_layout.addWidget(game_label)
        controls_layout.addWidget(self.game_combo)

        controls_layout.addStretch()

        # Player filter
        player_label = QLabel("Show:")
        player_label.setStyleSheet("color: #ffffff;")
        self.player_combo = QComboBox()
        self.player_combo.addItems(
            [
                "All Players",
                "Self",
                "Opponent",
            ]
        )
        self.player_combo.setStyleSheet(self.game_combo.styleSheet())
        self.player_combo.currentTextChanged.connect(self._on_player_changed)
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
                "player": "Self",
            },
            {
                "id": "rng",
                "title": "Run 'n' Gun",
                "description": "An aggressive offensive scheme featuring spread formations and option plays to keep the defense guessing.",
                "formations": 35,
                "player": "Opponent",
            },
            {
                "id": "43d",
                "title": "4-3 Defense",
                "description": "A defensive formation with four linemen and three linebackers, offering balanced run and pass defense.",
                "formations": 30,
                "player": "Self",
            },
            {
                "id": "34d",
                "title": "3-4 Defense",
                "description": "A versatile defensive scheme with three linemen and four linebackers for multiple blitz packages.",
                "formations": 32,
                "player": "Opponent",
            },
        ]

        self.playbooks = example_playbooks  # Store for filtering
        self._update_playbook_grid()

    def _update_playbook_grid(self):
        """Update the playbook grid based on current filters."""
        # Clear existing items
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Apply filters
        search_text = self.search_bar.text().lower()
        player_filter = self.player_combo.currentText()

        filtered_playbooks = [
            pb
            for pb in self.playbooks
            if (
                search_text in pb["title"].lower()
                or search_text in pb["description"].lower()
            )
            and (player_filter == "All Players" or pb["player"] == player_filter)
        ]

        # Add filtered playbooks to grid
        for i, playbook in enumerate(filtered_playbooks):
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
            # Add new playbook to the list
            new_playbook = {
                "id": f"pb_{len(self.playbooks)}",
                "title": name,
                "description": "New playbook",
                "formations": 0,
                "player": "Self",
            }
            self.playbooks.append(new_playbook)
            self._update_playbook_grid()

    def _filter_playbooks(self):
        """Filter playbooks based on search text."""
        self._update_playbook_grid()

    def _on_game_changed(self, game_version):
        """Handle game version change."""
        # TODO: Load playbooks for selected game version
        QMessageBox.information(
            self, "Game Version", f"Loading playbooks for {game_version} (Coming soon)"
        )

    def _on_player_changed(self, player_filter):
        """Handle player filter change."""
        self._update_playbook_grid()

    def _on_playbook_selected(self, playbook_id):
        """Handle playbook selection."""
        # Find the selected playbook
        playbook = next((pb for pb in self.playbooks if pb["id"] == playbook_id), None)
        if playbook:
            QMessageBox.information(
                self, "Playbook Selected", f"Opening {playbook['title']} (Coming soon)"
            )

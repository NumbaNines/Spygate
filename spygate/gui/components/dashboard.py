"""
Dashboard component for the main application window.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ClipCard(QFrame):
    """A card widget representing a video clip."""

    def __init__(self, title, player_name, tags, parent=None):
        """Initialize the clip card.

        Args:
            title (str): Title of the clip
            player_name (str): Name of the player
            tags (list): List of tags
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        self.title = title
        self.player_name = player_name
        self.tags = tags
        self.setup_ui()

    def setup_ui(self):
        """Set up the card UI."""
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                padding: 12px;
            }
            QLabel {
                color: white;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Player name
        self.player_label = QLabel(self.player_name)
        self.player_label.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(self.player_label)

        # Tags
        tags_layout = QHBoxLayout()
        self.tag_labels = []
        for tag in self.tags:
            tag_label = QLabel(tag)
            tag_label.setStyleSheet(
                """
                background: #3B82F6;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
            """
            )
            tags_layout.addWidget(tag_label)
            self.tag_labels.append(tag_label)
        tags_layout.addStretch()
        layout.addLayout(tags_layout)


class Sidebar(QFrame):
    """Sidebar navigation component."""

    # Navigation signals
    home_clicked = pyqtSignal()
    upload_clicked = pyqtSignal()
    clips_clicked = pyqtSignal()
    analytics_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    gameplan_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the sidebar."""
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the sidebar UI."""
        self.setFixedWidth(250)
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border: none;
            }
            QPushButton {
                color: white;
                text-align: left;
                padding: 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #3B82F6;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)

        # Navigation buttons
        nav_buttons = [
            ("Home", self.home_clicked),
            ("Upload", self.upload_clicked),
            ("Clips", self.clips_clicked),
            ("Analytics", self.analytics_clicked),
            ("Settings", self.settings_clicked),
        ]

        self.nav_buttons = {}
        for text, signal in nav_buttons:
            button = QPushButton(text)
            button.clicked.connect(signal)
            layout.addWidget(button)
            self.nav_buttons[text] = button

        layout.addStretch()

        # Gameplans section
        gameplans_label = QLabel("Gameplans")
        gameplans_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(gameplans_label)

        self.gameplans_tree = QTreeWidget()
        self.gameplans_tree.setHeaderHidden(True)
        self.gameplans_tree.setStyleSheet(
            """
            QTreeWidget {
                background: transparent;
                border: none;
            }
            QTreeWidget::item {
                color: white;
                padding: 8px;
            }
            QTreeWidget::item:hover {
                background: #3B82F6;
                border-radius: 4px;
            }
        """
        )
        self.gameplans_tree.itemClicked.connect(
            lambda item: self.gameplan_selected.emit(item.text(0))
        )
        layout.addWidget(self.gameplans_tree)


class Dashboard(QMainWindow):
    """Main dashboard window."""

    def __init__(self, parent=None):
        """Initialize the dashboard."""
        super().__init__(parent)
        self.clips = []
        self.gameplan_clips = {}
        self.setup_ui()

    def setup_ui(self):
        """Set up the dashboard UI."""
        self.setWindowTitle("Spygate")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background: #1E1E1E;")

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Add sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Create content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        main_layout.addWidget(content_widget)

        # Add search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search clips...")
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background: #2A2A2A;
                color: white;
                padding: 8px;
                border: 2px solid #3B82F6;
                border-radius: 4px;
            }
        """
        )
        self.search_bar.textChanged.connect(self.filter_clips)
        content_layout.addWidget(self.search_bar)

        # Add filters section
        filters_layout = QHBoxLayout()

        # Player filter
        self.player_combo = QComboBox()
        self.player_combo.setStyleSheet(
            """
            QComboBox {
                background: #2A2A2A;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
        """
        )
        self.player_combo.currentTextChanged.connect(self.filter_clips)
        filters_layout.addWidget(self.player_combo)

        # Tags filter
        self.tag_list = QListWidget()
        self.tag_list.setStyleSheet(
            """
            QListWidget {
                background: #2A2A2A;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background: #3B82F6;
            }
        """
        )
        self.tag_list.setMaximumHeight(100)
        self.tag_list.itemSelectionChanged.connect(self.filter_clips)
        filters_layout.addWidget(self.tag_list)

        content_layout.addLayout(filters_layout)

        # Add scrollable clips grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar {
                background: #2A2A2A;
            }
        """
        )

        scroll_widget = QWidget()
        self.grid_layout = QGridLayout(scroll_widget)
        self.grid_layout.setSpacing(16)
        scroll_area.setWidget(scroll_widget)
        content_layout.addWidget(scroll_area)

        # Connect gameplan signals
        self.sidebar.gameplan_selected.connect(self.on_gameplan_selected)

    def add_clip(self, clip_data):
        """Add a new clip to the dashboard.

        Args:
            clip_data (dict): Dictionary containing clip data with keys:
                - title (str): Title of the clip
                - player (str): Player name
                - tags (list): List of tags
        """
        # Create new clip card
        clip = ClipCard(clip_data["title"], clip_data["player"], clip_data["tags"])
        self.clips.append(clip)

        # Add to grid layout
        row = (len(self.clips) - 1) // 3
        col = (len(self.clips) - 1) % 3
        self.grid_layout.addWidget(clip, row, col)

        # Update filters
        if clip_data["player"] not in [
            self.player_combo.itemText(i) for i in range(self.player_combo.count())
        ]:
            self.player_combo.addItem(clip_data["player"])

        for tag in clip_data["tags"]:
            if not self.tag_list.findItems(tag, Qt.MatchFlag.MatchExactly):
                self.tag_list.addItem(tag)

    def filter_clips(self):
        """Filter clips based on search text, player, and tags."""
        search_text = self.search_bar.text().lower()
        selected_player = self.player_combo.currentText()
        selected_tags = [item.text() for item in self.tag_list.selectedItems()]

        for clip in self.clips:
            matches_search = search_text in clip.title.lower()
            matches_player = not selected_player or selected_player == clip.player_name
            matches_tags = not selected_tags or any(
                tag in clip.tags for tag in selected_tags
            )

            clip.setVisible(matches_search and matches_player and matches_tags)

    def create_new_gameplan(self):
        """Create a new gameplan."""
        gameplan = QTreeWidgetItem(self.sidebar.gameplans_tree, ["New Gameplan"])
        self.gameplan_clips["New Gameplan"] = []
        return gameplan

    def add_to_gameplan(self, gameplan_name, clip):
        """Add a clip to a gameplan.

        Args:
            gameplan_name (str): Name of the gameplan
            clip (ClipCard): Clip to add
        """
        if gameplan_name not in self.gameplan_clips:
            self.gameplan_clips[gameplan_name] = []
        self.gameplan_clips[gameplan_name].append(clip)

    def on_gameplan_selected(self, gameplan_name):
        """Filter clips to show only those in the selected gameplan.

        Args:
            gameplan_name (str): Name of the selected gameplan
        """
        if gameplan_name not in self.gameplan_clips:
            return

        for clip in self.clips:
            clip.setVisible(clip in self.gameplan_clips[gameplan_name])

    def clear_gameplan_filter(self):
        """Clear gameplan filtering and show all clips."""
        for clip in self.clips:
            clip.setVisible(True)

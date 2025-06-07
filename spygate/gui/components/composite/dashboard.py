"""
Spygate - Dashboard Component
Main interface for clip management and navigation
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..clip_card import ClipCard
from ..sidebar import Sidebar


class Dashboard(QWidget):
    """Main dashboard interface for Spygate."""

    # Signals
    clip_selected = pyqtSignal(dict)  # Emitted when a clip is selected

    def __init__(self):
        super().__init__()
        self.clips = []
        self.gameplans = []
        self.gameplan_clips = {}
        self.current_gameplan = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dashboard UI components."""
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Add sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Create content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(content_widget)

        # Add header with search and filters
        header_layout = QHBoxLayout()

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search clips by title or content...")
        self.search_bar.textChanged.connect(self.filter_clips)
        header_layout.addWidget(self.search_bar)

        # Player filter
        self.player_combo = QComboBox()
        self.player_combo.addItem("All Players")
        self.player_combo.currentTextChanged.connect(self.filter_clips)
        header_layout.addWidget(self.player_combo)

        content_layout.addLayout(header_layout)

        # Create filters section
        filters_layout = QHBoxLayout()

        # Tags filter
        tags_frame = QFrame()
        tags_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        tags_layout = QVBoxLayout(tags_frame)
        tags_label = QLabel("Tags")
        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.tag_list.itemSelectionChanged.connect(self.filter_clips)
        tags_layout.addWidget(tags_label)
        tags_layout.addWidget(self.tag_list)
        filters_layout.addWidget(tags_frame)

        # Gameplans section
        gameplans_frame = QFrame()
        gameplans_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        gameplans_layout = QVBoxLayout(gameplans_frame)
        gameplans_label = QLabel("Gameplans")
        self.gameplan_list = QListWidget()
        self.gameplan_list.itemClicked.connect(self._on_gameplan_selected)
        gameplans_layout.addWidget(gameplans_label)
        gameplans_layout.addWidget(self.gameplan_list)
        filters_layout.addWidget(gameplans_frame)

        content_layout.addLayout(filters_layout)

        # Create scrollable clip grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        self.grid_layout = QGridLayout(scroll_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(20)

        scroll_area.setWidget(scroll_widget)
        content_layout.addWidget(scroll_area)

        # Style components
        self._apply_styles()

    def _apply_styles(self):
        """Apply styles to dashboard components."""
        self.setStyleSheet(
            """
            QWidget {
                background: #1E1E1E;
                color: #FFFFFF;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #3B82F6;
                border-radius: 4px;
                background: #2A2A2A;
                color: #FFFFFF;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #3B82F6;
                border-radius: 4px;
                background: #2A2A2A;
                color: #FFFFFF;
            }
            QListWidget {
                background: #2A2A2A;
                border: 1px solid #3B82F6;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #D1D5DB;
                font-weight: bold;
            }
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                padding: 10px;
            }
        """
        )

    def add_clip(self, clip_data):
        """Add a new clip to the dashboard.

        Args:
            clip_data (dict): Data for the clip including title, player_name, and tags
        """
        # Create clip card
        clip = ClipCard(clip_data)
        clip.clicked.connect(lambda: self.clip_selected.emit(clip_data))
        self.clips.append(clip)

        # Update player filter if needed
        player_name = clip_data.get("player_name", "Unknown")
        if player_name not in [
            self.player_combo.itemText(i) for i in range(self.player_combo.count())
        ]:
            self.player_combo.addItem(player_name)

        # Update tags
        for tag in clip_data.get("tags", []):
            if self.tag_list.findItems(tag, Qt.MatchFlag.MatchExactly) == []:
                self.tag_list.addItem(tag)

        # Add to gameplan if specified
        gameplan = clip_data.get("gameplan")
        if gameplan:
            if gameplan not in self.gameplans:
                self.gameplans.append(gameplan)
                self.gameplan_list.addItem(gameplan)
                self.gameplan_clips[gameplan] = []
            self.gameplan_clips[gameplan].append(clip)

        # Refresh display
        self.filter_clips()

    def filter_clips(self):
        """Filter clips based on search text, player, tags, and gameplan."""
        search_text = self.search_bar.text().lower()
        selected_player = self.player_combo.currentText()
        selected_tags = {item.text() for item in self.tag_list.selectedItems()}

        # Hide all clips first
        for clip in self.clips:
            clip.hide()

        # Show clips that match all filters
        row = col = 0
        clips_to_show = []

        for clip in self.clips:
            # Check if clip matches search text (in title or gameplan)
            title_match = search_text in clip.title.lower()
            gameplan_match = any(
                search_text in gameplan.lower()
                for gameplan in self.gameplans
                if clip in self.gameplan_clips[gameplan]
            )

            # Check if clip matches player filter
            player_match = (
                selected_player == "All Players"
                or not selected_player
                or selected_player == clip.player_name
            )

            # Check if clip matches selected tags (match any selected tag)
            tags_match = not selected_tags or any(
                tag in clip.tags for tag in selected_tags
            )

            # Check if clip matches selected gameplan
            gameplan_match = (
                True
                if self.current_gameplan is None
                else clip in self.gameplan_clips.get(self.current_gameplan, [])
            )

            # Show clip if it matches all active filters
            if (
                (title_match or gameplan_match)
                and player_match
                and tags_match
                and gameplan_match
            ):
                clips_to_show.append(clip)

        # Add clips to grid in order
        for clip in clips_to_show:
            self.grid_layout.addWidget(clip, row, col)
            clip.show()

            # Update grid position
            col += 1
            if col > 2:  # 3 clips per row
                col = 0
                row += 1

    def _on_gameplan_selected(self, item):
        """Handle gameplan selection."""
        self.current_gameplan = item.text() if item else None
        self.filter_clips()

    def clear_clips(self):
        """Remove all clips from the dashboard."""
        for clip in self.clips:
            clip.deleteLater()
        self.clips.clear()
        self.gameplans.clear()
        self.gameplan_clips.clear()
        self.current_gameplan = None
        self.player_combo.clear()
        self.player_combo.addItem("All Players")
        self.tag_list.clear()
        self.gameplan_list.clear()

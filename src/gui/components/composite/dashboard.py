from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..base.base_components import BaseWidget


class SidebarButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(40)
        self.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 10px;
                border: none;
                border-radius: 5px;
                color: #D1D5DB;
                background: transparent;
            }
            QPushButton:hover {
                background: #3B82F6;
                color: white;
            }
        """
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class GameplanTreeWidget(QTreeWidget):
    gameplan_selected = pyqtSignal(str)
    clip_selected = pyqtSignal(str, str)  # gameplan_name, clip_title

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.DropOnly)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setStyleSheet(
            """
            QTreeWidget {
                background: #2A2A2A;
                color: #D1D5DB;
                border: none;
                padding: 5px;
            }
            QTreeWidget::item {
                padding: 5px;
                border-radius: 4px;
            }
            QTreeWidget::item:selected {
                background: #3B82F6;
                color: white;
            }
            QTreeWidget::item:hover {
                background: #4B4B4B;
            }
        """
        )
        self.itemClicked.connect(self._on_item_clicked)

    def _on_item_clicked(self, item):
        if item.parent() is None:  # Gameplan item
            # Toggle expansion state
            item.setExpanded(not item.isExpanded())
            # Emit selection signal
            self.gameplan_selected.emit(item.text(0))
            # Ensure the item is selected
            self.setCurrentItem(item)
        else:  # Clip item
            # Emit clip selection signal with parent gameplan name
            self.clip_selected.emit(item.parent().text(0), item.text(0))
            self.setCurrentItem(item)

    def add_gameplan(self, name):
        item = QTreeWidgetItem([name])
        self.addTopLevelItem(item)
        return item

    def add_clip_to_gameplan(self, gameplan_item, clip_title):
        clip_item = QTreeWidgetItem([clip_title])
        gameplan_item.addChild(clip_item)
        return clip_item

    def get_gameplan_item(self, gameplan_name):
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.text(0) == gameplan_name:
                return item
        return None

    def _show_context_menu(self, position):
        item = self.itemAt(position)
        if item and item.parent() is None:  # Only show menu for gameplan items
            menu = QMenu()
            rename_action = menu.addAction("Rename Gameplan")
            delete_action = menu.addAction("Delete Gameplan")

            action = menu.exec(self.viewport().mapToGlobal(position))

            if action == rename_action:
                self.window().rename_gameplan(item.text(0))
            elif action == delete_action:
                self.window().delete_gameplan(item.text(0))


class ClipCard(QFrame):
    def __init__(self, title, player_name, tags, parent=None):
        super().__init__(parent)
        self.title = title
        self.player_name = player_name
        self.tags = tags
        self.setAcceptDrops(True)

        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                padding: 10px;
            }
            QFrame:hover {
                background: #3B3B3B;
            }
        """
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)

        # Thumbnail placeholder
        thumbnail = QFrame()
        thumbnail.setFixedSize(320, 180)  # 16:9 aspect ratio
        thumbnail.setStyleSheet("background: #1E1E1E; border-radius: 4px;")
        layout.addWidget(thumbnail)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Player name
        player_label = QLabel(player_name)
        player_label.setStyleSheet("color: #D1D5DB; font-size: 12px;")
        layout.addWidget(player_label)

        # Tags
        tags_layout = QHBoxLayout()
        for tag in tags:
            tag_label = QLabel(tag)
            tag_label.setStyleSheet(
                """
                background: #3B82F6;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 11px;
            """
            )
            tags_layout.addWidget(tag_label)
        tags_layout.addStretch()
        layout.addLayout(tags_layout)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        add_to_gameplan = menu.addMenu("Add to Gameplan")

        # Get gameplans from parent dashboard
        dashboard = self.window()
        if hasattr(dashboard, "gameplans"):
            for gameplan in dashboard.gameplans:
                action = add_to_gameplan.addAction(gameplan)
                action.triggered.connect(
                    lambda checked, g=gameplan: dashboard.add_to_gameplan(g, self)
                )

        menu.exec(event.globalPos())


class Sidebar(QFrame):
    # Add signals
    home_clicked = pyqtSignal()
    upload_clicked = pyqtSignal()
    clips_clicked = pyqtSignal()
    analytics_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setFixedWidth(250)
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border: none;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Navigation buttons
        nav_buttons = [
            ("Home", "Home", self.home_clicked),
            ("Upload", "Upload Video", self.upload_clicked),
            ("Clips", "View All Clips", self.clips_clicked),
            ("Analytics", "Analytics", self.analytics_clicked),
            ("Settings", "Settings", self.settings_clicked),
        ]

        for name, tooltip, signal in nav_buttons:
            btn = SidebarButton(name)
            btn.setToolTip(tooltip)
            btn.clicked.connect(signal)
            layout.addWidget(btn)

        layout.addSpacing(20)

        # Gameplans section
        gameplans_label = QLabel("Gameplans")
        gameplans_label.setStyleSheet(
            """
            QLabel {
                color: #D1D5DB;
                padding: 10px;
                font-weight: bold;
            }
        """
        )
        layout.addWidget(gameplans_label)

        # Add gameplan tree
        self.gameplans_tree = GameplanTreeWidget()
        layout.addWidget(self.gameplans_tree)

        # Add gameplan button
        add_gameplan_btn = QPushButton("+ New Gameplan")
        add_gameplan_btn.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                color: #D1D5DB;
                padding: 10px;
                text-align: left;
                border: none;
                margin: 0 10px;
            }
            QPushButton:hover {
                color: #3B82F6;
            }
        """
        )
        add_gameplan_btn.clicked.connect(lambda: self.window().create_new_gameplan())
        layout.addWidget(add_gameplan_btn)

        layout.addStretch()


class TagListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.setStyleSheet(
            """
            QListWidget {
                background: #2A2A2A;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px;
                max-height: 100px;
            }
            QListWidget::item {
                padding: 3px;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background: #3B82F6;
                color: white;
            }
            QListWidget::item:hover {
                background: #4B4B4B;
            }
        """
        )


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate")
        self.resize(1200, 800)

        # Initialize data structures
        self.clips = []
        self.selected_tags = set()
        self.current_gameplan = None
        self.gameplans = []
        self.gameplan_clips = {}

        self.setup_ui()
        self.load_sample_data()

    def setup_ui(self):
        # Set dark theme
        self.setStyleSheet("background: #1E1E1E;")

        # Create central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # Add sidebar
        self.sidebar = Sidebar(self)
        self.sidebar.gameplans_tree.gameplan_selected.connect(self.on_gameplan_selected)
        self.sidebar.gameplans_tree.clip_selected.connect(self.on_clip_selected)
        main_layout.addWidget(self.sidebar)

        # Create content area
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(content_area)

        # Add header with filters and gameplan title
        header_layout = QVBoxLayout()

        # Gameplan title (hidden by default)
        self.gameplan_title = QLabel()
        self.gameplan_title.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """
        )
        self.gameplan_title.hide()
        header_layout.addWidget(self.gameplan_title)

        # Search and filters layout
        filters_layout = QHBoxLayout()

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search clips or gameplans...")
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background: #2A2A2A;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                min-width: 300px;
            }
        """
        )
        self.search_bar.textChanged.connect(self.filter_clips)
        filters_layout.addWidget(self.search_bar)

        filters_layout.addSpacing(20)

        # Player filter
        self.player_combo = QComboBox()
        self.player_combo.setFixedWidth(200)
        self.player_combo.setStyleSheet(
            """
            QComboBox {
                background: #2A2A2A;
                color: white;
                padding: 5px;
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

        # Clear gameplan filter button
        self.clear_gameplan_btn = QPushButton("Show All Clips")
        self.clear_gameplan_btn.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )
        self.clear_gameplan_btn.clicked.connect(self.clear_gameplan_filter)
        self.clear_gameplan_btn.hide()
        filters_layout.addWidget(self.clear_gameplan_btn)

        header_layout.addLayout(filters_layout)
        content_layout.addLayout(header_layout)

        # Add tag list
        self.tag_list = TagListWidget()
        self.tag_list.itemSelectionChanged.connect(self.on_tag_selection_changed)
        content_layout.addWidget(self.tag_list)

        # Add scrollable grid for clips
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(20)
        scroll.setWidget(self.grid_widget)

        content_layout.addWidget(scroll)

    def create_new_gameplan(self):
        name, ok = QInputDialog.getText(self, "New Gameplan", "Enter gameplan name:")
        if ok and name:
            self.gameplans.append(name)
            self.gameplan_clips[name] = []
            self._update_gameplans_tree()

    def rename_gameplan(self, old_name):
        new_name, ok = QInputDialog.getText(
            self, "Rename Gameplan", "Enter new gameplan name:", text=old_name
        )
        if ok and new_name and new_name != old_name:
            idx = self.gameplans.index(old_name)
            self.gameplans[idx] = new_name
            self.gameplan_clips[new_name] = self.gameplan_clips.pop(old_name)
            self._update_gameplans_tree()

    def delete_gameplan(self, name):
        confirm = QMessageBox.question(
            self,
            "Delete Gameplan",
            f"Are you sure you want to delete the gameplan '{name}'?\nThis will not delete the clips inside it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self.gameplans.remove(name)
            del self.gameplan_clips[name]
            self._update_gameplans_tree()

    def add_to_gameplan(self, gameplan_name, clip):
        if gameplan_name not in self.gameplan_clips:
            self.gameplan_clips[gameplan_name] = []
        if clip not in self.gameplan_clips[gameplan_name]:
            self.gameplan_clips[gameplan_name].append(clip)
            self._update_gameplans_tree()

    def _update_gameplans_tree(self):
        tree = self.sidebar.gameplans_tree
        tree.clear()
        for gameplan in self.gameplans:
            item = QTreeWidgetItem([gameplan])
            tree.addTopLevelItem(item)
            for clip in self.gameplan_clips.get(gameplan, []):
                clip_item = QTreeWidgetItem([clip.title])
                item.addChild(clip_item)

    def load_sample_data(self):
        # Sample clips data
        sample_clips = [
            {
                "title": "Offensive Play Analysis",
                "player": "Self",
                "tags": ["Offense", "Analysis"],
            },
            {
                "title": "Defensive Strategy Review",
                "player": "Opponent: JohnDoe",
                "tags": ["Defense", "Strategy"],
            },
            {
                "title": "Tournament Highlights",
                "player": "Self",
                "tags": ["Highlights", "Tournament"],
            },
            {
                "title": "Practice Session",
                "player": "Self",
                "tags": ["Practice", "Training"],
            },
            {
                "title": "Match Analysis",
                "player": "Opponent: ProGamer",
                "tags": ["Analysis", "Match"],
            },
            {"title": "Team Tactics", "player": "Self", "tags": ["Team", "Tactics"]},
        ]

        # Add sample clips
        row = col = 0
        for clip_data in sample_clips:
            clip = ClipCard(clip_data["title"], clip_data["player"], clip_data["tags"])
            self.grid_layout.addWidget(clip, row, col)
            self.clips.append(clip)

            # Update grid position
            col += 1
            if col > 2:  # 3 clips per row
                col = 0
                row += 1

        # Add unique tags to tag list
        all_tags = set()
        for clip_data in sample_clips:
            all_tags.update(clip_data["tags"])
        for tag in sorted(all_tags):
            self.tag_list.addItem(tag)

        # Add sample players to combo
        players = {"All Players", "Self", "Opponent: JohnDoe", "Opponent: ProGamer"}
        self.player_combo.addItems(sorted(players))

        # Add sample gameplans
        self.gameplans = ["Tournament Prep", "Strategy Review", "Training"]
        self.gameplan_clips = {gameplan: [] for gameplan in self.gameplans}
        self._update_gameplans_tree()

    def on_tag_selection_changed(self):
        self.selected_tags = {item.text() for item in self.tag_list.selectedItems()}
        self.filter_clips()

    def on_gameplan_selected(self, gameplan_name):
        # Get the gameplan item
        gameplan_item = self.sidebar.gameplans_tree.get_gameplan_item(gameplan_name)
        if gameplan_item:
            # Expand the gameplan to show its clips
            gameplan_item.setExpanded(True)
            # Select the gameplan
            self.sidebar.gameplans_tree.setCurrentItem(gameplan_item)

        # Update the UI to show only clips from this gameplan
        self.current_gameplan = gameplan_name
        self.gameplan_title.setText(f"Gameplan: {gameplan_name}")
        self.gameplan_title.show()
        self.clear_gameplan_btn.show()
        self.filter_clips()

    def on_clip_selected(self, gameplan_name, clip_title):
        # First select the gameplan
        self.on_gameplan_selected(gameplan_name)

        # Then find and scroll to the clip in the main view
        for clip in self.clips:
            if clip.title == clip_title and clip in self.gameplan_clips.get(gameplan_name, []):
                # Find the clip's position in the grid
                for i in range(self.grid_layout.count()):
                    widget = self.grid_layout.itemAt(i).widget()
                    if widget == clip:
                        # Ensure the clip is visible
                        clip.show()
                        # Scroll to the clip (if we have a scroll area)
                        scroll_area = self.findChild(QScrollArea)
                        if scroll_area:
                            scroll_area.ensureWidgetVisible(clip)
                        break

    def clear_gameplan_filter(self):
        self.current_gameplan = None
        self.gameplan_title.hide()
        self.clear_gameplan_btn.hide()
        self.filter_clips()

    def filter_clips(self):
        search_text = self.search_bar.text().lower()
        selected_player = self.player_combo.currentText()

        # Hide all clips first
        for clip in self.clips:
            clip.hide()

        # Show clips that match all filters
        row = col = 0
        clips_to_show = []

        for clip in self.clips:
            # Check if clip matches search text
            title_match = search_text in clip.title.lower()
            gameplan_match = any(
                search_text in gameplan.lower()
                for gameplan in self.gameplans
                if clip in self.gameplan_clips[gameplan]
            )

            # Check if clip matches player filter
            player_match = selected_player == "All Players" or selected_player == clip.player_name

            # Check if clip matches selected tags
            tags_match = not self.selected_tags or any(
                tag in clip.tags for tag in self.selected_tags
            )

            # Check if clip matches selected gameplan
            gameplan_match = (
                True
                if self.current_gameplan is None
                else clip in self.gameplan_clips.get(self.current_gameplan, [])
            )

            if (title_match or gameplan_match) and player_match and tags_match and gameplan_match:
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

    def add_clip_to_gameplan(self, clip, gameplan_name):
        if gameplan_name not in self.gameplan_clips:
            self.gameplan_clips[gameplan_name] = []
            self.gameplans.append(gameplan_name)
            gameplan_item = self.sidebar.gameplans_tree.add_gameplan(gameplan_name)

        if clip not in self.gameplan_clips[gameplan_name]:
            self.gameplan_clips[gameplan_name].append(clip)
            # Find the gameplan item and add the clip
            gameplan_item = self.sidebar.gameplans_tree.get_gameplan_item(gameplan_name)
            if gameplan_item:
                self.sidebar.gameplans_tree.add_clip_to_gameplan(gameplan_item, clip.title)

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

        # Add to grid layout
        row = col = 0
        # Find first empty cell
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item is None or item.widget() is None:
                row = i // 3  # 3 clips per row
                col = i % 3
                break
            else:
                row = (i + 1) // 3
                col = (i + 1) % 3

        self.grid_layout.addWidget(clip, row, col)
        self.clips.append(clip)

        # Add new tags to tag list
        current_tags = {self.tag_list.item(i).text() for i in range(self.tag_list.count())}
        for tag in clip_data["tags"]:
            if tag not in current_tags:
                self.tag_list.addItem(tag)
                current_tags.add(tag)

        # Add player to combo if not exists
        current_players = {self.player_combo.itemText(i) for i in range(self.player_combo.count())}
        if clip_data["player"] not in current_players:
            self.player_combo.addItem(clip_data["player"])

        # Refresh the view
        self.filter_clips()

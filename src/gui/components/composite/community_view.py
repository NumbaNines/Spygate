"""
Spygate - Community View Component
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .card import Card


class SharedPlaybookCard(Card):
    """Card widget for displaying a shared playbook."""

    def __init__(self, playbook_data, parent=None):
        super().__init__(parent)
        self.playbook_data = playbook_data
        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        # Title and author
        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()
        title = QLabel(self.playbook_data.get("name", "Untitled Playbook"))
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        author = QLabel(f"by {self.playbook_data.get('author', 'Unknown')}")
        author.setStyleSheet("color: #888888; font-size: 12px;")
        title_layout.addWidget(title)
        title_layout.addWidget(author)

        rating_label = QLabel(f"{self.playbook_data.get('rating', 0.0):.1f}★")
        rating_label.setStyleSheet("color: #ffd700; font-size: 16px; font-weight: bold;")

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(rating_label)

        self.content_layout.addLayout(header_layout)

        # Description
        description = QLabel(self.playbook_data.get("description", "No description available"))
        description.setWordWrap(True)
        description.setStyleSheet("color: #888888;")
        self.content_layout.addWidget(description)

        # Stats row
        stats_layout = QHBoxLayout()
        plays_label = QLabel(f"Plays: {len(self.playbook_data.get('plays', []))}")
        downloads_label = QLabel(f"Downloads: {self.playbook_data.get('downloads', 0)}")
        plays_label.setStyleSheet("color: #666666;")
        downloads_label.setStyleSheet("color: #666666;")
        stats_layout.addWidget(plays_label)
        stats_layout.addStretch()
        stats_layout.addWidget(downloads_label)
        self.content_layout.addLayout(stats_layout)

        # Action buttons
        actions_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet("background-color: #0078d4;")
        download_btn = QPushButton("Download")
        download_btn.setStyleSheet("background-color: #107c10;")
        actions_layout.addWidget(preview_btn)
        actions_layout.addWidget(download_btn)
        self.content_layout.addLayout(actions_layout)


class StreamSetupDialog(QDialog):
    """Dialog for configuring stream recording settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Recording Setup")
        self.setMinimumWidth(400)
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Stream URL
        url_layout = QVBoxLayout()
        url_label = QLabel("Stream URL:")
        url_label.setStyleSheet("color: #ffffff;")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://twitch.tv/channel")
        self.url_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus {
                border-color: #4c4c4c;
            }
        """
        )
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # Recording Method
        method_layout = QVBoxLayout()
        method_label = QLabel("Recording Method:")
        method_label.setStyleSheet("color: #ffffff;")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["OBS Studio", "Streamlink/FFmpeg"])
        self.method_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px;
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
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # Legal Compliance
        legal_frame = QFrame()
        legal_frame.setStyleSheet(
            """
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 10px;
            }
        """
        )
        legal_layout = QVBoxLayout(legal_frame)

        self.attribution_check = QCheckBox("I will provide proper attribution to players")
        self.attribution_check.setStyleSheet("color: #ffffff;")
        legal_layout.addWidget(self.attribution_check)

        self.fair_use_check = QCheckBox("I understand and will comply with fair use guidelines")
        self.fair_use_check.setStyleSheet("color: #ffffff;")
        legal_layout.addWidget(self.fair_use_check)

        layout.addWidget(legal_frame)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.validate_and_accept)
        self.start_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007acc;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
        """
        )

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
        """
        )

        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.attribution_check.stateChanged.connect(self.check_requirements)
        self.fair_use_check.stateChanged.connect(self.check_requirements)
        self.url_input.textChanged.connect(self.check_requirements)

    def check_requirements(self):
        """Enable start button only when all requirements are met."""
        url_valid = bool(self.url_input.text().strip())
        checks_valid = self.attribution_check.isChecked() and self.fair_use_check.isChecked()
        self.start_button.setEnabled(url_valid and checks_valid)

    def validate_and_accept(self):
        """Validate inputs before accepting."""
        url = self.url_input.text().strip()
        if not url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid stream URL.")
            return
        self.accept()


class CommunityView(QWidget):
    """View for managing community features and stream recording."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_example_channels()

    def _setup_ui(self):
        """Set up the community view UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header with actions
        header_layout = QHBoxLayout()

        title_label = QLabel("Community Streams")
        title_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
        """
        )
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        record_button = QPushButton("Record Stream")
        record_button.setIcon(QIcon("resources/icons/record.png"))
        record_button.clicked.connect(self._show_stream_setup)
        record_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007acc;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
        """
        )
        header_layout.addWidget(record_button)

        main_layout.addLayout(header_layout)

        # Channel table
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(4)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Game", "Status", "Actions"])
        self.channel_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.channel_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.channel_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #2c2c2c;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                gridline-color: #3c3c3c;
            }
            QHeaderView::section {
                background-color: #363636;
                color: #ffffff;
                padding: 8px;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #007acc;
            }
        """
        )
        main_layout.addWidget(self.channel_table)

        # Discord widget placeholder
        discord_frame = QFrame()
        discord_frame.setMinimumHeight(200)
        discord_frame.setStyleSheet(
            """
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """
        )
        discord_layout = QVBoxLayout(discord_frame)

        discord_label = QLabel("Discord Widget")
        discord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        discord_label.setStyleSheet("color: #808080;")
        discord_layout.addWidget(discord_label)

        main_layout.addWidget(discord_frame)

    def _load_example_channels(self):
        """Load example channel data into the table."""
        example_channels = [
            {"channel": "MaddenPro", "game": "Madden NFL 24", "status": "Live"},
            {"channel": "NFLStrategist", "game": "Madden NFL 25", "status": "Offline"},
            {"channel": "PlaybookMaster", "game": "Madden NFL 24", "status": "Live"},
        ]

        self.channel_table.setRowCount(len(example_channels))
        for i, channel in enumerate(example_channels):
            # Channel
            channel_item = QTableWidgetItem(channel["channel"])
            channel_item.setFlags(channel_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channel_table.setItem(i, 0, channel_item)

            # Game
            game_item = QTableWidgetItem(channel["game"])
            game_item.setFlags(game_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channel_table.setItem(i, 1, game_item)

            # Status
            status_item = QTableWidgetItem(channel["status"])
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if channel["status"] == "Live":
                status_item.setForeground(Qt.GlobalColor.green)
            else:
                status_item.setForeground(Qt.GlobalColor.gray)
            self.channel_table.setItem(i, 2, status_item)

            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 0, 4, 0)

            watch_button = QPushButton("Watch")
            watch_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #404040;
                    color: #ffffff;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                }
            """
            )
            actions_layout.addWidget(watch_button)

            if channel["status"] == "Live":
                record_button = QPushButton("Record")
                record_button.setStyleSheet(watch_button.styleSheet())
                actions_layout.addWidget(record_button)

            self.channel_table.setCellWidget(i, 3, actions_widget)

    def _show_stream_setup(self):
        """Show the stream setup dialog."""
        dialog = StreamSetupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            url = dialog.url_input.text().strip()
            method = dialog.method_combo.currentText()
            # TODO: Implement actual recording logic
            QMessageBox.information(
                self,
                "Recording Started",
                f"Started recording {url} using {method}.\nThis is a placeholder - implement actual recording logic.",
            )

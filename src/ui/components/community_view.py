"""
Spygate - Community View Component
Manages community sharing and interaction features
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
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


class StreamSetupDialog(QDialog):
    """Dialog for configuring stream recording settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Setup")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Stream URL
        url_layout = QHBoxLayout()
        url_label = QLabel("Stream URL:")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://twitch.tv/username")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)

        # Recording options
        options_label = QLabel("Recording Options:")
        options_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(options_label)

        # OBS Integration
        self.obs_check = QCheckBox("Use OBS Integration")
        self.obs_check.setChecked(True)
        layout.addWidget(self.obs_check)

        # Streamlink Fallback
        self.streamlink_check = QCheckBox("Use Streamlink (Fallback)")
        layout.addWidget(self.streamlink_check)

        # Quality selection
        quality_layout = QHBoxLayout()
        quality_label = QLabel("Quality:")
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            ["Source", "1080p60", "1080p", "720p60", "720p", "480p"]
        )
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Start Recording")
        start_btn.clicked.connect(self.accept)
        start_btn.setStyleSheet(
            """
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
        """
        )
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(start_btn)
        layout.addLayout(button_layout)


class SharedPlaybookItem(QFrame):
    """Individual shared playbook item widget."""

    def __init__(self, title, author, rating, downloads, parent=None):
        super().__init__(parent)
        self.title = title
        self.author = author
        self.rating = rating
        self.downloads = downloads
        self._setup_ui()

    def _setup_ui(self):
        """Set up the shared playbook item UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                padding: 12px;
            }
            QLabel {
                color: #D1D5DB;
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
            QLabel#stats {
                color: #9CA3AF;
                font-size: 12px;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Title and author
        title_layout = QHBoxLayout()

        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(title_label)

        author_label = QLabel(f"by {self.author}")
        author_label.setStyleSheet("color: #9CA3AF;")
        title_layout.addWidget(author_label)

        title_layout.addStretch()

        # Stats
        stats_label = QLabel(f"★ {self.rating:.1f} • {self.downloads} downloads")
        stats_label.setObjectName("stats")
        title_layout.addWidget(stats_label)

        layout.addLayout(title_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview_playbook)
        button_layout.addWidget(preview_btn)

        download_btn = QPushButton("Download")
        download_btn.clicked.connect(self._download_playbook)
        button_layout.addWidget(download_btn)

        button_layout.addStretch()

        # Rating button
        rate_btn = QPushButton("Rate")
        rate_btn.clicked.connect(self._rate_playbook)
        button_layout.addWidget(rate_btn)

        layout.addLayout(button_layout)

    def _preview_playbook(self):
        """Show playbook preview."""
        # TODO: Implement playbook preview
        QMessageBox.information(
            self, "Preview", f"Preview of '{self.title}' (Coming soon)"
        )

    def _download_playbook(self):
        """Download the playbook."""
        # TODO: Implement actual download
        QMessageBox.information(
            self, "Download", f"Downloading '{self.title}' (Coming soon)"
        )

    def _rate_playbook(self):
        """Rate the playbook."""
        # TODO: Implement rating system
        QMessageBox.information(
            self, "Rate", f"Rating system for '{self.title}' (Coming soon)"
        )


class CommunityView(QWidget):
    """Main community interaction view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_example_channels()

    def _setup_ui(self):
        """Set up the community view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header section
        header_layout = QHBoxLayout()

        title = QLabel("Community")
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
        self.search_bar.setPlaceholderText("Search shared playbooks...")
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

        # Sort dropdown
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(
            ["Most Popular", "Highest Rated", "Most Recent", "Most Downloads"]
        )
        self.sort_combo.setStyleSheet(
            """
            QComboBox {
                background: #2A2A2A;
                color: #D1D5DB;
                border: 2px solid #3B82F6;
                border-radius: 4px;
                padding: 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
        """
        )
        self.sort_combo.currentTextChanged.connect(self._sort_playbooks)
        header_layout.addWidget(self.sort_combo)

        # Share button
        share_btn = QPushButton("Share Playbook")
        share_btn.setStyleSheet(
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
        share_btn.clicked.connect(self._share_playbook)
        header_layout.addWidget(share_btn)

        # Record Stream button
        record_btn = QPushButton("Record Stream")
        record_btn.setIcon(QIcon("resources/icons/record.png"))
        record_btn.clicked.connect(self._show_stream_setup)
        record_btn.setStyleSheet(
            """
            QPushButton {
                background: #DC2626;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #B91C1C;
            }
        """
        )
        header_layout.addWidget(record_btn)

        layout.addLayout(header_layout)

        # Channels section
        channels_label = QLabel("Active Channels")
        channels_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            color: #D1D5DB;
            margin-top: 20px;
        """
        )
        layout.addWidget(channels_label)

        # Channel table
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(4)
        self.channel_table.setHorizontalHeaderLabels(
            ["Channel", "Game", "Status", "Actions"]
        )
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
                background: #2A2A2A;
                color: #D1D5DB;
                border: none;
                gridline-color: #4B5563;
            }
            QHeaderView::section {
                background: #374151;
                color: #D1D5DB;
                padding: 8px;
                border: none;
            }
            QTableWidget::item {
                padding: 8px;
            }
        """
        )
        layout.addWidget(self.channel_table)

        # Shared playbooks section
        playbooks_label = QLabel("Shared Playbooks")
        playbooks_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            color: #D1D5DB;
            margin-top: 20px;
        """
        )
        layout.addWidget(playbooks_label)

        # Playbooks grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: transparent;
            }
        """
        )

        scroll_content = QWidget()
        self.grid_layout = QVBoxLayout(scroll_content)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(16)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Add some sample shared playbooks
        self._add_sample_playbooks()

    def _add_sample_playbooks(self):
        """Add sample shared playbooks for testing."""
        sample_playbooks = [
            ("Pro Tournament Strategies", "CoachJohn", 4.8, 1250),
            ("Advanced Defense Guide", "DefenseMaster", 4.5, 890),
            ("Winning Formations", "ChampionCoach", 4.7, 2100),
            ("Blitz Tactics Collection", "BlitzKing", 4.3, 750),
            ("Red Zone Plays", "EndZoneHero", 4.6, 1500),
        ]

        for title, author, rating, downloads in sample_playbooks:
            self._add_playbook(title, author, rating, downloads)

    def _add_playbook(self, title, author, rating, downloads):
        """Add a new shared playbook item to the view."""
        playbook = SharedPlaybookItem(title, author, rating, downloads)
        self.grid_layout.addWidget(playbook)

    def _share_playbook(self):
        """Share a playbook with the community."""
        # TODO: Implement playbook sharing
        QMessageBox.information(self, "Share Playbook", "Playbook sharing coming soon!")

    def _filter_playbooks(self):
        """Filter playbooks based on search text."""
        search_text = self.search_bar.text().lower()

        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and isinstance(item.widget(), SharedPlaybookItem):
                playbook = item.widget()
                matches = (
                    search_text in playbook.title.lower()
                    or search_text in playbook.author.lower()
                )
                playbook.setVisible(matches)

    def _sort_playbooks(self, sort_by):
        """Sort playbooks based on selected criteria."""
        # TODO: Implement actual sorting
        QMessageBox.information(
            self, "Sort Playbooks", f"Sorting by {sort_by} (Coming soon)"
        )

    def _show_stream_setup(self):
        """Show the stream setup dialog."""
        dialog = StreamSetupDialog(self)
        if dialog.exec():
            url = dialog.url_input.text()
            quality = dialog.quality_combo.currentText()
            use_obs = dialog.obs_check.isChecked()
            use_streamlink = dialog.streamlink_check.isChecked()

            # TODO: Implement actual stream recording
            QMessageBox.information(
                self,
                "Stream Recording",
                f"Starting recording of {url} at {quality} quality\n"
                f"Using: {'OBS' if use_obs else 'Streamlink' if use_streamlink else 'Unknown'}",
            )

    def _load_example_channels(self):
        """Load example channels into the table."""
        self.channel_table.setRowCount(3)

        # Example channel 1
        self.channel_table.setItem(0, 0, QTableWidgetItem("ProGamer123"))
        self.channel_table.setItem(0, 1, QTableWidgetItem("Madden NFL 24"))
        self.channel_table.setItem(0, 2, QTableWidgetItem("Live"))
        watch_btn = QPushButton("Watch")
        watch_btn.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """
        )
        self.channel_table.setCellWidget(0, 3, watch_btn)

        # Example channel 2
        self.channel_table.setItem(1, 0, QTableWidgetItem("MaddenMaster"))
        self.channel_table.setItem(1, 1, QTableWidgetItem("Madden NFL 24"))
        self.channel_table.setItem(1, 2, QTableWidgetItem("Offline"))
        notify_btn = QPushButton("Notify")
        notify_btn.setStyleSheet(watch_btn.styleSheet())
        self.channel_table.setCellWidget(1, 3, notify_btn)

        # Example channel 3
        self.channel_table.setItem(2, 0, QTableWidgetItem("GameChanger"))
        self.channel_table.setItem(2, 1, QTableWidgetItem("Madden NFL 23"))
        self.channel_table.setItem(2, 2, QTableWidgetItem("Recording"))
        view_btn = QPushButton("View")
        view_btn.setStyleSheet(watch_btn.styleSheet())
        self.channel_table.setCellWidget(2, 3, view_btn)

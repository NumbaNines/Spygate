"""
Spygate - Community View Component
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWebEngineView,
    QWidget,
)


class StreamSetupDialog(QDialog):
    """Dialog for setting up stream recording."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stream Recording Setup")
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Player name input
        player_layout = QHBoxLayout()
        player_label = QLabel("Player Name:")
        self.player_input = QLineEdit()
        self.player_input.setPlaceholderText("Enter opponent's name")
        player_layout.addWidget(player_label)
        player_layout.addWidget(self.player_input)
        layout.addLayout(player_layout)

        # Channel input
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel URL:")
        self.channel_input = QLineEdit()
        self.channel_input.setPlaceholderText("Enter Twitch/YouTube channel URL")
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_input)
        layout.addLayout(channel_layout)

        # Recording method selection
        self.obs_radio = QCheckBox("Use OBS Studio")
        self.streamlink_radio = QCheckBox("Use Streamlink/FFmpeg")
        layout.addWidget(self.obs_radio)
        layout.addWidget(self.streamlink_radio)

        # Legal compliance
        self.compliance_check = QCheckBox(
            "I acknowledge and agree to comply with platform terms of service"
        )
        layout.addWidget(self.compliance_check)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)  # Disabled until compliance checked
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.compliance_check.stateChanged.connect(self.on_compliance_changed)
        self.start_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

    def on_compliance_changed(self, state):
        """Enable/disable start button based on compliance checkbox."""
        self.start_button.setEnabled(state == Qt.CheckState.Checked)


class CommunityView(QWidget):
    """View component for community features and stream recording."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the community view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Discord widget section
        discord_label = QLabel("Community Discussion")
        discord_label.setObjectName("section_label")
        layout.addWidget(discord_label)

        # Discord widget (placeholder - would be QWebEngineView in production)
        discord_placeholder = QLabel("Discord Widget Loading...")
        discord_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        discord_placeholder.setMinimumHeight(300)
        discord_placeholder.setStyleSheet("background-color: #2A2A2A; color: #FFFFFF;")
        layout.addWidget(discord_placeholder)

        # Stream recording section
        stream_label = QLabel("Stream Recording")
        stream_label.setObjectName("section_label")
        layout.addWidget(stream_label)

        # Recording buttons
        button_layout = QHBoxLayout()
        obs_button = QPushButton("Start OBS Capture")
        obs_button.clicked.connect(self.start_obs_recording)
        streamlink_button = QPushButton("Record Stream")
        streamlink_button.clicked.connect(self.start_stream_recording)
        button_layout.addWidget(obs_button)
        button_layout.addWidget(streamlink_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Channel management section
        channels_label = QLabel("Channel Management")
        channels_label.setObjectName("section_label")
        layout.addWidget(channels_label)

        # Channel table
        self.channel_table = QTableWidget(0, 3)  # 0 rows, 3 columns
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Player Name", "Status"])
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.channel_table)

        # Add example channels
        self.add_example_channels()

    def add_example_channels(self):
        """Add example channels to the table."""
        channels = [
            ("twitch.tv/example1", "ProPlayer1", "Active"),
            ("youtube.com/example2", "ProPlayer2", "Offline"),
        ]

        for channel, player, status in channels:
            row = self.channel_table.rowCount()
            self.channel_table.insertRow(row)
            self.channel_table.setItem(row, 0, QTableWidgetItem(channel))
            self.channel_table.setItem(row, 1, QTableWidgetItem(player))
            self.channel_table.setItem(row, 2, QTableWidgetItem(status))

    def start_obs_recording(self):
        """Start OBS Studio recording."""
        dialog = StreamSetupDialog(self)
        dialog.obs_radio.setChecked(True)
        dialog.streamlink_radio.setEnabled(False)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            player_name = dialog.player_input.text()
            if player_name:
                # TODO: Implement OBS recording start
                QMessageBox.information(
                    self,
                    "Recording Started",
                    f"Started OBS recording for player: {player_name}",
                )
            else:
                QMessageBox.warning(self, "Invalid Input", "Please enter a player name.")

    def start_stream_recording(self):
        """Start streamlink/FFmpeg recording."""
        dialog = StreamSetupDialog(self)
        dialog.streamlink_radio.setChecked(True)
        dialog.obs_radio.setEnabled(False)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            player_name = dialog.player_input.text()
            channel_url = dialog.channel_input.text()
            if player_name and channel_url:
                # TODO: Implement streamlink/FFmpeg recording start
                QMessageBox.information(
                    self,
                    "Recording Started",
                    f"Started stream recording for player: {player_name}\nChannel: {channel_url}",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Please enter both player name and channel URL.",
                )

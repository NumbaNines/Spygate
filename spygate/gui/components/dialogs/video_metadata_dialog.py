"""
Dialog for entering and displaying video metadata during import.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....video.metadata import VideoMetadata, format_duration


class VideoMetadataDialog(QDialog):
    """Dialog for entering and displaying video metadata."""

    def __init__(self, metadata: VideoMetadata, parent: Optional[QWidget] = None):
        """
        Initialize the video metadata dialog.

        Args:
            metadata: Extracted video metadata
            parent: Parent widget
        """
        super().__init__(parent)
        self.metadata = metadata
        self.setWindowTitle("Video Metadata")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # File Information Group
        file_group = QGroupBox("File Information")
        file_layout = QFormLayout()

        # File name
        file_name_label = QLabel(self.metadata.file_name)
        file_name_label.setStyleSheet("font-weight: bold;")
        file_layout.addRow("File Name:", file_name_label)

        # File size
        size_mb = round(self.metadata.file_size / 1024 / 1024, 2)
        file_size_label = QLabel(f"{size_mb} MB")
        file_layout.addRow("File Size:", file_size_label)

        # Import date
        import_date_label = QLabel(
            self.metadata.import_date.strftime("%Y-%m-%d %H:%M:%S")
        )
        file_layout.addRow("Import Date:", import_date_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Video Properties Group
        video_group = QGroupBox("Video Properties")
        video_layout = QFormLayout()

        # Duration
        duration_label = QLabel(format_duration(self.metadata.duration))
        video_layout.addRow("Duration:", duration_label)

        # Resolution
        resolution_label = QLabel(f"{self.metadata.width}x{self.metadata.height}")
        video_layout.addRow("Resolution:", resolution_label)

        # Frame rate
        fps_label = QLabel(f"{self.metadata.fps:.2f} fps")
        video_layout.addRow("Frame Rate:", fps_label)

        # Frame count
        frame_count_label = QLabel(str(self.metadata.frame_count))
        video_layout.addRow("Total Frames:", frame_count_label)

        # Codec
        codec_label = QLabel(self.metadata.codec)
        video_layout.addRow("Video Codec:", codec_label)

        # Bitrate (if available)
        if self.metadata.bit_rate is not None:
            bitrate_mbps = round(self.metadata.bit_rate / 1_000_000, 2)
            bitrate_label = QLabel(f"{bitrate_mbps} Mbps")
            video_layout.addRow("Video Bitrate:", bitrate_label)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Audio Properties Group (if has audio)
        if self.metadata.has_audio:
            audio_group = QGroupBox("Audio Properties")
            audio_layout = QFormLayout()

            # Audio codec
            if self.metadata.audio_codec:
                audio_codec_label = QLabel(self.metadata.audio_codec)
                audio_layout.addRow("Audio Codec:", audio_codec_label)

            audio_group.setLayout(audio_layout)
            layout.addWidget(audio_group)

        # Custom Metadata Group
        custom_group = QGroupBox("Additional Information")
        custom_layout = QFormLayout()

        # Game type
        self.game_type = QComboBox()
        self.game_type.addItems(["Singles", "Doubles"])
        custom_layout.addRow("Game Type:", self.game_type)

        # Match date
        self.match_date = QDateEdit()
        self.match_date.setCalendarPopup(True)
        self.match_date.setDate(datetime.now().date())
        custom_layout.addRow("Match Date:", self.match_date)

        # Notes
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Enter any additional notes about the video...")
        custom_layout.addRow("Notes:", self.notes)

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the complete metadata including user-entered information.

        Returns:
            Dict[str, Any]: Combined metadata dictionary
        """
        # Start with extracted metadata
        metadata = self.metadata.to_dict()

        # Add user-entered information
        metadata.update(
            {
                "game_type": self.game_type.currentText(),
                "match_date": self.match_date.date().toString(Qt.DateFormat.ISODate),
                "notes": self.notes.text().strip(),
            }
        )

        return metadata

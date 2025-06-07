"""
Video Import Component for Spygate
"""

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...video.codec_validator import CodecValidator
from ..dialogs.player_identification import PlayerIdentificationDialog
from ..dialogs.upload_progress import UploadProgressDialog


class VideoImport(QWidget):
    """Widget for importing video files with drag-and-drop support."""

    video_imported = pyqtSignal(str, str)  # Signals: (file_path, player_name)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the VideoImport widget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.codec_validator = CodecValidator()
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create drop zone frame
        self.drop_zone = QFrame(self)
        self.drop_zone.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.drop_zone.setMinimumSize(400, 200)
        self.drop_zone.setStyleSheet(
            "QFrame {"
            "   background-color: #1E1E1E;"
            "   border: 2px dashed #3B82F6;"
            "   border-radius: 8px;"
            "}"
        )

        # Drop zone layout
        drop_layout = QVBoxLayout(self.drop_zone)

        # Drop zone label
        drop_label = QLabel("Drag and drop video files here\nor")
        drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_label.setStyleSheet("color: #D1D5DB;")

        # Browse button
        self.browse_button = QPushButton("Browse Files")
        self.browse_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #3B82F6;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px 16px;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #2563EB;"
            "}"
        )
        self.browse_button.clicked.connect(self._browse_files)

        # Add widgets to drop zone
        drop_layout.addWidget(drop_label)
        drop_layout.addWidget(
            self.browse_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # Add drop zone to main layout
        layout.addWidget(self.drop_zone)

        # Enable drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_zone.setStyleSheet(
                "QFrame {"
                "   background-color: #1E1E1E;"
                "   border: 2px dashed #2563EB;"
                "   border-radius: 8px;"
                "}"
            )

    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        self.drop_zone.setStyleSheet(
            "QFrame {"
            "   background-color: #1E1E1E;"
            "   border: 2px dashed #3B82F6;"
            "   border-radius: 8px;"
            "}"
        )

    def dropEvent(self, event: QDropEvent):
        """Handle file drop events."""
        self.drop_zone.setStyleSheet(
            "QFrame {"
            "   background-color: #1E1E1E;"
            "   border: 2px dashed #3B82F6;"
            "   border-radius: 8px;"
            "}"
        )

        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self._process_files(files)

    def _browse_files(self):
        """Open file dialog for video selection."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*.*)",
        )
        if files:
            self._process_files(files)

    def _process_files(self, files: List[str]):
        """Process the selected video files.

        Args:
            files: List of file paths to process
        """
        for file_path in files:
            # Validate file exists
            if not Path(file_path).exists():
                QMessageBox.warning(
                    self, "File Not Found", f"The file {file_path} does not exist."
                )
                continue

            # Validate codec
            if not self.codec_validator.is_valid(file_path):
                QMessageBox.warning(
                    self,
                    "Invalid Codec",
                    f"The file {file_path} uses an unsupported codec. "
                    "Supported codecs: H.264, H.265, VP8, VP9",
                )
                continue

            # Get player identification
            dialog = PlayerIdentificationDialog(self)
            if dialog.exec():
                player_name = dialog.get_player_name()

                # Show upload progress
                progress = UploadProgressDialog(self)
                progress.show()

                # Emit signal for video processing
                self.video_imported.emit(file_path, player_name)

                # Close progress dialog
                progress.accept()

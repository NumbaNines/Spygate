"""Widget for importing video files with player identification."""

import os
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...services.video_service import VideoService
from ...video.codec_validator import CodecValidator, VideoMetadata
from ..dialogs.player_name_dialog import PlayerNameDialog


class VideoImportWidget(QWidget):
    """Widget for importing video files with player identification."""

    # Signals
    import_started = Signal()
    import_progress = Signal(int)  # Progress percentage
    import_completed = Signal(str)  # Video ID
    import_error = Signal(str)  # Error message
    dialog_about_to_show = Signal()  # Emitted before showing player name dialog

    def __init__(self, parent: Optional[QWidget] = None, skip_validation: bool = False):
        """Initialize the widget.

        Args:
            parent: Optional parent widget
            skip_validation: Skip codec validation (for testing)
        """
        super().__init__(parent)
        self.video_service = VideoService()
        self.codec_validator = CodecValidator()
        self.skip_validation = skip_validation
        self.current_dialog = None  # Store current dialog
        self.setup_ui()

    def setup_ui(self):
        """Set up the widget's UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Drag and drop video files here or click to select files.\n"
            "Supported formats: MP4 (H.264)"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Import button
        self.import_btn = QPushButton("Select Files")
        self.import_btn.clicked.connect(self.select_files)
        layout.addWidget(self.import_btn)

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Enable drag & drop
        self.setAcceptDrops(True)

    def select_files(self):
        """Open file dialog to select video files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4);;All Files (*.*)"
        )
        if files:
            self.process_files(files)

    def process_files(self, file_paths: list[str]):
        """Process the selected video files.

        Args:
            file_paths: List of file paths to process
        """
        # Validate files first
        valid_files = []
        for file_path in file_paths:
            try:
                if self.skip_validation:
                    # For testing, create dummy metadata
                    metadata = VideoMetadata(
                        width=1920, height=1080, fps=30.0, duration=60.0, codec="h264"
                    )
                    valid_files.append((file_path, metadata))
                else:
                    metadata = self.codec_validator.validate(file_path)
                    if metadata:
                        valid_files.append((file_path, metadata))
            except Exception as e:
                self.import_error.emit(f"Error validating {os.path.basename(file_path)}: {str(e)}")

        if not valid_files:
            QMessageBox.warning(self, "Import Error", "No valid video files were found to import.")
            return

        # For each valid file, get player identification
        for file_path, metadata in valid_files:
            # Create dialog first
            self.current_dialog = PlayerNameDialog(self)

            # Use QTimer to ensure dialog is created before signal
            QTimer.singleShot(0, self.dialog_about_to_show.emit)

            if self.current_dialog.exec():
                player_name = self.current_dialog.get_player_name()
                if not player_name:
                    QMessageBox.warning(self, "Import Error", "Player name is required.")
                    continue

                try:
                    # Start import
                    self.import_started.emit()
                    self.progress_bar.show()
                    self.progress_bar.setValue(0)

                    # Upload with progress updates
                    def update_progress(value: int):
                        self.progress_bar.setValue(value)
                        self.import_progress.emit(value)

                    video_id, _ = self.video_service.upload_video(
                        file_path, metadata, player_name, update_progress
                    )

                    self.import_completed.emit(str(video_id))
                    self.progress_bar.hide()

                except Exception as e:
                    self.import_error.emit(str(e))
                    self.progress_bar.hide()
            self.current_dialog = None

    def get_current_dialog(self) -> Optional[QDialog]:
        """Get the currently active dialog.

        Returns:
            The current dialog or None if no dialog is active
        """
        return self.current_dialog

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for files.

        Args:
            event: The drag enter event
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop events.

        Args:
            event: The drop event
        """
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(url.toLocalFile())
        self.process_files(file_paths)

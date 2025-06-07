"""Widget for importing video files."""

import os
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...services.video_service import VideoService
from ...video.codec_validator import CodecValidator, VideoMetadata
from ..dialogs.player_name_dialog import PlayerNameDialog
from ..workers.video_import_worker import VideoImportWorker


class VideoImportWidget(QWidget):
    """Widget for importing video files."""

    # Signals
    import_started = Signal()
    import_finished = Signal()
    import_progress = Signal(int)  # Progress percentage
    import_error = Signal(str)  # Error message

    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self.video_service = VideoService()
        self.codec_validator = CodecValidator()
        self.import_worker = None
        self.import_thread = None

        # Create layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add import button
        self.import_button = QPushButton("Select Files to Import")
        self.import_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """
        )
        self.import_button.clicked.connect(self.show_file_dialog)
        layout.addWidget(self.import_button)

        # Add progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #374151;
                text-align: center;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 4px;
            }
        """
        )
        layout.addWidget(self.progress_bar)

    def cleanup_import_thread(self):
        """Clean up the import thread and worker."""
        if self.import_worker:
            self.import_worker.stop()

        if self.import_thread and self.import_thread.isRunning():
            self.import_thread.quit()
            self.import_thread.wait()

        if self.import_worker:
            try:
                self.import_worker.deleteLater()
            except RuntimeError:
                pass  # Object may have been deleted already
            self.import_worker = None

        if self.import_thread:
            try:
                self.import_thread.deleteLater()
            except RuntimeError:
                pass  # Object may have been deleted already
            self.import_thread = None

    def show_file_dialog(self) -> None:
        """Show file selection dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)",
        )

        if files:
            self.import_videos(files)

    def import_videos(self, file_paths: List[str]) -> None:
        """Import selected video files."""
        # Clean up any existing import thread
        self.cleanup_import_thread()

        # Validate files first
        valid_files = []
        for file_path in file_paths:
            is_valid, error_msg, metadata = self.codec_validator.validate_video(
                file_path
            )
            if not is_valid:
                self.import_error.emit(
                    f"Error validating {os.path.basename(file_path)}: {error_msg}"
                )
                continue
            valid_files.append((file_path, metadata))

        if not valid_files:
            return

        # Show player name dialog
        dialog = PlayerNameDialog(self)
        if dialog.exec():
            player_name = dialog.get_player_name()

            # Start import in background thread
            self.import_thread = QThread()
            self.import_worker = VideoImportWorker(
                valid_files, player_name, self.video_service
            )
            self.import_worker.moveToThread(self.import_thread)

            # Connect signals
            self.import_thread.started.connect(self.import_worker.run)
            self.import_worker.finished.connect(self.import_thread.quit)
            self.import_worker.finished.connect(self.import_worker.deleteLater)
            self.import_thread.finished.connect(self.import_thread.deleteLater)
            self.import_worker.progress.connect(self.update_progress)
            self.import_worker.error.connect(self.handle_import_error)

            # Start import
            self.progress_bar.setVisible(True)
            self.import_started.emit()
            self.import_thread.start()

    @Slot(int)
    def update_progress(self, value: int) -> None:
        """Update progress bar value."""
        self.progress_bar.setValue(value)
        if value == 100:
            self.progress_bar.setVisible(False)
            self.import_finished.emit()
            self.cleanup_import_thread()

    @Slot(str)
    def handle_import_error(self, error: str) -> None:
        """Handle import errors."""
        self.progress_bar.setVisible(False)
        self.import_error.emit(error)
        QMessageBox.critical(self, "Import Error", error)
        self.cleanup_import_thread()

    def closeEvent(self, event):
        """Handle widget close event."""
        self.cleanup_import_thread()
        super().closeEvent(event)

"""
Video import widget with drag-and-drop support and file selection dialog.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...database.models import Player
from ...database.video_manager import VideoManager
from ...utils.error_handler import (
    DatabaseError,
    PlayerError,
    StorageError,
    ValidationError,
    cleanup_failed_import,
    handle_import_error,
)
from ...video.metadata import extract_metadata
from ..workers.video_import_worker import ImportProgress, VideoImportWorker
from .dialogs.player_identification_dialog import PlayerIdentificationDialog
from .dialogs.video_metadata_dialog import VideoMetadataDialog

logger = logging.getLogger(__name__)

# Maximum file size (500MB in bytes)
MAX_FILE_SIZE = 500 * 1024 * 1024


class VideoImportWidget(QWidget):
    """Widget for importing video files."""

    # Signals
    video_imported = pyqtSignal(str, list, dict)  # file_path, players, metadata

    def __init__(self, video_manager: Optional[VideoManager] = None):
        """Initialize the widget."""
        super().__init__()
        self.video_manager = video_manager or VideoManager()
        self.import_worker = None
        self.progress_dialog = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create drop frame
        self.drop_frame = QWidget()
        self.drop_frame.setStyleSheet(
            """
            QWidget {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """
        )
        drop_layout = QVBoxLayout()
        self.drop_frame.setLayout(drop_layout)

        # Add label
        self.label = QLabel("Drop video files here\nor click to select")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #666;")
        drop_layout.addWidget(self.label)

        # Add button
        self.select_button = QPushButton("Select Files")
        self.select_button.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )
        self.select_button.clicked.connect(self._show_file_dialog)
        drop_layout.addWidget(self.select_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.drop_frame)

        # Add video list
        self.video_list = QListWidget()
        self.video_list.setStyleSheet(
            """
            QListWidget {
                background: #2a2a2a;
                border: 1px solid #666;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #666;
            }
            QListWidget::item:last {
                border-bottom: none;
            }
        """
        )
        layout.addWidget(self.video_list)

        # Enable drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_frame.setStyleSheet(
                """
                QWidget {
                    border: 2px dashed #3B82F6;
                    border-radius: 8px;
                    background: #2d2d2d;
                }
            """
            )
            self.label.setStyleSheet("color: #3B82F6;")

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.drop_frame.setStyleSheet(
            """
            QWidget {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """
        )
        self.label.setStyleSheet("color: #666;")

    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        video_files = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.toLocalFile().lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        if video_files:
            self._handle_video_files(video_files)

        self.drop_frame.setStyleSheet(
            """
            QWidget {
                border: 2px dashed #666;
                border-radius: 8px;
                background: #2a2a2a;
            }
        """
        )
        self.label.setStyleSheet("color: #666;")

    def _show_file_dialog(self):
        """Show file selection dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )

        if files:
            self._handle_video_files(files)

    def _handle_video_files(self, file_paths: list[str]):
        """Handle video file import process."""
        # Extract metadata first
        files_with_metadata = []
        for file_path in file_paths:
            try:
                metadata = extract_metadata(file_path)
                files_with_metadata.append((file_path, metadata))
            except Exception as e:
                title, message = handle_import_error(e, file_path)
                QMessageBox.warning(self, title, message)
                continue

        if not files_with_metadata:
            return

        # Show player identification dialog
        dialog = PlayerIdentificationDialog(self.video_manager, self)
        if dialog.exec():
            selected_players = dialog.get_selected_players()
            if not selected_players:
                QMessageBox.warning(
                    self,
                    "Player Selection Error",
                    "At least one player must be selected.",
                )
                return

            # Create and start worker thread
            self.import_worker = VideoImportWorker(
                files_with_metadata, selected_players, self.video_manager
            )

            # Connect signals
            self.import_worker.progress.connect(self._update_progress)
            self.import_worker.error.connect(self._handle_error)
            self.import_worker.warning.connect(self._handle_warning)
            self.import_worker.file_started.connect(self._handle_file_started)
            self.import_worker.file_finished.connect(self._handle_file_finished)
            self.import_worker.all_finished.connect(self._handle_all_finished)

            # Create progress dialog
            self.progress_dialog = QProgressDialog(self)
            self.progress_dialog.setWindowTitle("Importing Videos")
            self.progress_dialog.setRange(0, 100)
            self.progress_dialog.setMinimumWidth(400)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)

            # Style the progress dialog
            self.progress_dialog.setStyleSheet(
                """
                QProgressDialog {
                    background-color: #2d2d2d;
                }
                QProgressBar {
                    border: 1px solid #666;
                    border-radius: 4px;
                    background-color: #1a1a1a;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #3B82F6;
                    border-radius: 2px;
                }
            """
            )

            # Connect cancel button
            self.progress_dialog.canceled.connect(self._cancel_import)

            # Start import
            self.progress_dialog.show()
            self.import_worker.start()

    def _update_progress(self, progress: ImportProgress):
        """Update progress dialog."""
        if self.progress_dialog:
            self.progress_dialog.setValue(progress.progress)
            self.progress_dialog.setLabelText(progress.message)

    def _handle_error(self, title: str, message: str):
        """Handle import error."""
        QMessageBox.critical(self, title, message)

    def _handle_warning(self, title: str, message: str):
        """Handle import warning."""
        QMessageBox.warning(self, title, message)

    def _handle_file_started(self, file_name: str):
        """Handle file import start."""
        logger.info(f"Starting import of {file_name}")

    def _handle_file_finished(self, file_name: str, success: bool, message: str):
        """Handle file import completion."""
        if success:
            logger.info(f"Successfully imported {file_name}")
            # Add to list widget
            item = QListWidgetItem(self.video_list)
            item.setText(f"{file_name} - {message}")
            item.setIcon(
                self.style().standardIcon(self.style().StandardPixmap.SP_DialogApplyButton)
            )
            self.video_list.addItem(item)
        else:
            logger.error(f"Failed to import {file_name}: {message}")
            # Add to list widget with error indicator
            item = QListWidgetItem(self.video_list)
            item.setText(f"{file_name} - {message}")
            item.setIcon(
                self.style().standardIcon(self.style().StandardPixmap.SP_DialogCancelButton)
            )
            self.video_list.addItem(item)

    def _handle_all_finished(self):
        """Handle completion of all imports."""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.import_worker = None

    def _cancel_import(self):
        """Cancel the current import process."""
        if self.import_worker:
            self.import_worker.stop()
            self.import_worker = None

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
